/**
 * Client for the Afya Jamii API.
 *
 * The base URL comes from VITE_API_BASE_URL so the same build can point at a
 * local backend, staging, or production. It was previously hard-coded to a
 * single hosted URL, which made local development impossible without editing
 * source.
 */

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL;

/**
 * True when the build has no API URL configured.
 *
 * This used to `throw` here, at module scope. Because this module is imported
 * by the auth context and therefore by the whole app, a missing environment
 * variable took the entire site down — including the landing page, which needs
 * no API at all. A misconfiguration should degrade the parts that depend on
 * the API, not blank the page, so it is now reported per request instead.
 */
const isUnconfigured = !rawBaseUrl && import.meta.env.PROD;

if (isUnconfigured) {
  console.error(
    "VITE_API_BASE_URL is not set for this build. Static pages will render, " +
      "but anything that talks to the API will fail until it is configured.",
  );
}

const API_BASE_URL = (rawBaseUrl ?? "http://localhost:8000").replace(/\/+$/, "");
const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS ?? 60_000);

export type AccountType = "pregnant" | "postnatal" | "general";
export type TemperatureUnit = "celsius" | "fahrenheit";

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface SignupData {
  username: string;
  email: string;
  account_type: AccountType;
  full_name: string;
  password: string;
}

export interface AuthSession {
  access_token: string;
  token_type: string;
  expires_in: number;
  username: string;
  account_type: AccountType;
}

export interface UserProfile {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  account_type: AccountType;
  created_at: string;
  is_active: boolean;
}

export interface ProfileUpdate {
  username?: string;
  email?: string;
  full_name?: string;
  account_type?: AccountType;
}

/** A profile, plus a replacement token when the username changed. */
export interface ProfileResponse extends UserProfile {
  access_token?: string | null;
}

export interface DeletionSummary {
  detail: string;
  username: string;
  vitals_records_deleted: number;
  conversations_deleted: number;
  deleted_at: string;
}

export interface VitalsData {
  age: number;
  systolic_bp: number;
  diastolic_bp: number;
  bs: number;
  body_temp: number;
  body_temp_unit: TemperatureUnit;
  heart_rate: number;
  patient_history?: string;
}

export interface VitalsSubmitPayload {
  vitals: VitalsData;
  account_type?: AccountType;
}

export interface MLOutput {
  risk_label: string;
  probability: number;
  class_probabilities: Record<string, number>;
  feature_importances: Record<string, number>;
}

export interface Advice {
  advice: string;
  timestamp: string;
  generated: boolean;
}

export interface VitalsSubmitResponse {
  user_id: number;
  submission_id: number;
  timestamp: string;
  ml_output: MLOutput;
  llm_advice: Advice;
}

export interface VitalsRecord {
  id: number;
  age: number;
  systolic_bp: number;
  diastolic_bp: number;
  bs: number;
  body_temp: number;
  body_temp_unit: string;
  patient_history: string | null;
  heart_rate: number;
  ml_risk_label: string;
  ml_probability: number;
  ml_feature_importances: Record<string, number>;
  created_at: string;
}

export interface ConversationRecord {
  id: number;
  vitals_record_id: number | null;
  user_message: string;
  ai_response: string;
  created_at: string;
}

/** An error carrying the HTTP status, so callers can react to 401 and 503. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** The session is gone or invalid; the caller should sign the user out. */
  get isUnauthorized() {
    return this.status === 401;
  }

  /** A dependency is down; the request is worth retrying shortly. */
  get isUnavailable() {
    return this.status === 503;
  }
}

interface ErrorBody {
  detail?: string | { field?: string; message?: string }[];
  request_id?: string;
}

/** Turn the API's error shape into a single sentence a person can act on. */
function describeError(body: ErrorBody | null, status: number): string {
  const detail = body?.detail;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    const problems = detail
      .map((item) => (item.field ? `${item.field}: ${item.message}` : item.message))
      .filter(Boolean);
    if (problems.length > 0) return problems.join("; ");
  }

  if (status === 0) return "Could not reach the server. Check your connection and try again.";
  if (status === 429) return "Too many requests. Please wait a moment before trying again.";
  return `Request failed (${status}).`;
}

interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  token?: string;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, token, signal } = options;

  if (isUnconfigured) {
    throw new ApiError(
      "This site is not yet connected to its health service. Please try again shortly.",
      0,
    );
  }

  // Abort on timeout, but keep honouring a caller-supplied signal too.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  signal?.addEventListener("abort", () => controller.abort(), { once: true });

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: {
        ...(body ? { "Content-Type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch {
    if (controller.signal.aborted) {
      throw new ApiError("The request took too long. Please try again.", 0);
    }
    throw new ApiError("Could not reach the server. Check your connection.", 0);
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    const errorBody = (await response.json().catch(() => null)) as ErrorBody | null;
    throw new ApiError(
      describeError(errorBody, response.status),
      response.status,
      errorBody?.request_id ?? response.headers.get("X-Request-ID") ?? undefined,
    );
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  login: (credentials: LoginCredentials) =>
    request<AuthSession>("/api/v1/auth/login", { method: "POST", body: credentials }),

  signup: (data: SignupData) =>
    request<UserProfile>("/api/v1/auth/signup", { method: "POST", body: data }),

  me: (token: string) => request<UserProfile>("/api/v1/auth/me", { token }),

  submitVitals: (payload: VitalsSubmitPayload, token: string) =>
    request<VitalsSubmitResponse>("/api/v1/vitals/submit", {
      method: "POST",
      body: payload,
      token,
    }),

  chatAdvice: (question: string, token: string) =>
    request<Advice>("/api/v1/chat/advice", { method: "POST", body: { question }, token }),

  getVitalsHistory: (token: string, limit = 10) =>
    request<VitalsRecord[]>(`/api/v1/history/vitals?limit=${limit}`, { token }),

  getConversationsHistory: (token: string, limit = 20) =>
    request<ConversationRecord[]>(`/api/v1/history/conversations?limit=${limit}`, { token }),

  // ── Account settings ────────────────────────────────────────────────────

  getProfile: (token: string) => request<UserProfile>("/api/v1/users/me", { token }),

  updateProfile: (updates: ProfileUpdate, token: string) =>
    request<ProfileResponse>("/api/v1/users/me", { method: "PATCH", body: updates, token }),

  changePassword: (current_password: string, new_password: string, token: string) =>
    request<void>("/api/v1/users/me/password", {
      method: "POST",
      body: { current_password, new_password },
      token,
    }),

  deactivateAccount: (token: string) =>
    request<void>("/api/v1/users/me/deactivate", { method: "POST", token }),

  deleteAccount: (password: string, confirmation: string, token: string) =>
    request<DeletionSummary>("/api/v1/users/me", {
      method: "DELETE",
      body: { password, confirmation },
      token,
    }),
};
