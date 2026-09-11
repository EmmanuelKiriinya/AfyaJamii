import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, ApiError, type AccountType, type AuthSession } from "@/lib/api";

/**
 * Holds the signed-in session.
 *
 * The token is kept in localStorage so a refresh does not sign the user out.
 * On load it is verified against the API before being trusted — an expired
 * token used to leave the app looking signed in until the first request
 * failed. The account type now comes from the login response rather than
 * being assumed to be "general".
 */

const STORAGE_KEY = "afyajamii.session";

interface StoredSession {
  token: string;
  username: string;
  accountType: AccountType;
  /** Epoch milliseconds. Checked before the token is used. */
  expiresAt: number;
}

interface AuthContextValue {
  token: string | null;
  username: string | null;
  accountType: AccountType | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  signIn: (session: AuthSession) => void;
  signOut: () => void;
  /**
   * Patch the stored session in place.
   *
   * Needed by account settings: renaming the account changes the username the
   * header shows, and the API returns a replacement token because the old one
   * names the previous username as its subject.
   */
  updateSession: (changes: Partial<Omit<StoredSession, "expiresAt">>) => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function readStoredSession(): StoredSession | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw) as StoredSession;
    if (!parsed.token || !parsed.username) return null;

    // Drop a token that has already expired without troubling the network.
    if (parsed.expiresAt && parsed.expiresAt < Date.now()) return null;

    return parsed;
  } catch {
    // Corrupt or unreadable storage (private mode, cleared data) — start fresh.
    return null;
  }
}

function writeStoredSession(session: StoredSession | null) {
  try {
    if (session) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // Storage can be unavailable; the session simply won't survive a refresh.
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<StoredSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const stored = readStoredSession();

    if (!stored) {
      setIsLoading(false);
      return;
    }

    let cancelled = false;

    // Confirm the token is still good before showing a signed-in interface.
    api
      .me(stored.token)
      .then((profile) => {
        if (cancelled) return;
        setSession({ ...stored, accountType: profile.account_type, username: profile.username });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        // Only discard the session when the server actually rejects it. A
        // network blip should not sign the user out.
        if (error instanceof ApiError && error.isUnauthorized) {
          writeStoredSession(null);
          setSession(null);
        } else {
          setSession(stored);
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback((auth: AuthSession) => {
    const next: StoredSession = {
      token: auth.access_token,
      username: auth.username,
      accountType: auth.account_type,
      expiresAt: Date.now() + auth.expires_in * 1000,
    };
    writeStoredSession(next);
    setSession(next);
  }, []);

  const updateSession = useCallback((changes: Partial<Omit<StoredSession, "expiresAt">>) => {
    setSession((current) => {
      if (!current) return current;
      const next = { ...current, ...changes };
      writeStoredSession(next);
      return next;
    });
  }, []);

  const signOut = useCallback(() => {
    writeStoredSession(null);
    setSession(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      token: session?.token ?? null,
      username: session?.username ?? null,
      accountType: session?.accountType ?? null,
      isAuthenticated: Boolean(session?.token),
      isLoading,
      signIn,
      signOut,
      updateSession,
    }),
    [session, isLoading, signIn, signOut, updateSession],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
