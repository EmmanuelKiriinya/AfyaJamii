/**
 * Shared vocabulary for presenting health data.
 *
 * The risk-label and advice formatting logic was previously duplicated across
 * three components, each with slightly different colours and casing.
 */

import type { IconName } from "@/lib/icons";

export type RiskTone = "low" | "mid" | "high" | "unknown";

interface RiskPresentation {
  tone: RiskTone;
  /** Sentence-case label for display, e.g. "High risk". */
  label: string;
  icon: IconName;
  /** Tailwind classes for a filled badge. */
  badgeClass: string;
  /** Tailwind classes for a tinted panel. */
  panelClass: string;
  /** One line explaining what the label means in practice. */
  meaning: string;
}

const PRESENTATIONS: Record<RiskTone, RiskPresentation> = {
  low: {
    tone: "low",
    label: "Low risk",
    icon: "check",
    badgeClass: "bg-success text-success-foreground",
    panelClass: "border-success/30 bg-success/5",
    meaning: "Your readings look reassuring. Keep to your usual antenatal schedule.",
  },
  mid: {
    tone: "mid",
    label: "Moderate risk",
    icon: "warning",
    badgeClass: "bg-warning text-warning-foreground",
    panelClass: "border-warning/35 bg-warning/5",
    meaning: "Some readings need watching. Mention them at your next clinic visit.",
  },
  high: {
    tone: "high",
    label: "High risk",
    icon: "alert",
    badgeClass: "bg-destructive text-destructive-foreground",
    panelClass: "border-destructive/35 bg-destructive/5",
    meaning: "These readings need prompt attention. Contact a health worker today.",
  },
  unknown: {
    tone: "unknown",
    label: "Not assessed",
    icon: "info",
    badgeClass: "bg-muted text-muted-foreground",
    panelClass: "border-border bg-muted/40",
    meaning: "No assessment is available for this record.",
  },
};

/** Map a raw label from the API onto its presentation. */
export function presentRisk(rawLabel: string): RiskPresentation {
  const label = rawLabel.toLowerCase();
  if (label.includes("high")) return PRESENTATIONS.high;
  if (label.includes("mid") || label.includes("medium") || label.includes("moderate")) {
    return PRESENTATIONS.mid;
  }
  if (label.includes("low")) return PRESENTATIONS.low;
  return PRESENTATIONS.unknown;
}

/** Human-readable names for the model's feature keys. */
const FEATURE_LABELS: Record<string, string> = {
  Age: "Age",
  SystolicBP: "Systolic blood pressure",
  DiastolicBP: "Diastolic blood pressure",
  BS: "Blood sugar",
  BodyTemp: "Body temperature",
  HeartRate: "Heart rate",
};

export function featureLabel(key: string): string {
  return FEATURE_LABELS[key] ?? key.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/_/g, " ");
}

export function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-KE", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString("en-KE", { hour: "2-digit", minute: "2-digit" });
}
