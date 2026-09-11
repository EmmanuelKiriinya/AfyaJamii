import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Icon } from "@/components/Icon";
import { VitalsTrend } from "@/components/VitalsTrend";
import { useAuth } from "@/contexts/AuthContext";
import { useConversations, useVitalsHistory } from "@/lib/queries";
import { formatDateTime, presentRisk } from "@/lib/health";
import { cn } from "@/lib/utils";
import type { VitalsRecord } from "@/lib/api";
import type { IconName } from "@/lib/icons";

/**
 * The dashboard's landing view: what the most recent reading said, how the
 * numbers have moved, and the one or two things worth doing next.
 */

interface OverviewProps {
  onRecord: () => void;
  onAsk: () => void;
}

function greeting() {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

/** Reference ranges, used only to flag a reading as worth a second look. */
function readingTone(key: string, record: VitalsRecord): "normal" | "watch" {
  switch (key) {
    case "bp":
      return record.systolic_bp >= 140 || record.diastolic_bp >= 90 ? "watch" : "normal";
    case "hr":
      return record.heart_rate < 60 || record.heart_rate > 100 ? "watch" : "normal";
    case "bs":
      return record.bs > 7.0 ? "watch" : "normal";
    case "temp": {
      const celsius =
        record.body_temp_unit === "fahrenheit"
          ? (record.body_temp - 32) * (5 / 9)
          : record.body_temp;
      return celsius >= 37.5 ? "watch" : "normal";
    }
    default:
      return "normal";
  }
}

const Overview = ({ onRecord, onAsk }: OverviewProps) => {
  const { username } = useAuth();
  const vitals = useVitalsHistory(20);
  const conversations = useConversations(5);

  const latest = vitals.data?.[0];

  if (vitals.isLoading) {
    return (
      <div className="mx-auto max-w-5xl space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-48 w-full rounded-xl" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28 rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  // First-run state: nothing recorded yet, so the only useful thing to offer
  // is the path to a first reading.
  if (!latest) {
    return (
      <div className="mx-auto max-w-2xl">
        <Card className="border-2 border-primary/10 shadow-lg">
          <CardContent className="flex flex-col items-center px-6 py-14 text-center">
            <div className="mb-5 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/10 to-purple-500/10">
              <Icon name="vitals" size={36} className="text-primary" />
            </div>
            <h2 className="text-2xl font-bold">
              {greeting()}, {username}
            </h2>
            <p className="mt-3 max-w-md leading-relaxed text-muted-foreground">
              You haven&rsquo;t recorded any readings yet. Enter the numbers from your clinic
              card or home monitor and you&rsquo;ll get an assessment straight away.
            </p>
            <Button size="lg" className="mt-7" onClick={onRecord}>
              <Icon name="vitals" size={18} className="mr-2" />
              Record your first reading
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const risk = presentRisk(latest.ml_risk_label);

  const tiles: { key: string; icon: IconName; label: string; value: string; unit: string }[] = [
    {
      key: "bp",
      icon: "bloodPressure",
      label: "Blood pressure",
      value: `${latest.systolic_bp}/${latest.diastolic_bp}`,
      unit: "mmHg",
    },
    { key: "hr", icon: "heartRate", label: "Heart rate", value: String(latest.heart_rate), unit: "bpm" },
    { key: "bs", icon: "bloodSugar", label: "Blood sugar", value: String(latest.bs), unit: "mmol/L" },
    {
      key: "temp",
      icon: "temperature",
      label: "Temperature",
      value: String(latest.body_temp),
      unit: latest.body_temp_unit === "celsius" ? "°C" : "°F",
    },
  ];

  const watchCount = tiles.filter((tile) => readingTone(tile.key, latest) === "watch").length;

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      {/* Greeting */}
      <div>
        <h2 className="text-2xl font-bold sm:text-3xl">
          {greeting()}, {username}
        </h2>
        <p className="mt-1 text-muted-foreground">
          Last reading {formatDateTime(latest.created_at)}
        </p>
      </div>

      {/* Latest assessment */}
      <Card
        className={cn(
          "overflow-hidden border-2 shadow-lg",
          risk.tone === "high" && "border-destructive/30",
          risk.tone === "mid" && "border-warning/30",
          risk.tone === "low" && "border-success/30",
        )}
      >
        <div
          className={cn(
            "px-6 py-5 sm:px-8",
            risk.tone === "high" && "bg-gradient-to-r from-destructive/10 to-destructive/5",
            risk.tone === "mid" && "bg-gradient-to-r from-warning/10 to-warning/5",
            risk.tone === "low" && "bg-gradient-to-r from-success/10 to-success/5",
            risk.tone === "unknown" && "bg-muted/40",
          )}
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/70 shadow-sm dark:bg-card/70">
                <Icon name={risk.icon} size={28} className="text-foreground" />
              </div>
              <div>
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Latest assessment
                </p>
                <p className="text-2xl font-bold">{risk.label}</p>
              </div>
            </div>

            <div className="text-right">
              <p className="text-xs text-muted-foreground">Confidence</p>
              <p className="tabular text-2xl font-bold">
                {Math.round(latest.ml_probability * 100)}%
              </p>
            </div>
          </div>

          <p className="mt-4 leading-relaxed">{risk.meaning}</p>
        </div>

        <CardContent className="flex flex-wrap gap-3 border-t bg-card px-6 py-4 sm:px-8">
          <Button onClick={onRecord}>
            <Icon name="vitals" size={16} className="mr-2" />
            Record a new reading
          </Button>
          <Button variant="outline" onClick={onAsk}>
            <Icon name="chat" size={16} className="mr-2" />
            Ask about this result
          </Button>
        </CardContent>
      </Card>

      {/* Vitals tiles */}
      <div>
        <div className="mb-3 flex items-baseline justify-between">
          <h3 className="font-semibold">Your last readings</h3>
          {watchCount > 0 ? (
            <span className="text-xs text-warning">
              {watchCount} outside the usual range
            </span>
          ) : (
            <span className="text-xs text-muted-foreground">All within usual ranges</span>
          )}
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {tiles.map((tile) => {
            const tone = readingTone(tile.key, latest);
            return (
              <Card
                key={tile.key}
                className={cn(
                  "border-2 transition-shadow hover:shadow-md",
                  tone === "watch" ? "border-warning/40" : "border-transparent",
                )}
              >
                <CardContent className="p-5">
                  <div className="flex items-center justify-between">
                    <Icon
                      name={tile.icon}
                      size={20}
                      className={tone === "watch" ? "text-warning" : "text-primary"}
                    />
                    {tone === "watch" ? (
                      <Icon name="warning" size={14} className="text-warning" title="Outside the usual range" />
                    ) : null}
                  </div>
                  <p className="mt-3 text-xs text-muted-foreground">{tile.label}</p>
                  <p className="tabular mt-0.5 text-2xl font-bold">
                    {tile.value}{" "}
                    <span className="text-sm font-normal text-muted-foreground">{tile.unit}</span>
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Trend + recent questions */}
      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Blood pressure over time</CardTitle>
          </CardHeader>
          <CardContent>
            <VitalsTrend records={vitals.data ?? []} />
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Recent questions</CardTitle>
          </CardHeader>
          <CardContent>
            {conversations.isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </div>
            ) : (conversations.data?.length ?? 0) === 0 ? (
              <div className="py-4 text-center">
                <p className="text-sm text-muted-foreground">
                  Nothing asked yet.
                </p>
                <Button variant="link" size="sm" onClick={onAsk} className="mt-1">
                  Ask your first question
                </Button>
              </div>
            ) : (
              <ul className="space-y-3">
                {conversations.data?.slice(0, 4).map((conversation) => (
                  <li key={conversation.id} className="border-b pb-3 last:border-0 last:pb-0">
                    <p className="line-clamp-2 text-sm font-medium">{conversation.user_message}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {formatDateTime(conversation.created_at)}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Overview;
