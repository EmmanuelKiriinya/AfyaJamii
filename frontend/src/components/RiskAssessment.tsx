import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Icon } from "@/components/Icon";
import { FormattedAdvice } from "@/components/FormattedAdvice";
import { featureLabel, presentRisk } from "@/lib/health";
import type { VitalsSubmitResponse } from "@/lib/api";

/** The result of one assessment: the level, what drove it, and what to do. */

const RiskAssessment = ({ data }: { data: VitalsSubmitResponse }) => {
  const risk = presentRisk(data.ml_output.risk_label);

  const factors = Object.entries(data.ml_output.feature_importances)
    .filter(([, weight]) => weight > 0)
    .sort(([, a], [, b]) => b - a);

  // Scaled against the largest factor so smaller bars stay visible rather
  // than collapsing to a sliver.
  const largest = factors[0]?.[1] ?? 1;

  return (
    <div className="mx-auto max-w-3xl space-y-6 duration-500 animate-in fade-in slide-in-from-bottom-4">
      <Card className="border-2 shadow-lg">
        <CardHeader className="border-b bg-gradient-to-r from-primary/5 to-purple-500/5">
          <CardTitle className="flex items-center gap-3 text-2xl">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10">
              <Icon name="care" size={22} className="text-primary" />
            </span>
            Assessment results
          </CardTitle>
          <CardDescription className="text-base">
            An analysis of the vitals you just recorded
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6 pt-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border-2 border-primary/20 bg-gradient-to-br from-primary/5 to-primary/10 p-6">
              <p className="mb-2 text-sm font-medium text-muted-foreground">Risk level</p>
              <Badge className={`${risk.badgeClass} flex w-fit items-center gap-2 px-4 py-2 text-base`}>
                <Icon name={risk.icon} size={16} />
                {risk.label}
              </Badge>
              <p className="mt-3 text-xs text-muted-foreground">{risk.meaning}</p>
            </div>

            <div className="rounded-xl border-2 border-blue-200 bg-gradient-to-br from-blue-50 to-blue-100/50 p-6 dark:border-blue-800 dark:from-blue-950/20 dark:to-blue-900/10">
              <p className="mb-2 text-sm font-medium text-muted-foreground">Confidence</p>
              <div className="flex items-baseline gap-1">
                <p className="tabular text-4xl font-bold text-blue-600 dark:text-blue-400">
                  {(data.ml_output.probability * 100).toFixed(1)}
                </p>
                <span className="text-xl text-blue-600 dark:text-blue-400">%</span>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                How sure the model is of this level
              </p>
            </div>
          </div>

          {factors.length > 0 ? (
            <div className="rounded-xl border bg-muted/30 p-6">
              <h4 className="flex items-center gap-2 text-lg font-semibold">
                <Icon name="vitals" size={18} className="text-primary" />
                What influenced this
              </h4>
              <p className="mb-5 mt-1 text-sm text-muted-foreground">
                How much weight each reading carried in the assessment
              </p>

              <div className="space-y-4">
                {factors.map(([feature, weight]) => (
                  <div key={feature} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{featureLabel(feature)}</span>
                      <span className="tabular text-sm font-semibold text-primary">
                        {(weight * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-3 overflow-hidden rounded-full bg-secondary shadow-inner">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-primary to-purple-500 transition-all duration-500 ease-out"
                        style={{ width: `${Math.max((weight / largest) * 100, 3)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card className="border-2 border-green-200 shadow-lg dark:border-green-800">
        <CardHeader className="border-b bg-gradient-to-r from-green-50 to-green-100/50 dark:from-green-950/20 dark:to-green-900/10">
          <CardTitle className="flex items-center gap-3 text-2xl">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-green-500/10">
              <Icon name="nutrition" size={22} className="text-green-600 dark:text-green-400" />
            </span>
            What to do about it
          </CardTitle>
          <CardDescription className="text-base">
            Recommendations based on your vitals and health profile
          </CardDescription>
        </CardHeader>

        <CardContent className="pt-6">
          {data.llm_advice.generated ? (
            <div className="rounded-xl border bg-gradient-to-br from-white to-green-50/30 p-6 dark:from-card dark:to-green-950/10">
              <FormattedAdvice text={data.llm_advice.advice} />
            </div>
          ) : (
            // The advice service was unreachable; this is a fallback, and must
            // not be presented as clinical guidance.
            <div className="flex items-start gap-3 rounded-xl border border-warning/40 bg-warning/5 p-5">
              <Icon name="warning" size={18} className="mt-0.5 text-warning" />
              <p className="text-sm leading-relaxed">{data.llm_advice.advice}</p>
            </div>
          )}

          <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-950/20">
            <p className="flex items-start gap-2 text-sm text-muted-foreground">
              <Icon name="info" size={16} className="mt-0.5 shrink-0" />
              <span>
                <strong>Note:</strong> this guidance is generated automatically. Always consult
                your healthcare provider for medical decisions.
              </span>
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default RiskAssessment;
