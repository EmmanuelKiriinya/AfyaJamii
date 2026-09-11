import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Icon } from "@/components/Icon";
import { FormattedAdvice } from "@/components/FormattedAdvice";
import { useConversations, useVitalsHistory } from "@/lib/queries";
import { formatDateTime, presentRisk } from "@/lib/health";
import type { IconName } from "@/lib/icons";

/** Past readings and conversations. */

function EmptyState({ icon, title, body }: { icon: IconName; title: string; body: string }) {
  return (
    <div className="py-16 text-center">
      <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/10 to-purple-500/10">
        <Icon name={icon} size={30} className="text-primary" />
      </div>
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="mx-auto mt-2 max-w-sm leading-relaxed text-muted-foreground">{body}</p>
    </div>
  );
}

function LoadingRows() {
  return (
    <div className="space-y-4" aria-busy="true" aria-label="Loading your history">
      {[0, 1, 2].map((row) => (
        <Card key={row}>
          <CardContent className="p-5">
            <Skeleton className="h-4 w-44" />
            <Skeleton className="mt-4 h-3 w-full" />
            <Skeleton className="mt-2 h-3 w-2/3" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

const HistoryView = () => {
  const vitals = useVitalsHistory(20);
  const conversations = useConversations(20);

  const isLoading = vitals.isLoading || conversations.isLoading;
  const error = vitals.error ?? conversations.error;

  const refresh = () => {
    void vitals.refetch();
    void conversations.refetch();
  };

  return (
    <Card className="mx-auto max-w-4xl border-2 border-primary/10 shadow-lg">
      <CardHeader className="border-b bg-gradient-to-r from-primary/5 to-purple-500/5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-3 text-2xl">
              <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-blue-600 text-primary-foreground">
                <Icon name="history" size={22} />
              </span>
              Your history
            </CardTitle>
            <CardDescription className="mt-2 text-base">
              Everything you have recorded, newest first. Useful to show a health worker.
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={refresh} disabled={isLoading}>
            Refresh
          </Button>
        </div>
      </CardHeader>

      <CardContent className="pt-6">
        {error ? (
          <div
            role="alert"
            className="mb-6 flex items-start gap-2.5 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm"
          >
            <Icon name="alert" size={16} className="mt-0.5 text-destructive" />
            <span>
              {error instanceof Error ? error.message : "Could not load your history."}
            </span>
          </div>
        ) : null}

        <Tabs defaultValue="vitals">
          <TabsList className="mx-auto grid h-12 w-full max-w-md grid-cols-2">
            <TabsTrigger
              value="vitals"
              className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Icon name="vitals" size={15} />
              Readings
            </TabsTrigger>
            <TabsTrigger
              value="conversations"
              className="flex items-center gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Icon name="chat" size={15} />
              Questions
            </TabsTrigger>
          </TabsList>

          <TabsContent value="vitals" className="mt-6 focus-visible:outline-none">
            {isLoading ? (
              <LoadingRows />
            ) : (vitals.data?.length ?? 0) === 0 ? (
              <EmptyState
                icon="vitals"
                title="No readings yet"
                body="Once you record your first set of readings, they will appear here so you can watch how things change."
              />
            ) : (
              <ol className="space-y-4">
                {vitals.data?.map((record, index) => {
                  const risk = presentRisk(record.ml_risk_label);
                  return (
                    <li key={record.id}>
                      <Card className="transition-shadow hover:shadow-md">
                        <CardHeader className="pb-3">
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-center gap-3">
                              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10">
                                <Icon name="vitals" size={18} className="text-primary" />
                              </span>
                              <div>
                                <CardTitle className="text-base font-semibold">
                                  <time dateTime={record.created_at}>
                                    {formatDateTime(record.created_at)}
                                  </time>
                                </CardTitle>
                                <CardDescription className="mt-0.5 text-xs">
                                  Record #{record.id}
                                  {index === 0 ? " · Latest" : ""}
                                </CardDescription>
                              </div>
                            </div>
                            <Badge className={`${risk.badgeClass} flex items-center gap-1.5 px-3 py-1`}>
                              <Icon name={risk.icon} size={12} />
                              {risk.label}
                            </Badge>
                          </div>
                        </CardHeader>

                        <CardContent className="space-y-4">
                          <dl className="grid grid-cols-2 gap-3 md:grid-cols-3">
                            {[
                              ["Age", `${record.age} years`],
                              ["Blood pressure", `${record.systolic_bp}/${record.diastolic_bp} mmHg`],
                              ["Heart rate", `${record.heart_rate} bpm`],
                              ["Blood sugar", `${record.bs} mmol/L`],
                              [
                                "Temperature",
                                `${record.body_temp}°${record.body_temp_unit === "celsius" ? "C" : "F"}`,
                              ],
                              ["Confidence", `${(record.ml_probability * 100).toFixed(1)}%`],
                            ].map(([label, value]) => (
                              <div key={label} className="rounded-lg bg-muted/50 p-3">
                                <dt className="mb-1 text-xs text-muted-foreground">{label}</dt>
                                <dd className="tabular font-semibold">{value}</dd>
                              </div>
                            ))}
                          </dl>

                          {record.patient_history ? (
                            <div className="border-t pt-3">
                              <p className="mb-2 text-xs font-medium text-muted-foreground">
                                You noted:
                              </p>
                              <p className="rounded-lg bg-muted/30 p-3 text-sm leading-relaxed">
                                {record.patient_history}
                              </p>
                            </div>
                          ) : null}
                        </CardContent>
                      </Card>
                    </li>
                  );
                })}
              </ol>
            )}
          </TabsContent>

          <TabsContent value="conversations" className="mt-6 focus-visible:outline-none">
            {isLoading ? (
              <LoadingRows />
            ) : (conversations.data?.length ?? 0) === 0 ? (
              <EmptyState
                icon="chat"
                title="No questions yet"
                body="Questions you ask, and the answers you get, are kept here so you can look back at them."
              />
            ) : (
              <ol className="space-y-4">
                {conversations.data?.map((conversation, index) => (
                  <li key={conversation.id}>
                    <Card className="transition-shadow hover:shadow-md">
                      <CardHeader className="pb-3">
                        <div className="flex items-center gap-3">
                          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-green-500/10">
                            <Icon name="chat" size={18} className="text-green-600 dark:text-green-400" />
                          </span>
                          <div>
                            <CardTitle className="text-base font-semibold">
                              <time dateTime={conversation.created_at}>
                                {formatDateTime(conversation.created_at)}
                              </time>
                            </CardTitle>
                            <CardDescription className="mt-0.5 text-xs">
                              Conversation #{conversation.id}
                              {index === 0 ? " · Latest" : ""}
                            </CardDescription>
                          </div>
                        </div>
                      </CardHeader>

                      <CardContent className="space-y-4">
                        <div className="rounded-xl border border-blue-200 bg-gradient-to-br from-blue-50 to-blue-100/50 p-4 dark:border-blue-800 dark:from-blue-950/20 dark:to-blue-900/10">
                          <p className="mb-1 text-xs font-medium text-muted-foreground">
                            Your question
                          </p>
                          <p className="text-sm font-medium leading-relaxed">
                            {conversation.user_message}
                          </p>
                        </div>

                        <div className="rounded-xl border border-green-200 bg-gradient-to-br from-green-50 to-green-100/50 p-4 dark:border-green-800 dark:from-green-950/20 dark:to-green-900/10">
                          <p className="mb-2 text-xs font-medium text-muted-foreground">
                            AfyaJamii replied
                          </p>
                          <FormattedAdvice text={conversation.ai_response} />
                        </div>
                      </CardContent>
                    </Card>
                  </li>
                ))}
              </ol>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default HistoryView;
