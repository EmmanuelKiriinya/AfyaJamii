import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Icon } from "@/components/Icon";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import type { IconName } from "@/lib/icons";
import Overview from "@/components/Overview";
import VitalsForm from "@/components/VitalsForm";
import RiskAssessment from "@/components/RiskAssessment";
import ChatInterface from "@/components/ChatInterface";
import HistoryView from "@/components/HistoryView";
import type { VitalsSubmitResponse } from "@/lib/api";
import logo from "@/assets/logo.png";

/**
 * Dashboard shell.
 *
 * Replaces the previous three-tab layout with a persistent sidebar on desktop
 * and a fixed bottom bar on mobile, plus an overview that answers "how am I
 * doing?" before asking the user to enter anything. The tab strip gave equal
 * weight to four destinations and led with an empty form; this leads with the
 * user's most recent result.
 */

type View = "overview" | "record" | "ask" | "history";

const NAV: { view: View; icon: IconName; label: string; hint: string }[] = [
  { view: "overview", icon: "care", label: "Overview", hint: "How you are doing" },
  { view: "record", icon: "vitals", label: "New reading", hint: "Enter your vitals" },
  { view: "ask", icon: "chat", label: "Ask", hint: "Questions and advice" },
  { view: "history", icon: "history", label: "History", hint: "Everything recorded" },
];

const ACCOUNT_LABELS: Record<string, string> = {
  pregnant: "Antenatal",
  postnatal: "Postnatal",
  general: "General health",
};

const Dashboard = () => {
  const { username, accountType, signOut } = useAuth();
  const navigate = useNavigate();
  const [view, setView] = useState<View>("overview");
  const [assessment, setAssessment] = useState<VitalsSubmitResponse | null>(null);

  const handleSignOut = () => {
    signOut();
    navigate("/login", { replace: true });
  };

  const go = (next: View) => {
    // A result belongs to the reading that produced it; leaving the recording
    // view discards it rather than showing a stale assessment on return.
    if (next !== "record") setAssessment(null);
    setView(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const active = NAV.find((item) => item.view === view);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50/60 via-purple-50/30 to-pink-50/40 dark:from-blue-950/20 dark:via-purple-950/10 dark:to-pink-950/20">
      <div className="flex">
        {/* Sidebar — desktop */}
        <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r bg-white/70 backdrop-blur-md dark:bg-card/60 lg:flex">
          <button
            type="button"
            onClick={() => go("overview")}
            aria-label="AfyaJamii, go to your overview"
            className="flex w-full items-center gap-3 border-b px-6 py-5 text-left transition-opacity hover:opacity-80"
          >
            <img src={logo} alt="" aria-hidden className="h-9 w-9" />
            <span className="text-lg font-bold">AfyaJamii</span>
          </button>

          <nav className="flex-1 space-y-1 p-4" aria-label="Main">
            {NAV.map((item) => (
              <button
                key={item.view}
                type="button"
                onClick={() => go(item.view)}
                aria-current={view === item.view ? "page" : undefined}
                className={cn(
                  "flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-all",
                  view === item.view
                    ? "bg-gradient-to-r from-primary to-blue-600 text-primary-foreground shadow-md"
                    : "text-muted-foreground hover:bg-primary/5 hover:text-foreground",
                )}
              >
                <Icon name={item.icon} size={18} />
                <span className="flex-1">
                  <span className="block text-sm font-medium">{item.label}</span>
                  <span
                    className={cn(
                      "block text-xs",
                      view === item.view ? "text-primary-foreground/75" : "text-muted-foreground",
                    )}
                  >
                    {item.hint}
                  </span>
                </span>
              </button>
            ))}
          </nav>

          <div className="border-t p-4">
            <div className="mb-3 flex items-center gap-3 px-1">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-primary to-purple-600 text-sm font-semibold text-primary-foreground">
                {username?.slice(0, 1).toUpperCase() ?? "?"}
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{username}</p>
                <p className="truncate text-xs text-muted-foreground">
                  {accountType ? ACCOUNT_LABELS[accountType] ?? accountType : ""}
                </p>
              </div>
            </div>
            <Button variant="outline" size="sm" className="w-full" onClick={handleSignOut}>
              <Icon name="logout" size={15} className="mr-2" />
              Sign out
            </Button>
          </div>
        </aside>

        {/* Main column */}
        <div className="min-w-0 flex-1 pb-24 lg:pb-0">
          {/* Header — mobile shows the brand, desktop shows the section title */}
          <header className="sticky top-0 z-40 border-b bg-white/80 backdrop-blur-md dark:bg-card/70">
            <div className="flex items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
              <div className="flex items-center gap-3">
                {/* Tappable on mobile, where the sidebar mark is hidden. On
                    desktop this slot shows the section title instead. */}
                <button
                  type="button"
                  onClick={() => go("overview")}
                  aria-label="AfyaJamii, go to your overview"
                  className="flex items-center gap-3 text-left transition-opacity hover:opacity-80 lg:pointer-events-none lg:cursor-default"
                >
                  <img src={logo} alt="" aria-hidden className="h-9 w-9 lg:hidden" />
                  <div>
                    <h1 className="text-lg font-bold sm:text-xl">
                      <span className="lg:hidden">AfyaJamii</span>
                      <span className="hidden lg:inline">{active?.label}</span>
                    </h1>
                    <p className="hidden text-xs text-muted-foreground lg:block">{active?.hint}</p>
                    <p className="text-xs text-muted-foreground lg:hidden">
                      Welcome back, {username}
                    </p>
                  </div>
                </button>
              </div>

              <div className="flex items-center gap-2">
                {accountType ? (
                  <span className="hidden rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary sm:inline">
                    {ACCOUNT_LABELS[accountType] ?? accountType}
                  </span>
                ) : null}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleSignOut}
                  className="lg:hidden"
                  aria-label="Sign out"
                >
                  <Icon name="logout" size={18} />
                </Button>
              </div>
            </div>
          </header>

          <main className="px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
            {view === "overview" ? (
              <Overview onRecord={() => go("record")} onAsk={() => go("ask")} />
            ) : null}

            {view === "record" ? (
              <div className="space-y-6">
                <VitalsForm onAssessment={setAssessment} />
                {assessment ? <RiskAssessment data={assessment} /> : null}
              </div>
            ) : null}

            {view === "ask" ? <ChatInterface /> : null}

            {view === "history" ? <HistoryView /> : null}
          </main>

          <footer className="border-t px-4 py-5 sm:px-6 lg:px-8">
            <p className="flex items-start gap-2 text-xs text-muted-foreground">
              <Icon name="info" size={14} className="mt-0.5 shrink-0" />
              <span>
                Guidance here supports, and does not replace, care from a health worker. In an
                emergency call 999, or 1199 for a Red Cross ambulance.
              </span>
            </p>
          </footer>
        </div>
      </div>

      {/* Bottom navigation — mobile */}
      <nav
        aria-label="Main, bottom"
        className="fixed inset-x-0 bottom-0 z-40 border-t bg-white/90 backdrop-blur-md dark:bg-card/90 lg:hidden"
      >
        <div className="grid grid-cols-4">
          {NAV.map((item) => (
            <button
              key={item.view}
              type="button"
              onClick={() => go(item.view)}
              aria-current={view === item.view ? "page" : undefined}
              className={cn(
                "flex flex-col items-center gap-1 py-2.5 text-[11px] transition-colors",
                view === item.view ? "text-primary" : "text-muted-foreground",
              )}
            >
              <span
                className={cn(
                  "flex h-8 w-12 items-center justify-center rounded-full transition-colors",
                  view === item.view && "bg-primary/10",
                )}
              >
                <Icon name={item.icon} size={18} />
              </span>
              {item.label}
            </button>
          ))}
        </div>
      </nav>
    </div>
  );
};

export default Dashboard;
