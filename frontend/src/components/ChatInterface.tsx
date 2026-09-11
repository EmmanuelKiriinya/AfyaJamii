import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Icon, IconSpinner } from "@/components/Icon";
import { FormattedAdvice } from "@/components/FormattedAdvice";
import { useAskQuestion } from "@/lib/queries";
import { ApiError } from "@/lib/api";
import { formatTime } from "@/lib/health";
import { cn } from "@/lib/utils";
import type { IconName } from "@/lib/icons";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  failed?: boolean;
}

/** Openers that reflect what the service can actually answer. */
const SUGGESTIONS: { icon: IconName; text: string }[] = [
  { icon: "nutrition", text: "What should I be eating this trimester?" },
  { icon: "heartRate", text: "My ankles are swelling — is that normal?" },
  { icon: "bloodPressure", text: "How do I bring my blood pressure down?" },
  { icon: "care", text: "Ninawezaje kuongeza damu?" },
];

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const askQuestion = useAskQuestion();

  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const isSending = askQuestion.isPending;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

  const send = async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || isSending) return;

    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
        timestamp: new Date().toISOString(),
      },
    ]);
    setInput("");

    try {
      const response = await askQuestion.mutateAsync(trimmed);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.advice,
          timestamp: response.timestamp,
        },
      ]);
    } catch (error) {
      // Show the failure in the thread rather than a toast, so the question
      // and what happened to it stay together.
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            error instanceof ApiError
              ? error.message
              : "Something went wrong. Please try asking again.",
          timestamp: new Date().toISOString(),
          failed: true,
        },
      ]);
    } finally {
      inputRef.current?.focus();
    }
  };

  return (
    <Card className="mx-auto flex h-[min(72vh,680px)] max-w-3xl flex-col border-2 border-primary/10 shadow-lg">
      <CardHeader className="border-b bg-gradient-to-r from-primary/5 to-purple-500/5">
        <CardTitle className="flex items-center gap-3 text-2xl">
          <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-blue-600 text-primary-foreground">
            <Icon name="chat" size={22} />
          </span>
          Ask about your health
        </CardTitle>
        <CardDescription className="text-base">
          Nutrition, symptoms, or what a reading means. English or Kiswahili.
        </CardDescription>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-4 overflow-hidden pt-6">
        <div className="flex-1 overflow-y-auto pr-1">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col justify-center py-6">
              <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/10 to-purple-500/10">
                <Icon name="chat" size={30} className="text-primary" />
              </div>
              <h3 className="text-center text-lg font-semibold">Start a conversation</h3>
              <p className="mt-1 text-center text-sm text-muted-foreground">
                You could begin with one of these
              </p>

              <div className="mx-auto mt-6 grid w-full max-w-lg gap-2.5">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion.text}
                    type="button"
                    onClick={() => void send(suggestion.text)}
                    className="flex items-center gap-3 rounded-xl border-2 border-transparent bg-muted/50 px-4 py-3 text-left text-sm transition-all hover:border-primary/30 hover:bg-primary/5"
                  >
                    <Icon name={suggestion.icon} size={17} className="shrink-0 text-primary" />
                    {suggestion.text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              {messages.map((message) =>
                message.role === "user" ? (
                  <div key={message.id} className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl rounded-br-md bg-gradient-to-br from-primary to-blue-600 px-4 py-2.5 text-primary-foreground shadow-sm">
                      <p className="text-[0.95rem] leading-relaxed">{message.content}</p>
                      <p className="mt-1.5 text-[11px] opacity-75">
                        {formatTime(message.timestamp)}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div key={message.id} className="duration-300 animate-in fade-in slide-in-from-bottom-2">
                    <div className="mb-2 flex items-center gap-2">
                      <span
                        className={cn(
                          "flex h-6 w-6 items-center justify-center rounded-full",
                          message.failed ? "bg-warning/15" : "bg-primary/10",
                        )}
                      >
                        <Icon
                          name={message.failed ? "warning" : "care"}
                          size={13}
                          className={message.failed ? "text-warning" : "text-primary"}
                        />
                      </span>
                      <span className="text-xs font-medium">AfyaJamii</span>
                      <span className="text-[11px] text-muted-foreground">
                        {formatTime(message.timestamp)}
                      </span>
                    </div>

                    <div
                      className={cn(
                        "rounded-2xl rounded-tl-md border px-4 py-3.5 shadow-sm",
                        message.failed
                          ? "border-warning/40 bg-warning/5"
                          : "border-border bg-secondary/40",
                      )}
                    >
                      {message.failed ? (
                        <p className="text-sm leading-relaxed">{message.content}</p>
                      ) : (
                        <FormattedAdvice text={message.content} />
                      )}
                    </div>
                  </div>
                ),
              )}

              {isSending ? (
                <div className="flex items-center gap-2.5 text-sm text-muted-foreground">
                  <IconSpinner size={15} />
                  <span>Thinking about your question…</span>
                </div>
              ) : null}
            </div>
          )}
          <div ref={endRef} />
        </div>

        <form
          onSubmit={(event) => {
            event.preventDefault();
            void send(input);
          }}
          className="flex gap-3 border-t pt-4"
        >
          <Input
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask me anything about your health..."
            disabled={isSending}
            aria-label="Your question"
            maxLength={500}
            className="h-12 flex-1 text-base"
          />
          <Button
            type="submit"
            size="lg"
            className="px-6"
            disabled={isSending || !input.trim()}
            aria-label="Send question"
          >
            {isSending ? <IconSpinner size={18} /> : <Icon name="send" size={18} />}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default ChatInterface;
