import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Icon } from "@/components/Icon";
import { useTheme } from "@/contexts/ThemeContext";
import { cn } from "@/lib/utils";
import type { IconName } from "@/lib/icons";

/**
 * Reading-comfort controls.
 *
 * This app is used on small, cheap phones and often in bright sunlight, so
 * text size and contrast matter more than usual. Preferences persist across
 * sessions — previously they reset on every page load, which made them close
 * to useless for anyone who actually needed them.
 */

const STORAGE_KEY = "afyajamii.display";
const FONT_MIN = 90;
const FONT_MAX = 150;
const FONT_STEP = 10;

interface DisplayPreferences {
  fontScale: number;
  highContrast: boolean;
}

const DEFAULTS: DisplayPreferences = { fontScale: 100, highContrast: false };

function readPreferences(): DisplayPreferences {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) as Partial<DisplayPreferences>;
    return {
      fontScale: Math.min(Math.max(Number(parsed.fontScale) || 100, FONT_MIN), FONT_MAX),
      highContrast: Boolean(parsed.highContrast),
    };
  } catch {
    return DEFAULTS;
  }
}

const THEME_OPTIONS: { value: "light" | "dark" | "system"; icon: IconName; label: string }[] = [
  { value: "light", icon: "sun", label: "Light" },
  { value: "dark", icon: "moon", label: "Dark" },
  { value: "system", icon: "monitor", label: "Auto" },
];

const AccessibilityMenu = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [preferences, setPreferences] = useState<DisplayPreferences>(readPreferences);
  const { theme, setTheme } = useTheme();

  const panelRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // Apply and persist whenever the preferences change.
  useEffect(() => {
    document.documentElement.style.fontSize = `${preferences.fontScale}%`;
    document.documentElement.classList.toggle("high-contrast", preferences.highContrast);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    } catch {
      // Preferences just won't survive a reload.
    }
  }, [preferences]);

  // Close on Escape or a click outside, and return focus to the trigger.
  useEffect(() => {
    if (!isOpen) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false);
        triggerRef.current?.focus();
      }
    };

    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!panelRef.current?.contains(target) && !triggerRef.current?.contains(target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [isOpen]);

  const setFontScale = (next: number) =>
    setPreferences((current) => ({
      ...current,
      fontScale: Math.min(Math.max(next, FONT_MIN), FONT_MAX),
    }));

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
        aria-label="Reading options"
        className="fixed bottom-24 right-5 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-gradient-to-br from-primary to-blue-600 text-primary-foreground shadow-2xl transition-transform hover:scale-110 lg:bottom-6"
      >
        <Icon name="accessibility" size={20} />
      </button>

      {isOpen ? (
        <div
          ref={panelRef}
          role="dialog"
          aria-label="Reading options"
          className="fixed bottom-40 right-5 z-50 w-[19rem] rounded-xl border-2 border-primary/10 bg-card p-5 shadow-2xl duration-300 animate-in fade-in slide-in-from-bottom-4 lg:bottom-24"
        >
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold">Reading options</h2>
            <button
              type="button"
              onClick={() => {
                setIsOpen(false);
                triggerRef.current?.focus();
              }}
              aria-label="Close reading options"
              className="rounded-sm p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Icon name="close" size={14} />
            </button>
          </div>

          {/* Text size */}
          <div className="mt-5">
            <div className="flex items-baseline justify-between">
              <span className="text-sm font-medium">Text size</span>
              <span className="tabular text-xs text-muted-foreground">
                {preferences.fontScale}%
              </span>
            </div>
            <div className="mt-2 flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="flex-1"
                onClick={() => setFontScale(preferences.fontScale - FONT_STEP)}
                disabled={preferences.fontScale <= FONT_MIN}
              >
                <Icon name="textSmaller" size={14} />
                <span className="sr-only">Smaller text</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="flex-1 text-xs"
                onClick={() => setFontScale(100)}
                disabled={preferences.fontScale === 100}
              >
                Reset
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="flex-1"
                onClick={() => setFontScale(preferences.fontScale + FONT_STEP)}
                disabled={preferences.fontScale >= FONT_MAX}
              >
                <Icon name="textLarger" size={14} />
                <span className="sr-only">Larger text</span>
              </Button>
            </div>
          </div>

          {/* Contrast */}
          <div className="mt-5">
            <span className="text-sm font-medium">Contrast</span>
            <Button
              variant={preferences.highContrast ? "default" : "outline"}
              size="sm"
              className="mt-2 w-full justify-start"
              aria-pressed={preferences.highContrast}
              onClick={() =>
                setPreferences((current) => ({
                  ...current,
                  highContrast: !current.highContrast,
                }))
              }
            >
              <Icon name="contrast" size={14} className="mr-2" />
              {preferences.highContrast ? "High contrast on" : "High contrast off"}
            </Button>
          </div>

          {/* Theme */}
          <div className="mt-5">
            <span className="text-sm font-medium">Appearance</span>
            <div
              role="group"
              aria-label="Appearance"
              className="mt-2 flex overflow-hidden rounded-md border border-input"
            >
              {THEME_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setTheme(option.value)}
                  aria-pressed={theme === option.value}
                  className={cn(
                    "flex flex-1 items-center justify-center gap-1.5 py-2 text-xs transition-colors",
                    theme === option.value
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-muted",
                  )}
                >
                  <Icon name={option.icon} size={13} />
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <p className="mt-5 border-t border-border pt-4 text-xs leading-relaxed text-muted-foreground">
            These settings are saved on this device.
          </p>
        </div>
      ) : null}
    </>
  );
};

export default AccessibilityMenu;
