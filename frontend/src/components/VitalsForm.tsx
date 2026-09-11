import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Icon, IconSpinner } from "@/components/Icon";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { useSubmitVitals } from "@/lib/queries";
import { ApiError, type TemperatureUnit, type VitalsSubmitResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { IconName } from "@/lib/icons";

interface VitalsFormProps {
  onAssessment: (data: VitalsSubmitResponse) => void;
}

interface FormState {
  age: string;
  systolic_bp: string;
  diastolic_bp: string;
  bs: string;
  body_temp: string;
  heart_rate: string;
  patient_history: string;
}

interface FieldSpec {
  key: keyof FormState;
  label: string;
  icon: IconName;
  unit: string;
  hint: string;
  min: number;
  max: number;
  step?: string;
  placeholder: string;
}

const EMPTY_FORM: FormState = {
  age: "",
  systolic_bp: "",
  diastolic_bp: "",
  bs: "",
  body_temp: "",
  heart_rate: "",
  patient_history: "",
};

// Ranges mirror the API schema, so out-of-range values are caught before a
// round trip rather than coming back as a 422.
const FIELDS: FieldSpec[] = [
  { key: "age", label: "Age", icon: "age", unit: "years", hint: "Between 15 and 50", min: 15, max: 50, placeholder: "28" },
  { key: "systolic_bp", label: "Systolic BP", icon: "bloodPressure", unit: "mmHg", hint: "The upper number", min: 70, max: 200, placeholder: "120" },
  { key: "diastolic_bp", label: "Diastolic BP", icon: "bloodPressure", unit: "mmHg", hint: "The lower number", min: 40, max: 130, placeholder: "80" },
  { key: "heart_rate", label: "Heart rate", icon: "heartRate", unit: "bpm", hint: "Usually 60–100 at rest", min: 40, max: 150, placeholder: "76" },
  { key: "bs", label: "Blood sugar", icon: "bloodSugar", unit: "mmol/L", hint: "Usually 4.0–7.0", min: 3, max: 30, step: "0.1", placeholder: "5.5" },
];

const TEMPERATURE_RANGES: Record<TemperatureUnit, { min: number; max: number; hint: string }> = {
  celsius: { min: 35, max: 42, hint: "Normal is about 36.5–37.5 °C" },
  fahrenheit: { min: 95, max: 107.6, hint: "Normal is about 97.7–99.5 °F" },
};

const VitalsForm = ({ onAssessment }: VitalsFormProps) => {
  const { accountType } = useAuth();
  const { toast } = useToast();
  const submitVitals = useSubmitVitals();

  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [unit, setUnit] = useState<TemperatureUnit>("celsius");
  const [errors, setErrors] = useState<Partial<Record<keyof FormState, string>>>({});

  const isLoading = submitVitals.isPending;

  const update = (key: keyof FormState, value: string) => {
    setForm((current) => ({ ...current, [key]: value }));
    setErrors((current) => ({ ...current, [key]: undefined }));
  };

  const validate = (): boolean => {
    const found: Partial<Record<keyof FormState, string>> = {};

    for (const field of FIELDS) {
      const raw = form[field.key].trim();
      if (!raw) {
        found[field.key] = "Required.";
        continue;
      }
      const value = Number(raw);
      if (Number.isNaN(value)) {
        found[field.key] = "Enter a number.";
      } else if (value < field.min || value > field.max) {
        found[field.key] = `Must be ${field.min}–${field.max}.`;
      }
    }

    const range = TEMPERATURE_RANGES[unit];
    const temperature = Number(form.body_temp.trim());
    if (!form.body_temp.trim()) {
      found.body_temp = "Required.";
    } else if (Number.isNaN(temperature)) {
      found.body_temp = "Enter a number.";
    } else if (temperature < range.min || temperature > range.max) {
      found.body_temp = `Must be ${range.min}–${range.max}.`;
    }

    // The API rejects this too, but catching it here explains it better.
    if (!found.systolic_bp && !found.diastolic_bp) {
      if (Number(form.diastolic_bp) >= Number(form.systolic_bp)) {
        found.diastolic_bp = "Must be below the upper number.";
      }
    }

    setErrors(found);
    if (Object.keys(found).length > 0) {
      toast({
        title: "Check your readings",
        description: "Some values need correcting before we can assess them.",
        variant: "destructive",
      });
      return false;
    }
    return true;
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!validate()) return;

    try {
      const response = await submitVitals.mutateAsync({
        vitals: {
          age: Number(form.age),
          systolic_bp: Number(form.systolic_bp),
          diastolic_bp: Number(form.diastolic_bp),
          bs: Number(form.bs),
          body_temp: Number(form.body_temp),
          body_temp_unit: unit,
          heart_rate: Number(form.heart_rate),
          patient_history: form.patient_history.trim() || undefined,
        },
        account_type: accountType ?? undefined,
      });

      onAssessment(response);
      setForm(EMPTY_FORM);
      toast({ title: "Saved", description: "Your readings have been assessed." });
    } catch (error) {
      toast({
        title: "Could not save",
        description:
          error instanceof ApiError ? error.message : "Failed to submit your readings",
        variant: "destructive",
      });
    }
  };

  return (
    <Card className="mx-auto max-w-3xl border-2 border-primary/10 shadow-lg">
      <CardHeader className="border-b bg-gradient-to-r from-primary/5 to-purple-500/5">
        <CardTitle className="flex items-center gap-3 text-2xl">
          <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-blue-600 text-primary-foreground">
            <Icon name="vitals" size={22} />
          </span>
          New reading
        </CardTitle>
        <CardDescription className="text-base">
          Enter the numbers from your clinic card or home monitor for an assessment.
        </CardDescription>
      </CardHeader>

      <CardContent className="pt-6">
        <form onSubmit={handleSubmit} className="space-y-6" noValidate>
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            {FIELDS.map((field) => (
              <div key={field.key} className="space-y-2">
                <Label htmlFor={field.key} className="flex items-center gap-2 text-sm font-medium">
                  <Icon name={field.icon} size={15} className="text-primary" />
                  {field.label}
                  <span className="font-normal text-muted-foreground">({field.unit})</span>
                </Label>
                <Input
                  id={field.key}
                  type="number"
                  inputMode="decimal"
                  step={field.step ?? "1"}
                  min={field.min}
                  max={field.max}
                  placeholder={field.placeholder}
                  value={form[field.key]}
                  onChange={(event) => update(field.key, event.target.value)}
                  disabled={isLoading}
                  className={cn("tabular h-11", errors[field.key] && "border-destructive")}
                  aria-invalid={Boolean(errors[field.key])}
                  aria-describedby={`${field.key}-hint`}
                />
                <p
                  id={`${field.key}-hint`}
                  className={cn(
                    "text-xs",
                    errors[field.key] ? "text-destructive" : "text-muted-foreground",
                  )}
                >
                  {errors[field.key] ?? field.hint}
                </p>
              </div>
            ))}

            {/* Temperature carries its own unit switch. */}
            <div className="space-y-2">
              <Label htmlFor="body_temp" className="flex items-center gap-2 text-sm font-medium">
                <Icon name="temperature" size={15} className="text-primary" />
                Temperature
              </Label>
              <div className="flex gap-2">
                <Input
                  id="body_temp"
                  type="number"
                  inputMode="decimal"
                  step="0.1"
                  min={TEMPERATURE_RANGES[unit].min}
                  max={TEMPERATURE_RANGES[unit].max}
                  placeholder={unit === "celsius" ? "36.9" : "98.4"}
                  value={form.body_temp}
                  onChange={(event) => update("body_temp", event.target.value)}
                  disabled={isLoading}
                  className={cn("tabular h-11 flex-1", errors.body_temp && "border-destructive")}
                  aria-invalid={Boolean(errors.body_temp)}
                  aria-describedby="body_temp-hint"
                />
                <div
                  role="group"
                  aria-label="Temperature unit"
                  className="flex shrink-0 overflow-hidden rounded-md border border-input"
                >
                  {(["celsius", "fahrenheit"] as const).map((option) => (
                    <button
                      key={option}
                      type="button"
                      onClick={() => {
                        setUnit(option);
                        // The number means something different in the other
                        // scale, so clear it rather than mislabel it.
                        update("body_temp", "");
                      }}
                      disabled={isLoading}
                      aria-pressed={unit === option}
                      className={cn(
                        "px-3.5 text-sm transition-colors",
                        unit === option
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-muted",
                      )}
                    >
                      °{option === "celsius" ? "C" : "F"}
                    </button>
                  ))}
                </div>
              </div>
              <p
                id="body_temp-hint"
                className={cn(
                  "text-xs",
                  errors.body_temp ? "text-destructive" : "text-muted-foreground",
                )}
              >
                {errors.body_temp ?? TEMPERATURE_RANGES[unit].hint}
              </p>
            </div>
          </div>

          {/* This field was collected in state but never rendered, so the
              history a user typed could never actually be entered. */}
          <div className="space-y-2">
            <Label htmlFor="patient_history" className="text-sm font-medium">
              Anything else worth knowing?
            </Label>
            <Textarea
              id="patient_history"
              rows={3}
              maxLength={1000}
              placeholder="Previous pregnancies, conditions you are managing, medicines you take, how you have been feeling…"
              value={form.patient_history}
              onChange={(event) => update("patient_history", event.target.value)}
              disabled={isLoading}
              className="resize-y"
            />
            <p className="text-xs text-muted-foreground">
              Optional, but it makes the guidance more useful. {form.patient_history.length}/1000
            </p>
          </div>

          <Button type="submit" className="h-12 w-full text-base font-medium" disabled={isLoading}>
            {isLoading ? (
              <>
                <IconSpinner size={18} className="mr-2" />
                Checking your readings...
              </>
            ) : (
              <>
                <Icon name="vitals" size={18} className="mr-2" />
                Check my readings
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default VitalsForm;
