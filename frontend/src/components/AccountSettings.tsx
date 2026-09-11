import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Icon, IconSpinner } from "@/components/Icon";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { api, ApiError, type AccountType, type UserProfile } from "@/lib/api";
import { formatDateTime } from "@/lib/health";
import { cn } from "@/lib/utils";

/**
 * Account settings: profile, password, and the two ways to leave.
 *
 * Deactivation is offered above deletion because it is usually what people
 * actually want — sign-in stops, the health history survives. Deletion is
 * behind a collapsed panel that has to be opened deliberately, and then asks
 * for the password and a typed phrase, matching what the API enforces.
 */

const ACCOUNT_TYPES: { value: AccountType; label: string }[] = [
  { value: "pregnant", label: "Pregnant" },
  { value: "postnatal", label: "Postnatal" },
  { value: "general", label: "General health" },
];

const DELETE_PHRASE = "DELETE MY ACCOUNT";

const AccountSettings = () => {
  const { token, signOut, updateSession } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Profile form
  const [form, setForm] = useState({ username: "", email: "", full_name: "", account_type: "" as AccountType | "" });
  const [isSavingProfile, setIsSavingProfile] = useState(false);

  // Password form
  const [passwords, setPasswords] = useState({ current: "", next: "", confirm: "" });
  const [isSavingPassword, setIsSavingPassword] = useState(false);

  // Leaving
  const [isDeactivating, setIsDeactivating] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [deleteForm, setDeleteForm] = useState({ password: "", confirmation: "" });
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;

    api
      .getProfile(token)
      .then((data) => {
        if (cancelled) return;
        setProfile(data);
        setForm({
          username: data.username,
          email: data.email,
          full_name: data.full_name ?? "",
          account_type: data.account_type,
        });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setLoadError(error instanceof ApiError ? error.message : "Could not load your account.");
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [token]);

  /** Only send what actually changed, so a no-op save is not a write. */
  const changedFields = () => {
    if (!profile) return {};
    const changes: Record<string, string> = {};
    if (form.username.trim() && form.username.trim() !== profile.username) {
      changes.username = form.username.trim();
    }
    if (form.email.trim() && form.email.trim() !== profile.email) changes.email = form.email.trim();
    if (form.full_name.trim() !== (profile.full_name ?? "")) changes.full_name = form.full_name.trim();
    if (form.account_type && form.account_type !== profile.account_type) {
      changes.account_type = form.account_type;
    }
    return changes;
  };

  const handleProfileSave = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!token) return;

    const changes = changedFields();
    if (Object.keys(changes).length === 0) {
      toast({ title: "Nothing to save", description: "You haven't changed anything yet." });
      return;
    }

    setIsSavingProfile(true);
    try {
      const updated = await api.updateProfile(changes, token);
      setProfile(updated);

      // A rename invalidates the current token, so the API returns a
      // replacement. Store it, or the next request signs the user out.
      updateSession({
        username: updated.username,
        accountType: updated.account_type,
        ...(updated.access_token ? { token: updated.access_token } : {}),
      });

      toast({
        title: "Saved",
        description: updated.access_token
          ? "Your profile and username have been updated."
          : "Your profile has been updated.",
      });
    } catch (error) {
      toast({
        title: "Could not save",
        description: error instanceof ApiError ? error.message : "Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSavingProfile(false);
    }
  };

  const handlePasswordChange = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!token) return;

    if (passwords.next !== passwords.confirm) {
      toast({
        title: "Passwords don't match",
        description: "The new password and its confirmation are different.",
        variant: "destructive",
      });
      return;
    }

    setIsSavingPassword(true);
    try {
      await api.changePassword(passwords.current, passwords.next, token);
      setPasswords({ current: "", next: "", confirm: "" });
      toast({
        title: "Password changed",
        description: "Use your new password the next time you sign in.",
      });
    } catch (error) {
      toast({
        title: "Could not change password",
        description: error instanceof ApiError ? error.message : "Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSavingPassword(false);
    }
  };

  const handleDeactivate = async () => {
    if (!token) return;
    setIsDeactivating(true);
    try {
      await api.deactivateAccount(token);
      toast({
        title: "Account deactivated",
        description: "Your records are kept. Contact support to reopen your account.",
      });
      signOut();
      navigate("/", { replace: true });
    } catch (error) {
      toast({
        title: "Could not deactivate",
        description: error instanceof ApiError ? error.message : "Please try again.",
        variant: "destructive",
      });
      setIsDeactivating(false);
    }
  };

  const handleDelete = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!token) return;

    setIsDeleting(true);
    try {
      const summary = await api.deleteAccount(
        deleteForm.password,
        deleteForm.confirmation.trim(),
        token,
      );
      toast({
        title: "Account deleted",
        description: `${summary.vitals_records_deleted} readings and ${summary.conversations_deleted} conversations were removed.`,
      });
      signOut();
      navigate("/", { replace: true });
    } catch (error) {
      toast({
        title: "Could not delete account",
        description: error instanceof ApiError ? error.message : "Please try again.",
        variant: "destructive",
      });
      setIsDeleting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="mx-auto max-w-2xl space-y-6">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-52 w-full rounded-xl" />
        ))}
      </div>
    );
  }

  if (loadError || !profile) {
    return (
      <Card className="mx-auto max-w-2xl border-destructive/40">
        <CardContent className="flex items-start gap-3 p-6">
          <Icon name="alert" size={18} className="mt-0.5 text-destructive" />
          <p className="text-sm leading-relaxed">{loadError ?? "Could not load your account."}</p>
        </CardContent>
      </Card>
    );
  }

  const deleteReady =
    deleteForm.password.length > 0 && deleteForm.confirmation.trim() === DELETE_PHRASE;

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      {/* ── Profile ─────────────────────────────────────────────────────── */}
      <Card className="border-2 border-primary/10 shadow-lg">
        <CardHeader className="border-b bg-gradient-to-r from-primary/5 to-purple-500/5">
          <CardTitle className="flex items-center gap-3 text-2xl">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-blue-600 text-primary-foreground">
              <Icon name="age" size={22} />
            </span>
            Your details
          </CardTitle>
          <CardDescription className="text-base">
            Member since {formatDateTime(profile.created_at)}
          </CardDescription>
        </CardHeader>

        <CardContent className="pt-6">
          <form onSubmit={handleProfileSave} className="space-y-5" noValidate>
            <div className="space-y-2">
              <Label htmlFor="set-username">Username</Label>
              <Input
                id="set-username"
                value={form.username}
                onChange={(e) => setForm({ ...form, username: e.target.value })}
                disabled={isSavingProfile}
                autoCapitalize="none"
                className="h-11"
              />
              <p className="text-xs text-muted-foreground">
                This is what you sign in with. Changing it signs you in again automatically.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="set-fullname">Full name</Label>
              <Input
                id="set-fullname"
                value={form.full_name}
                onChange={(e) => setForm({ ...form, full_name: e.target.value })}
                disabled={isSavingProfile}
                className="h-11"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="set-email">Email</Label>
              <Input
                id="set-email"
                type="email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                disabled={isSavingProfile}
                autoCapitalize="none"
                className="h-11"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="set-account-type">Account type</Label>
              <Select
                value={form.account_type}
                onValueChange={(value) => setForm({ ...form, account_type: value as AccountType })}
                disabled={isSavingProfile}
              >
                <SelectTrigger id="set-account-type" className="h-11">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ACCOUNT_TYPES.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Guidance is tailored to this, so keep it current.
              </p>
            </div>

            <Button type="submit" className="h-11 w-full" disabled={isSavingProfile}>
              {isSavingProfile ? (
                <>
                  <IconSpinner size={16} className="mr-2" />
                  Saving...
                </>
              ) : (
                "Save changes"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* ── Password ────────────────────────────────────────────────────── */}
      <Card className="border-2 border-primary/10 shadow-lg">
        <CardHeader className="border-b bg-gradient-to-r from-primary/5 to-purple-500/5">
          <CardTitle className="flex items-center gap-3 text-xl">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
              <Icon name="privacy" size={20} className="text-primary" />
            </span>
            Password
          </CardTitle>
        </CardHeader>

        <CardContent className="pt-6">
          <form onSubmit={handlePasswordChange} className="space-y-5" noValidate>
            <div className="space-y-2">
              <Label htmlFor="set-current-password">Current password</Label>
              <Input
                id="set-current-password"
                type="password"
                autoComplete="current-password"
                value={passwords.current}
                onChange={(e) => setPasswords({ ...passwords, current: e.target.value })}
                disabled={isSavingPassword}
                className="h-11"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="set-new-password">New password</Label>
              <Input
                id="set-new-password"
                type="password"
                autoComplete="new-password"
                value={passwords.next}
                onChange={(e) => setPasswords({ ...passwords, next: e.target.value })}
                disabled={isSavingPassword}
                className="h-11"
              />
              <p className="text-xs text-muted-foreground">
                At least 8 characters, mixing letters with numbers or symbols.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="set-confirm-password">Confirm new password</Label>
              <Input
                id="set-confirm-password"
                type="password"
                autoComplete="new-password"
                value={passwords.confirm}
                onChange={(e) => setPasswords({ ...passwords, confirm: e.target.value })}
                disabled={isSavingPassword}
                className="h-11"
              />
            </div>

            <Button
              type="submit"
              variant="outline"
              className="h-11 w-full"
              disabled={
                isSavingPassword || !passwords.current || !passwords.next || !passwords.confirm
              }
            >
              {isSavingPassword ? (
                <>
                  <IconSpinner size={16} className="mr-2" />
                  Changing...
                </>
              ) : (
                "Change password"
              )}
            </Button>

            <p className="text-xs text-muted-foreground">
              Sessions already signed in elsewhere stay active until they expire.
            </p>
          </form>
        </CardContent>
      </Card>

      {/* ── Leaving ─────────────────────────────────────────────────────── */}
      <Card className="border-2 border-destructive/30 shadow-lg">
        <CardHeader className="border-b bg-destructive/5">
          <CardTitle className="flex items-center gap-3 text-xl">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-destructive/10">
              <Icon name="alert" size={20} className="text-destructive" />
            </span>
            Closing your account
          </CardTitle>
          <CardDescription className="text-base">
            Two options, and only one of them can be undone.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-5 pt-6">
          <div className="rounded-xl border p-5">
            <h3 className="font-semibold">Deactivate</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
              Signing in stops working, but your readings and conversations are kept. If you come
              back, your history is still here.
            </p>
            <Button
              variant="outline"
              className="mt-4"
              onClick={handleDeactivate}
              disabled={isDeactivating || isDeleting}
            >
              {isDeactivating ? (
                <>
                  <IconSpinner size={16} className="mr-2" />
                  Deactivating...
                </>
              ) : (
                "Deactivate my account"
              )}
            </Button>
          </div>

          <div className={cn("rounded-xl border p-5", showDelete && "border-destructive/40 bg-destructive/5")}>
            <h3 className="font-semibold text-destructive">Delete permanently</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
              Your account and <strong>every reading and conversation</strong> are erased. This
              cannot be undone, and nothing can be recovered afterwards.
            </p>

            {!showDelete ? (
              <Button
                variant="outline"
                className="mt-4 border-destructive/40 text-destructive hover:bg-destructive/10 hover:text-destructive"
                onClick={() => setShowDelete(true)}
                disabled={isDeactivating}
              >
                Delete my account
              </Button>
            ) : (
              <form onSubmit={handleDelete} className="mt-5 space-y-4" noValidate>
                <div className="space-y-2">
                  <Label htmlFor="del-password">Confirm your password</Label>
                  <Input
                    id="del-password"
                    type="password"
                    autoComplete="current-password"
                    value={deleteForm.password}
                    onChange={(e) => setDeleteForm({ ...deleteForm, password: e.target.value })}
                    disabled={isDeleting}
                    className="h-11"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="del-confirm">
                    Type <code className="rounded bg-muted px-1.5 py-0.5">{DELETE_PHRASE}</code> to
                    confirm
                  </Label>
                  <Input
                    id="del-confirm"
                    value={deleteForm.confirmation}
                    onChange={(e) => setDeleteForm({ ...deleteForm, confirmation: e.target.value })}
                    disabled={isDeleting}
                    placeholder={DELETE_PHRASE}
                    className="h-11"
                  />
                </div>

                <div className="flex flex-col gap-2 sm:flex-row">
                  <Button
                    type="submit"
                    variant="destructive"
                    className="h-11 flex-1"
                    disabled={!deleteReady || isDeleting}
                  >
                    {isDeleting ? (
                      <>
                        <IconSpinner size={16} className="mr-2" />
                        Deleting...
                      </>
                    ) : (
                      "Permanently delete everything"
                    )}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    className="h-11"
                    onClick={() => {
                      setShowDelete(false);
                      setDeleteForm({ password: "", confirmation: "" });
                    }}
                    disabled={isDeleting}
                  >
                    Cancel
                  </Button>
                </div>
              </form>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AccountSettings;
