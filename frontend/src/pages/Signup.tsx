import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { IconSpinner } from "@/components/Icon";
import { BrandLink } from "@/components/BrandLink";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { api, ApiError, type AccountType } from "@/lib/api";

const Signup = () => {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    full_name: "",
    password: "",
    confirmPassword: "",
    account_type: "" as AccountType | "",
  });
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();
  const { signIn } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (
      !formData.username.trim() ||
      !formData.email.trim() ||
      !formData.full_name.trim() ||
      !formData.password ||
      !formData.account_type
    ) {
      toast({
        title: "Error",
        description: "Please fill in all fields",
        variant: "destructive",
      });
      return;
    }

    if (formData.password.length < 8) {
      toast({
        title: "Error",
        description: "Password must be at least 8 characters long",
        variant: "destructive",
      });
      return;
    }

    // Mirrors the backend rule, so the failure is caught before a round trip.
    if (/^\d+$/.test(formData.password) || /^[a-zA-Z]+$/.test(formData.password)) {
      toast({
        title: "Error",
        description: "Password must mix letters with numbers or symbols",
        variant: "destructive",
      });
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      toast({
        title: "Error",
        description: "Passwords do not match",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      await api.signup({
        username: formData.username.trim(),
        email: formData.email.trim(),
        full_name: formData.full_name.trim(),
        password: formData.password,
        account_type: formData.account_type as AccountType,
      });
    } catch (error) {
      toast({
        title: "Signup Failed",
        description: error instanceof ApiError ? error.message : "Failed to create account",
        variant: "destructive",
      });
      setIsLoading(false);
      return;
    }

    // The account exists from here on. If the automatic sign-in fails (a rate
    // limit, a dropped connection), reporting "Signup Failed" would send the
    // user back to sign up again, and the retry would hit a 409 for the
    // username or email they have just registered.
    try {
      const session = await api.login({
        username: formData.username.trim(),
        password: formData.password,
      });
      // Backfill from what was submitted — see the note in Login.tsx.
      signIn({
        ...session,
        username: session.username || formData.username.trim(),
        account_type: session.account_type || (formData.account_type as AccountType),
      });

      toast({
        title: "Success",
        description: "Account created successfully!",
      });
      navigate("/dashboard", { replace: true });
    } catch (error) {
      toast({
        title: "Account created",
        description:
          error instanceof ApiError && error.status === 429
            ? "Your account is ready. Please wait a minute, then sign in."
            : "Your account is ready. Please sign in.",
      });
      navigate("/login", { replace: true, state: { username: formData.username.trim() } });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-100 via-blue-50 to-purple-100 dark:from-green-950/40 dark:via-blue-950/20 dark:to-purple-950/40 p-4 py-12">
      <Card className="w-full max-w-md shadow-2xl border-2 border-primary/10">
        <CardHeader className="space-y-4 text-center pb-6">
          <div className="flex justify-center">
            <BrandLink size={80} showName={false} />
          </div>
          <div>
            <CardTitle className="text-3xl font-bold">Create Account</CardTitle>
            <CardDescription className="text-base mt-2">
              Join AfyaJamii for personalised maternal health support
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="pb-8">
          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            <div className="space-y-2">
              <Label htmlFor="username" className="text-sm font-medium">Username</Label>
              <Input
                id="username"
                type="text"
                autoComplete="username"
                autoCapitalize="none"
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                placeholder="Choose a username"
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email" className="text-sm font-medium">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                autoCapitalize="none"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                placeholder="your@email.com"
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="full_name" className="text-sm font-medium">Full Name</Label>
              <Input
                id="full_name"
                type="text"
                autoComplete="name"
                value={formData.full_name}
                onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
                placeholder="Enter your full name"
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="account_type" className="text-sm font-medium">Account Type</Label>
              <Select
                value={formData.account_type}
                onValueChange={(value) =>
                  setFormData({ ...formData, account_type: value as AccountType })
                }
                disabled={isLoading}
              >
                <SelectTrigger className="h-11" id="account_type">
                  <SelectValue placeholder="Select your account type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pregnant">Pregnant</SelectItem>
                  <SelectItem value="postnatal">Postnatal</SelectItem>
                  <SelectItem value="general">General Health</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Choose the option that best describes your current situation
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="password" className="text-sm font-medium">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="new-password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                placeholder="At least 8 characters"
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirmPassword" className="text-sm font-medium">Confirm Password</Label>
              <Input
                id="confirmPassword"
                type="password"
                autoComplete="new-password"
                value={formData.confirmPassword}
                onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                placeholder="Re-enter your password"
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <Button
              type="submit"
              className="w-full h-11 text-base font-medium mt-6"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <IconSpinner size={16} className="mr-2" />
                  Creating account...
                </>
              ) : (
                "Create Account"
              )}
            </Button>
          </form>
          <div className="mt-6 text-center text-sm">
            Already have an account?{" "}
            <Link to="/login" className="text-primary hover:underline font-semibold">
              Sign in
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Signup;
