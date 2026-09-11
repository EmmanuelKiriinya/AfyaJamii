import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { IconSpinner } from "@/components/Icon";
import { BrandLink } from "@/components/BrandLink";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { api, ApiError } from "@/lib/api";

const Login = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();
  const { signIn } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!username.trim() || !password) {
      toast({
        title: "Error",
        description: "Please fill in all fields",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const session = await api.login({ username: username.trim(), password });
      // The account type comes from the server, so the dashboard shows the
      // right guidance instead of assuming "general".
      signIn(session);
      toast({
        title: "Success",
        description: "Logged in successfully",
      });
      navigate("/dashboard", { replace: true });
    } catch (error) {
      toast({
        title: "Login Failed",
        description: error instanceof ApiError ? error.message : "Invalid credentials",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-100 via-purple-50 to-pink-100 dark:from-blue-950/40 dark:via-purple-950/20 dark:to-pink-950/40 p-4">
      <Card className="w-full max-w-md shadow-2xl border-2 border-primary/10">
        <CardHeader className="space-y-4 text-center pb-8">
          <div className="flex justify-center">
            <BrandLink size={96} showName={false} />
          </div>
          <div>
            <CardTitle className="text-3xl font-bold">Welcome Back</CardTitle>
            <CardDescription className="text-base mt-2">
              Sign in to your AfyaJamii account to continue your health journey
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="pb-8">
          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            <div className="space-y-2">
              <Label htmlFor="username" className="text-sm font-medium">Username</Label>
              <Input
                id="username"
                type="text"
                autoComplete="username"
                autoCapitalize="none"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password" className="text-sm font-medium">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                disabled={isLoading}
                className="h-11"
              />
            </div>
            <Button type="submit" className="w-full h-11 text-base font-medium" disabled={isLoading}>
              {isLoading ? (
                <>
                  <IconSpinner size={16} className="mr-2" />
                  Signing in...
                </>
              ) : (
                "Sign In"
              )}
            </Button>
          </form>
          <div className="mt-6 text-center text-sm">
            Don&rsquo;t have an account?{" "}
            <Link to="/signup" className="text-primary hover:underline font-semibold">
              Sign up for free
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Login;
