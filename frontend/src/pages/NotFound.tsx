import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Icon } from "@/components/Icon";
import { BrandLink } from "@/components/BrandLink";

const NotFound = () => {
  const location = useLocation();

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-100 via-purple-50 to-pink-100 p-4 dark:from-blue-950/40 dark:via-purple-950/20 dark:to-pink-950/40">
      <Card className="w-full max-w-md border-2 border-primary/10 shadow-2xl">
        <CardContent className="px-6 py-12 text-center">
          <BrandLink size={64} showName={false} className="justify-center" />
          <p className="mt-6 text-5xl font-bold text-primary">404</p>
          <h1 className="mt-3 text-2xl font-bold">We couldn&rsquo;t find that page</h1>
          <p className="mt-3 leading-relaxed text-muted-foreground">
            Nothing lives at{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 text-sm">{location.pathname}</code>.
            It may have moved, or the link may be mistyped.
          </p>

          <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:justify-center">
            <Button asChild>
              <Link to="/">
                Go to the home page
                <Icon name="next" size={16} className="ml-2" />
              </Link>
            </Button>
            <Button asChild variant="outline">
              <Link to="/dashboard">Open my dashboard</Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default NotFound;
