import { Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import logo from "@/assets/logo.png";

/**
 * The AfyaJamii logo, as a link home.
 *
 * Where "home" points depends on who is looking: a signed-in user goes to
 * their dashboard overview, everyone else to the landing page. That matches
 * the convention people already expect from the logo in a web app, and saves
 * a signed-in user a bounce through the marketing page to get back to their
 * readings.
 *
 * The dashboard does not use this component — its logo switches the active
 * view in place rather than navigating, since the dashboard is a single route.
 */

interface BrandLinkProps {
  className?: string;
  /** Pixel size of the mark. */
  size?: number;
  /** Show the "AfyaJamii" wordmark beside the mark. */
  showName?: boolean;
  /** Extra classes for the wordmark. */
  nameClassName?: string;
}

export function BrandLink({
  className,
  size = 40,
  showName = true,
  nameClassName,
}: BrandLinkProps) {
  const { isAuthenticated } = useAuth();
  const target = isAuthenticated ? "/dashboard" : "/";

  return (
    <Link
      to={target}
      aria-label={isAuthenticated ? "AfyaJamii, go to your dashboard" : "AfyaJamii, go to the home page"}
      className={cn(
        "flex items-center gap-3 rounded-md transition-opacity hover:opacity-80",
        className,
      )}
    >
      <img src={logo} alt="" aria-hidden style={{ height: size, width: size }} />
      {showName ? (
        <span className={cn("text-xl font-bold", nameClassName)}>AfyaJamii</span>
      ) : null}
    </Link>
  );
}
