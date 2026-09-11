import { cn } from "@/lib/utils";
import { icons, type IconName } from "@/lib/icons";

/**
 * Renders a vendored Icons8 icon.
 *
 * The PNG is used as a CSS mask rather than as an <img>, so the icon takes its
 * colour from `currentColor`. That keeps a single black-line asset usable on
 * any background and lets icons follow the light and dark themes, which a
 * plain <img> could not do.
 *
 * Icons are decorative by default and hidden from assistive technology. Pass
 * `title` for a standalone icon that carries meaning of its own — an icon-only
 * button, for instance — and it will be exposed as an image with that label.
 */

export interface IconProps extends React.HTMLAttributes<HTMLSpanElement> {
  name: IconName;
  /** Rendered size in pixels. Source art is 96px, so it stays crisp. */
  size?: number;
  /** Accessible label. Omit for icons that sit beside their own text. */
  title?: string;
}

export function Icon({ name, size = 20, title, className, style, ...props }: IconProps) {
  const source = icons[name];

  return (
    <span
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      className={cn("inline-block shrink-0 bg-current", className)}
      style={{
        width: size,
        height: size,
        maskImage: `url(${source})`,
        WebkitMaskImage: `url(${source})`,
        maskSize: "contain",
        WebkitMaskSize: "contain",
        maskRepeat: "no-repeat",
        WebkitMaskRepeat: "no-repeat",
        maskPosition: "center",
        WebkitMaskPosition: "center",
        ...style,
      }}
      {...props}
    />
  );
}

/** A spinning icon for pending states. */
export function IconSpinner({ className, title = "Loading", ...props }: Omit<IconProps, "name">) {
  return <Icon name="spinner" title={title} className={cn("animate-spin", className)} {...props} />;
}
