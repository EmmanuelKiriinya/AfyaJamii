/**
 * Downloads the Icons8 icons used by the interface into src/assets/icons.
 *
 * The icons are vendored rather than hot-linked so the app renders offline,
 * survives an Icons8 outage or rate limit, and makes no third-party request
 * when a user loads a page containing their health data.
 *
 * Icons8's free tier serves PNG only (the SVG endpoint answers 403
 * PAID_FORMAT), so these are PNGs at 3x the largest rendered size and are
 * painted through a CSS mask so they still follow the current text colour.
 * See src/components/Icon.tsx.
 *
 * Usage:  npm run icons:fetch
 *
 * Licence: the free tier requires visible attribution to icons8.com, which
 * the site footer carries. See ATTRIBUTION.md.
 */

import { mkdir, writeFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(HERE, "..", "src", "assets", "icons");

/** Icons8 style to pull from. `fluency-systems-regular` is a clean line set. */
const STYLE = "fluency-systems-regular";
const SIZE = 96;

/**
 * Icon name as used in the app -> Icons8 slug.
 * Keep this in step with ICON_NAMES in src/lib/icons.ts.
 */
const ICONS = {
  "heart-rate": "heart-with-pulse",
  "blood-pressure": "pulse",
  "blood-sugar": "blood-sample",
  thermometer: "thermometer",
  person: "user",
  clipboard: "clipboard-list",
  chat: "chat-message",
  history: "time-machine",
  logout: "exit",
  send: "sent",
  check: "checkmark",
  warning: "error",
  alert: "high-priority",
  info: "info",
  settings: "settings",
  accessibility: "accessibility2",
  close: "delete-sign",
  "text-larger": "increase-font",
  "text-smaller": "decrease-font",
  contrast: "contrast",
  moon: "moon-symbol",
  sun: "sun",
  monitor: "monitor",
  shield: "security-checked",
  leaf: "natural-food",
  phone: "phone",
  calendar: "calendar",
  "arrow-right": "long-arrow-right",
  stethoscope: "stethoscope",
  baby: "baby",
  pregnant: "pregnant",
  spinner: "spinner-frame-5",
  "chevron-down": "chevron-down",
  "chevron-up": "chevron-up",
};

async function download(name, slug) {
  const url = `https://img.icons8.com/${STYLE}/${SIZE}/${slug}.png`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`${name} (${slug}): HTTP ${response.status} from ${url}`);
  }

  const type = response.headers.get("content-type") ?? "";
  if (!type.startsWith("image/")) {
    // Icons8 answers 200 with a JSON error body for paid formats.
    throw new Error(`${name} (${slug}): expected an image, got ${type}`);
  }

  const bytes = Buffer.from(await response.arrayBuffer());
  await writeFile(join(OUT_DIR, `${name}.png`), bytes);
  return bytes.length;
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });

  const entries = Object.entries(ICONS);
  const failures = [];
  let downloaded = 0;
  let skipped = 0;

  for (const [name, slug] of entries) {
    const target = join(OUT_DIR, `${name}.png`);

    if (existsSync(target) && !process.argv.includes("--force")) {
      skipped += 1;
      continue;
    }

    try {
      const size = await download(name, slug);
      downloaded += 1;
      console.log(`  ${name.padEnd(16)} ${slug.padEnd(24)} ${size} bytes`);
    } catch (error) {
      failures.push(error.message);
    }
  }

  console.log(
    `\n${downloaded} downloaded, ${skipped} already present, ${failures.length} failed.`,
  );

  if (failures.length > 0) {
    console.error("\nFailures:");
    for (const message of failures) console.error(`  - ${message}`);
    process.exitCode = 1;
  }
}

main();
