/**
 * The icon set.
 *
 * Icons are from Icons8 (https://icons8.com), vendored into the repository by
 * `npm run icons:fetch` rather than loaded from their CDN, so that pages
 * carrying a user's health data make no third-party request and the interface
 * still renders offline.
 *
 * Each entry maps a semantic name — what the icon *means* here, not what it
 * depicts — to its file. Naming them by role keeps call sites readable and
 * means swapping artwork later touches only this file.
 */

import accessibility from "@/assets/icons/accessibility.png";
import alert from "@/assets/icons/alert.png";
import arrowRight from "@/assets/icons/arrow-right.png";
import baby from "@/assets/icons/baby.png";
import bloodPressure from "@/assets/icons/blood-pressure.png";
import bloodSugar from "@/assets/icons/blood-sugar.png";
import calendar from "@/assets/icons/calendar.png";
import chat from "@/assets/icons/chat.png";
import check from "@/assets/icons/check.png";
import chevronDown from "@/assets/icons/chevron-down.png";
import chevronUp from "@/assets/icons/chevron-up.png";
import clipboard from "@/assets/icons/clipboard.png";
import close from "@/assets/icons/close.png";
import contrast from "@/assets/icons/contrast.png";
import heartRate from "@/assets/icons/heart-rate.png";
import history from "@/assets/icons/history.png";
import info from "@/assets/icons/info.png";
import leaf from "@/assets/icons/leaf.png";
import logout from "@/assets/icons/logout.png";
import monitor from "@/assets/icons/monitor.png";
import moon from "@/assets/icons/moon.png";
import person from "@/assets/icons/person.png";
import phone from "@/assets/icons/phone.png";
import pregnant from "@/assets/icons/pregnant.png";
import send from "@/assets/icons/send.png";
import settings from "@/assets/icons/settings.png";
import shield from "@/assets/icons/shield.png";
import spinner from "@/assets/icons/spinner.png";
import stethoscope from "@/assets/icons/stethoscope.png";
import sun from "@/assets/icons/sun.png";
import textLarger from "@/assets/icons/text-larger.png";
import textSmaller from "@/assets/icons/text-smaller.png";
import thermometer from "@/assets/icons/thermometer.png";
import warning from "@/assets/icons/warning.png";

export const icons = {
  // Vitals
  age: person,
  heartRate,
  bloodPressure,
  bloodSugar,
  temperature: thermometer,

  // Navigation and actions
  vitals: clipboard,
  chat,
  history,
  send,
  logout,
  settings,
  next: arrowRight,
  close,
  chevronDown,
  chevronUp,

  // Status
  check,
  warning,
  alert,
  info,
  spinner,

  // Accessibility panel
  accessibility,
  textLarger,
  textSmaller,
  contrast,
  moon,
  sun,
  monitor,

  // Marketing and account types
  care: stethoscope,
  privacy: shield,
  nutrition: leaf,
  emergency: phone,
  schedule: calendar,
  pregnant,
  postnatal: baby,
  general: heartRate,
} as const;

export type IconName = keyof typeof icons;

/**
 * Provenance of the icon set.
 *
 * No longer rendered — the footer link was removed by request. Note that the
 * Icons8 *free* licence requires visible attribution, so displaying this
 * somewhere is a condition of that tier; a paid Icons8 plan lifts the
 * requirement. Kept here so the source of the artwork stays recorded.
 */
export const ICON_ATTRIBUTION = {
  label: "Icons by Icons8",
  href: "https://icons8.com",
} as const;
