export type Theme = "system" | "light" | "dark";

export const THEME_COOKIE = "nano_dashboard_theme";

export function parseTheme(value: string | undefined): Theme {
  return value === "light" || value === "dark" ? value : "system";
}
