"use client";

import { useState } from "react";
import { parseTheme, THEME_COOKIE, type Theme } from "@/lib/theme";

export default function ThemeSelect({ initialTheme }: { initialTheme: Theme }) {
  const [theme, setTheme] = useState(initialTheme);

  function changeTheme(value: string) {
    const next = parseTheme(value);
    document.documentElement.dataset.theme = next;
    document.cookie = `${THEME_COOKIE}=${next}; Path=/; Max-Age=31536000; SameSite=Lax`;
    setTheme(next);
  }

  return (
    <div className="theme-select">
      <label className="muted small" htmlFor="dashboard-theme">Theme</label>
      <select id="dashboard-theme" value={theme} onChange={(event) => changeTheme(event.target.value)}>
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </div>
  );
}
