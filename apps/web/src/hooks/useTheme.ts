import { useState, useEffect, useCallback } from "react";

export type Theme = "modern" | "warm" | "dark";

const STORAGE_KEY = "novel-platform:theme";
const VALID_THEMES: Theme[] = ["modern", "warm", "dark"];

function getStoredTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && VALID_THEMES.includes(stored as Theme)) {
      return stored as Theme;
    }
  } catch {
    // localStorage may be unavailable
  }
  return "modern";
}

function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getStoredTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
    applyTheme(next);
  }, []);

  const cycleTheme = useCallback(() => {
    setThemeState((prev) => {
      const idx = VALID_THEMES.indexOf(prev);
      const next = VALID_THEMES[(idx + 1) % VALID_THEMES.length];
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch {
        // ignore
      }
      applyTheme(next);
      return next;
    });
  }, []);

  return { theme, setTheme, cycleTheme } as const;
}
