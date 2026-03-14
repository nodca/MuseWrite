import { Sun, Moon, Palette, Check } from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import type { Theme } from "../../hooks/useTheme";
import { IconButton } from "./IconButton";

type ThemeSwitcherProps = {
  theme: Theme;
  setTheme: (t: Theme) => void;
};

const THEME_OPTIONS: { value: Theme; label: string; icon: typeof Sun }[] = [
  { value: "modern", label: "现代清爽", icon: Sun },
  { value: "warm", label: "暖色纸感", icon: Palette },
  { value: "dark", label: "暗夜模式", icon: Moon },
];

function currentIcon(theme: Theme) {
  switch (theme) {
    case "modern":
      return <Sun size={18} />;
    case "warm":
      return <Palette size={18} />;
    case "dark":
      return <Moon size={18} />;
  }
}

export function ThemeSwitcher({ theme, setTheme }: ThemeSwitcherProps) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <IconButton icon={currentIcon(theme)} label="切换主题" />
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          sideOffset={6}
          align="end"
          className="z-50 min-w-[140px] bg-surface-primary border border-border-default shadow-lg rounded-lg p-1 animate-in fade-in-0 zoom-in-95"
        >
          {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
            <DropdownMenu.Item
              key={value}
              onSelect={() => setTheme(value)}
              className="flex items-center gap-2 px-3 py-2 text-sm text-text-primary cursor-pointer rounded outline-none hover:bg-surface-elevated focus:bg-surface-elevated"
            >
              <Icon size={16} className="text-text-secondary" />
              <span className="flex-1">{label}</span>
              {theme === value && (
                <Check size={14} className="text-accent-primary" />
              )}
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
