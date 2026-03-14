import { memo } from "react";
import {
  Feather,
  Maximize2,
  Minimize2,
  PanelRightOpen,
  PanelRightClose,
  RefreshCw,
  Plus,
  Settings,
} from "lucide-react";
import type { Theme } from "../hooks/useTheme";
import { IconButton } from "./shared/IconButton";
import { ThemeSwitcher } from "./shared/ThemeSwitcher";

export type TopBarProps = {
  advancedPanelOpen: boolean;
  zenMode: boolean;
  streaming: boolean;
  settingsDialogOpen: boolean;
  theme: Theme;
  setTheme: (t: Theme) => void;
  onToggleAdvancedPanel: () => void;
  onToggleZenMode: () => void;
  onOpenSettingsDialog: () => void;
  onRefreshSnapshot: () => Promise<void>;
  onStartNewSession: () => void;
};

export const TopBar = memo(function TopBar({
  advancedPanelOpen,
  zenMode,
  streaming,
  settingsDialogOpen,
  theme,
  setTheme,
  onToggleAdvancedPanel,
  onToggleZenMode,
  onOpenSettingsDialog,
  onRefreshSnapshot,
  onStartNewSession,
}: TopBarProps) {
  return (
    <header className="flex items-center justify-between px-5 py-3 border-b border-border-default bg-surface-primary/80 backdrop-blur-sm">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <Feather size={20} className="text-accent-primary" />
        <span className="text-lg font-semibold text-text-primary tracking-tight">
          Novel Platform
        </span>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1">
        <IconButton
          icon={zenMode ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          label={zenMode ? "退出专注" : "专注模式"}
          active={zenMode}
          onClick={onToggleZenMode}
          disabled={streaming || advancedPanelOpen}
          aria-pressed={zenMode}
        />

        <IconButton
          icon={
            advancedPanelOpen ? (
              <PanelRightClose size={18} />
            ) : (
              <PanelRightOpen size={18} />
            )
          }
          label={advancedPanelOpen ? "收起进阶面板" : "展开进阶面板"}
          active={advancedPanelOpen}
          onClick={onToggleAdvancedPanel}
          disabled={streaming}
        />

        <IconButton
          icon={<RefreshCw size={18} />}
          label="刷新快照"
          onClick={() => void onRefreshSnapshot()}
          disabled={streaming}
        />

        <IconButton
          icon={<Plus size={18} />}
          label="新会话"
          onClick={onStartNewSession}
          disabled={streaming}
        />

        <IconButton
          icon={<Settings size={18} />}
          label="写作设置"
          onClick={onOpenSettingsDialog}
          disabled={streaming}
          aria-haspopup="dialog"
          aria-expanded={settingsDialogOpen}
          aria-controls="settings-dialog"
        />

        <ThemeSwitcher theme={theme} setTheme={setTheme} />
      </div>
    </header>
  );
});
