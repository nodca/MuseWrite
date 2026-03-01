type ModeSwitchProps = {
  uiMode: "writing" | "pro";
  onToggle: () => void;
  disabled?: boolean;
};

export function ModeSwitch({ uiMode, onToggle, disabled = false }: ModeSwitchProps) {
  return (
    <button className="btn ghost" onClick={onToggle} disabled={disabled} data-testid="ui-mode-toggle">
      {uiMode === "writing" ? "切到工作台模式" : "切到写作模式"}
    </button>
  );
}
