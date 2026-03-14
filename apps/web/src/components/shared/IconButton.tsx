import { forwardRef, type ReactNode, type ButtonHTMLAttributes } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import clsx from "clsx";

type IconButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  icon: ReactNode;
  /** Tooltip text */
  label: string;
  /** Visual active / pressed state */
  active?: boolean;
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton({ icon, label, active, className, ...rest }, ref) {
    return (
      <Tooltip.Provider delayDuration={300}>
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <button
              ref={ref}
              type="button"
              aria-label={label}
              className={clsx(
                "rounded-lg p-2 transition-colors",
                "text-text-secondary hover:text-text-primary hover:bg-surface-elevated",
                "disabled:opacity-40 disabled:cursor-not-allowed",
                active && "bg-accent-secondary text-accent-primary",
                className,
              )}
              {...rest}
            >
              {icon}
            </button>
          </Tooltip.Trigger>

          <Tooltip.Portal>
            <Tooltip.Content
              sideOffset={6}
              className="z-50 text-xs bg-surface-elevated text-text-primary px-2 py-1 rounded shadow-md border border-border-default animate-in fade-in-0 zoom-in-95"
            >
              {label}
              <Tooltip.Arrow className="fill-surface-elevated" />
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>
      </Tooltip.Provider>
    );
  },
);
