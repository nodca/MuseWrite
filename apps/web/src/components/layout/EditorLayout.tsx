import type { ReactNode } from "react";

export type EditorLayoutProps = {
  sidebar: ReactNode;
  toolbar: ReactNode;
  editor: ReactNode;
  statusBar: ReactNode;
};

export function EditorLayout({ sidebar, toolbar, editor, statusBar }: EditorLayoutProps) {
  return (
    <div className="flex h-[calc(100vh-56px)]">
      {sidebar}
      <div className="flex flex-1 flex-col min-w-0">
        {toolbar}
        <div className="flex-1 overflow-y-auto">
          {editor}
        </div>
        {statusBar}
      </div>
    </div>
  );
}
