import React from "react";

type ErrorBoundaryProps = {
  fallbackTitle: string;
  fallbackDescription: string;
  children: React.ReactNode;
};

type ErrorBoundaryState = {
  hasError: boolean;
};

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error): void {
    if (import.meta.env.DEV) {
      console.error("UI error boundary captured an error:", error);
    }
  }

  render(): React.ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }
    return (
      <section className="panel error-boundary-fallback" role="alert" aria-live="assertive">
        <div className="panel-title">
          <h2>{this.props.fallbackTitle}</h2>
          <small>已自动降级</small>
        </div>
        <p>{this.props.fallbackDescription}</p>
      </section>
    );
  }
}

