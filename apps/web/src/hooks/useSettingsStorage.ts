import { useEffect, useMemo } from "react";
import type { ModelProfile } from "../types";

export type UseSettingsStorageArgs = {
  projectId: number;
  modelProfiles: ModelProfile[];
  suggestionModelProfileId: string | null;
  setSuggestionModelProfileId: (value: string | null) => void;
  chatTemperatureProfile: "action" | "chat" | "brainstorm";
  setChatTemperatureProfile: (value: "action" | "chat" | "brainstorm") => void;
  suggestionTemperatureProfile: "suggestion" | "chat" | "action" | "brainstorm";
  setSuggestionTemperatureProfile: (value: "suggestion" | "chat" | "action" | "brainstorm") => void;
};

export function useSettingsStorage({
  projectId,
  modelProfiles,
  suggestionModelProfileId,
  setSuggestionModelProfileId,
  chatTemperatureProfile,
  setChatTemperatureProfile,
  suggestionTemperatureProfile,
  setSuggestionTemperatureProfile,
}: UseSettingsStorageArgs): void {
  const suggestionModelProfileStorageKey = useMemo(() => {
    return `novel-platform:suggestion-model-profile:v1:${projectId}`;
  }, [projectId]);
  const chatTemperatureProfileStorageKey = useMemo(() => {
    return `novel-platform:chat-temperature-profile:v1:${projectId}`;
  }, [projectId]);
  const suggestionTemperatureProfileStorageKey = useMemo(() => {
    return `novel-platform:suggestion-temperature-profile:v1:${projectId}`;
  }, [projectId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    let stored: string | null = null;
    try {
      const raw = window.localStorage.getItem(suggestionModelProfileStorageKey);
      stored = raw && raw.trim() ? raw.trim() : null;
    } catch {
      stored = null;
    }
    if (stored && !modelProfiles.some((item) => item.profile_id === stored)) {
      stored = null;
    }
    setSuggestionModelProfileId(stored);
  }, [suggestionModelProfileStorageKey, modelProfiles]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      if (!suggestionModelProfileId) {
        window.localStorage.removeItem(suggestionModelProfileStorageKey);
        return;
      }
      window.localStorage.setItem(suggestionModelProfileStorageKey, suggestionModelProfileId);
    } catch {
      // ignore localStorage quota/permission failures
    }
  }, [suggestionModelProfileId, suggestionModelProfileStorageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = (window.localStorage.getItem(chatTemperatureProfileStorageKey) || "").trim();
      if (raw === "action" || raw === "chat" || raw === "brainstorm") {
        setChatTemperatureProfile(raw);
      }
    } catch {
      // ignore localStorage quota/permission failures
    }
  }, [chatTemperatureProfileStorageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(chatTemperatureProfileStorageKey, chatTemperatureProfile);
    } catch {
      // ignore localStorage quota/permission failures
    }
  }, [chatTemperatureProfile, chatTemperatureProfileStorageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = (window.localStorage.getItem(suggestionTemperatureProfileStorageKey) || "").trim();
      if (raw === "suggestion" || raw === "chat" || raw === "action" || raw === "brainstorm") {
        setSuggestionTemperatureProfile(raw);
      }
    } catch {
      // ignore localStorage quota/permission failures
    }
  }, [suggestionTemperatureProfileStorageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(suggestionTemperatureProfileStorageKey, suggestionTemperatureProfile);
    } catch {
      // ignore localStorage quota/permission failures
    }
  }, [suggestionTemperatureProfile, suggestionTemperatureProfileStorageKey]);
}
