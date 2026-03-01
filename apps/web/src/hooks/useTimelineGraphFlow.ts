import { useEffect, useMemo, useState } from "react";
import type { GraphTimelineSnapshot } from "../types";

type TimelineRelationFilter = "all" | string;

type UseTimelineGraphFlowArgs = {
  graphTimeline: GraphTimelineSnapshot | null;
  graphTimelineChapterIndex: number;
  maxChapterIndex: number;
};

export function useTimelineGraphFlow({
  graphTimeline,
  graphTimelineChapterIndex,
  maxChapterIndex,
}: UseTimelineGraphFlowArgs) {
  const [timelineEntityQuery, setTimelineEntityQuery] = useState("");
  const [timelineRelationFilter, setTimelineRelationFilter] = useState<TimelineRelationFilter>("all");
  const [selectedTimelineNodeId, setSelectedTimelineNodeId] = useState<string | null>(null);

  const timelineNodesRaw = graphTimeline?.nodes ?? [];
  const timelineEdgesRaw = graphTimeline?.edges ?? [];

  const timelineRelationOptions = useMemo(
    () =>
      Array.from(
        new Set(
          timelineEdgesRaw
            .map((item) => String(item.relation || "").trim().toUpperCase())
            .filter((item) => item.length > 0)
        )
      ).sort((left, right) => left.localeCompare(right, "zh-Hans-CN")),
    [timelineEdgesRaw]
  );

  useEffect(() => {
    if (timelineRelationFilter === "all") return;
    if (timelineRelationOptions.includes(timelineRelationFilter)) return;
    setTimelineRelationFilter("all");
  }, [timelineRelationFilter, timelineRelationOptions]);

  const timelineQuery = timelineEntityQuery.trim().toLowerCase();
  const graphEdgesFiltered = useMemo(
    () =>
      timelineEdgesRaw.filter((edge) => {
        const relation = String(edge.relation || "").trim().toUpperCase();
        if (timelineRelationFilter !== "all" && relation !== timelineRelationFilter) {
          return false;
        }
        if (!timelineQuery) return true;
        const source = String(edge.source || "").toLowerCase();
        const target = String(edge.target || "").toLowerCase();
        return (
          source.includes(timelineQuery) ||
          target.includes(timelineQuery) ||
          relation.toLowerCase().includes(timelineQuery)
        );
      }),
    [timelineEdgesRaw, timelineRelationFilter, timelineQuery]
  );

  const graphCandidateNodeIds = useMemo(() => {
    const ids = new Set<string>();
    graphEdgesFiltered.forEach((edge) => {
      if (edge.source) ids.add(String(edge.source));
      if (edge.target) ids.add(String(edge.target));
    });
    return ids;
  }, [graphEdgesFiltered]);

  const graphNodesFiltered = useMemo(() => {
    if (!timelineQuery && timelineRelationFilter === "all") {
      return timelineNodesRaw;
    }
    return timelineNodesRaw.filter((node) => {
      const id = String(node.id || "");
      const label = String(node.label || "").toLowerCase();
      if (timelineQuery) {
        if (label.includes(timelineQuery) || id.toLowerCase().includes(timelineQuery)) {
          return true;
        }
      }
      return graphCandidateNodeIds.has(id);
    });
  }, [timelineNodesRaw, timelineQuery, timelineRelationFilter, graphCandidateNodeIds]);

  const graphNodes = useMemo(() => graphNodesFiltered.slice(0, 18), [graphNodesFiltered]);
  const graphNodeIdSet = useMemo(() => new Set(graphNodes.map((item) => item.id)), [graphNodes]);

  const graphEdges = useMemo(
    () =>
      graphEdgesFiltered
        .filter((item) => graphNodeIdSet.has(item.source) && graphNodeIdSet.has(item.target))
        .slice(0, 32),
    [graphEdgesFiltered, graphNodeIdSet]
  );

  useEffect(() => {
    if (!selectedTimelineNodeId) return;
    if (graphNodeIdSet.has(selectedTimelineNodeId)) return;
    setSelectedTimelineNodeId(null);
  }, [selectedTimelineNodeId, graphNodeIdSet]);

  const highlightedEdgeIdSet = useMemo(() => {
    if (!selectedTimelineNodeId) return new Set<string>();
    const ids = new Set<string>();
    graphEdges.forEach((edge) => {
      if (edge.source === selectedTimelineNodeId || edge.target === selectedTimelineNodeId) {
        ids.add(edge.id);
      }
    });
    return ids;
  }, [graphEdges, selectedTimelineNodeId]);

  const highlightedNodeIdSet = useMemo(() => {
    if (!selectedTimelineNodeId) return new Set<string>();
    const ids = new Set<string>([selectedTimelineNodeId]);
    graphEdges.forEach((edge) => {
      if (edge.source === selectedTimelineNodeId) {
        ids.add(edge.target);
      } else if (edge.target === selectedTimelineNodeId) {
        ids.add(edge.source);
      }
    });
    return ids;
  }, [graphEdges, selectedTimelineNodeId]);

  const selectedTimelineNodeLabel = useMemo(() => {
    if (!selectedTimelineNodeId) return "";
    const node = graphNodes.find((item) => item.id === selectedTimelineNodeId);
    return String(node?.label || selectedTimelineNodeId);
  }, [graphNodes, selectedTimelineNodeId]);

  const graphLayout = useMemo(() => {
    const width = 340;
    const height = 220;
    const centerX = width / 2;
    const centerY = height / 2;
    const count = graphNodes.length;
    const radius = Math.max(42, Math.min(92, 44 + count * 1.4));
    const positions: Record<string, { x: number; y: number }> = {};

    if (count <= 0) {
      return { width, height, positions };
    }
    if (count === 1) {
      positions[graphNodes[0].id] = { x: centerX, y: centerY };
      return { width, height, positions };
    }

    graphNodes.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / count - Math.PI / 2;
      positions[node.id] = {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      };
    });

    return { width, height, positions };
  }, [graphNodes]);

  const sliderValue = Math.max(
    1,
    Math.min(
      Math.max(1, maxChapterIndex),
      graphTimelineChapterIndex || Number(graphTimeline?.chapter_index || 0) || 1
    )
  );

  const timelineStats = graphTimeline?.stats ?? {};
  const timelineNodesTotal = Number(timelineStats.nodes ?? timelineNodesRaw.length ?? 0) || 0;
  const timelineEdgesTotal = Number(timelineStats.edges ?? timelineEdgesRaw.length ?? 0) || 0;
  const timelineNodesCount = graphNodes.length;
  const timelineEdgesCount = graphEdges.length;
  const hasTimelineFilter = timelineQuery.length > 0 || timelineRelationFilter !== "all";

  return {
    timelineEntityQuery,
    setTimelineEntityQuery,
    timelineRelationFilter,
    setTimelineRelationFilter,
    selectedTimelineNodeId,
    setSelectedTimelineNodeId,
    timelineRelationOptions,
    graphNodes,
    graphEdges,
    highlightedEdgeIdSet,
    highlightedNodeIdSet,
    selectedTimelineNodeLabel,
    graphLayout,
    sliderValue,
    timelineNodesTotal,
    timelineEdgesTotal,
    timelineNodesCount,
    timelineEdgesCount,
    hasTimelineFilter,
  };
}
