import { useEffect, useMemo, useState } from "react";
import type {
  GraphBlastRadiusPreview,
  GraphTimelineEdge,
  GraphTimelineNode,
  GraphTimelineSnapshot,
} from "../types";

type TimelineRelationFilter = "all" | string;

type UseTimelineGraphFlowArgs = {
  graphTimeline: GraphTimelineSnapshot | null;
  graphTimelineChapterIndex: number;
  maxChapterIndex: number;
  previewBlastRadius?: GraphBlastRadiusPreview | null;
};

export function useTimelineGraphFlow({
  graphTimeline,
  graphTimelineChapterIndex,
  maxChapterIndex,
  previewBlastRadius,
}: UseTimelineGraphFlowArgs) {
  const GRAPH_NODE_LIMIT = 18;
  const GRAPH_EDGE_LIMIT = 32;
  const [timelineEntityQuery, setTimelineEntityQuery] = useState("");
  const [timelineRelationFilter, setTimelineRelationFilter] = useState<TimelineRelationFilter>("all");
  const [selectedTimelineNodeId, setSelectedTimelineNodeId] = useState<string | null>(null);

  const baseTimelineNodesRaw = graphTimeline?.nodes ?? [];
  const baseTimelineEdgesRaw = graphTimeline?.edges ?? [];

  const previewEdgesRaw = useMemo<GraphTimelineEdge[]>(
    () =>
      (previewBlastRadius?.edges ?? [])
        .map((item) => {
          const source = String(item.source || "").trim();
          const target = String(item.target || "").trim();
          const relation = String(item.relation || "").trim().toUpperCase();
          if (!source || !target || !relation) return null;
          const key = String(item.key || `${source}|${relation}|${target}`).trim();
          return {
            id: `preview:${key}`,
            source,
            target,
            relation,
          };
        })
        .filter((item): item is GraphTimelineEdge => item !== null),
    [previewBlastRadius]
  );

  const timelineEdgesRaw = useMemo(() => {
    const merged = new Map<string, GraphTimelineEdge>();
    const pushEdge = (edge: GraphTimelineEdge) => {
      const source = String(edge.source || "").trim();
      const target = String(edge.target || "").trim();
      const relation = String(edge.relation || "").trim().toUpperCase();
      if (!source || !target || !relation) return;
      const key = `${source}|${relation}|${target}`;
      if (merged.has(key)) return;
      merged.set(key, {
        ...edge,
        id: String(edge.id || key),
        source,
        target,
        relation,
      });
    };
    baseTimelineEdgesRaw.forEach(pushEdge);
    previewEdgesRaw.forEach(pushEdge);
    return Array.from(merged.values());
  }, [baseTimelineEdgesRaw, previewEdgesRaw]);

  const timelineNodesRaw = useMemo<GraphTimelineNode[]>(() => {
    const degreeMap = new Map<string, number>();
    timelineEdgesRaw.forEach((edge) => {
      const source = String(edge.source || "");
      const target = String(edge.target || "");
      if (source) degreeMap.set(source, (degreeMap.get(source) ?? 0) + 1);
      if (target) degreeMap.set(target, (degreeMap.get(target) ?? 0) + 1);
    });

    const merged = new Map<string, GraphTimelineNode>();
    baseTimelineNodesRaw.forEach((node) => {
      const id = String(node.id || "").trim();
      if (!id) return;
      const label = String(node.label || id).trim() || id;
      const kindRaw = String(node.kind || "").trim();
      const kind = kindRaw || "entity";
      merged.set(id, {
        id,
        label,
        kind,
        degree: Math.max(Number(node.degree || 0) || 0, degreeMap.get(id) ?? 0),
      });
    });

    (previewBlastRadius?.nodes ?? []).forEach((node) => {
      const id = String(node.id || node.label || "").trim();
      if (!id) return;
      const label = String(node.label || node.id || "").trim() || id;
      const existing = merged.get(id);
      if (existing) {
        const shouldUpgradeLabel =
          (existing.label === existing.id || !existing.label) && label && label !== existing.id;
        merged.set(id, {
          ...existing,
          label: shouldUpgradeLabel ? label : existing.label,
          degree: Math.max(existing.degree, degreeMap.get(id) ?? 0),
        });
        return;
      }
      merged.set(id, {
        id,
        label,
        kind: "preview",
        degree: degreeMap.get(id) ?? 0,
      });
    });
    degreeMap.forEach((degree, id) => {
      const existing = merged.get(id);
      if (existing) {
        if (existing.degree >= degree) return;
        merged.set(id, { ...existing, degree });
        return;
      }
      merged.set(id, {
        id,
        label: id,
        kind: "preview",
        degree,
      });
    });
    return Array.from(merged.values());
  }, [baseTimelineNodesRaw, previewBlastRadius, timelineEdgesRaw]);

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
      if (timelineQuery && (label.includes(timelineQuery) || id.toLowerCase().includes(timelineQuery))) {
        return true;
      }
      return graphCandidateNodeIds.has(id);
    });
  }, [timelineNodesRaw, timelineQuery, timelineRelationFilter, graphCandidateNodeIds]);

  const graphNodesCandidateCount = graphNodesFiltered.length;
  const graphNodesTruncated = graphNodesCandidateCount > GRAPH_NODE_LIMIT;
  const graphNodes = useMemo(() => graphNodesFiltered.slice(0, GRAPH_NODE_LIMIT), [graphNodesFiltered]);
  const graphNodeIdSet = useMemo(() => new Set(graphNodes.map((item) => item.id)), [graphNodes]);

  const graphEdgesCandidate = useMemo(
    () => graphEdgesFiltered.filter((item) => graphNodeIdSet.has(item.source) && graphNodeIdSet.has(item.target)),
    [graphEdgesFiltered, graphNodeIdSet]
  );
  const graphEdgesCandidateCount = graphEdgesCandidate.length;
  const graphEdgesTruncated = graphEdgesCandidateCount > GRAPH_EDGE_LIMIT;
  const graphEdges = useMemo(() => graphEdgesCandidate.slice(0, GRAPH_EDGE_LIMIT), [graphEdgesCandidate]);

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
  const timelineNodesTotal = Math.max(Number(timelineStats.nodes ?? 0) || 0, timelineNodesRaw.length);
  const timelineEdgesTotal = Math.max(Number(timelineStats.edges ?? 0) || 0, timelineEdgesRaw.length);
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
    graphNodeLimit: GRAPH_NODE_LIMIT,
    graphEdgeLimit: GRAPH_EDGE_LIMIT,
    graphNodesCandidateCount,
    graphEdgesCandidateCount,
    graphNodesTruncated,
    graphEdgesTruncated,
  };
}
