/* ******************************************************************************
 *
 *  Copyright (c) 2026 Konduit K.K.
 *
 *  This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

package org.nd4j.autodiff.samediff.execution;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Immutable point-in-time view of the native DSP plan lifecycle.
 *
 * <p>The native {@code PlanLifecycle} is the authority for these values.  Java
 * receives one compact payload and exposes the decoded values without keeping
 * a second mutable phase machine.  A snapshot is intentionally not live; call
 * {@link DynamicShapePlanExecutor#getLifecycleSnapshot()} again when a fresh
 * view is required.</p>
 */
public final class DspLifecycleSnapshot {
    private final boolean valid;
    private final PlanPhase planPhase;
    private final GraphNodePhase graphNodePhase;
    private final int buildStage;
    private final int executionCount;
    private final int postFreezeExecutionCount;
    private final int pointersStableCount;
    private final boolean compilationDone;
    private final int segmentCount;
    private final int buildingSegmentCount;
    private final int sealedSegmentCount;
    private final int failedSegmentCount;

    private DspLifecycleSnapshot(boolean valid,
                                 PlanPhase planPhase,
                                 GraphNodePhase graphNodePhase,
                                 int buildStage,
                                 int executionCount,
                                 int postFreezeExecutionCount,
                                 int pointersStableCount,
                                 boolean compilationDone,
                                 int segmentCount,
                                 int buildingSegmentCount,
                                 int sealedSegmentCount,
                                 int failedSegmentCount) {
        this.valid = valid;
        this.planPhase = planPhase;
        this.graphNodePhase = graphNodePhase;
        this.buildStage = buildStage;
        this.executionCount = executionCount;
        this.postFreezeExecutionCount = postFreezeExecutionCount;
        this.pointersStableCount = pointersStableCount;
        this.compilationDone = compilationDone;
        this.segmentCount = segmentCount;
        this.buildingSegmentCount = buildingSegmentCount;
        this.sealedSegmentCount = sealedSegmentCount;
        this.failedSegmentCount = failedSegmentCount;
    }

    /** Snapshot returned when no native plan is currently installed. */
    public static DspLifecycleSnapshot unavailable() {
        return new DspLifecycleSnapshot(false, null, null, -1, -1, -1, -1,
                false, -1, -1, -1, -1);
    }

    /** Decode the stable native key/value payload. */
    public static DspLifecycleSnapshot fromNativePayload(String payload) {
        if (payload == null || payload.trim().isEmpty()) return unavailable();

        Map<String, String> values = new HashMap<>();
        for (String entry : payload.split(";")) {
            int separator = entry.indexOf('=');
            if (separator <= 0) continue;
            values.put(entry.substring(0, separator), entry.substring(separator + 1));
        }
        if (!parseBoolean(values.get("valid"), false)) return unavailable();

        int planCode = parseInt(values.get("planPhase"), -1);
        int graphCode = parseInt(values.get("graphNodePhase"), -1);
        PlanPhase planPhase = PlanPhase.fromNativeCode(planCode);
        GraphNodePhase graphNodePhase = graphCode >= 0
                ? GraphNodePhase.fromCode(graphCode) : null;
        if (planPhase == null || graphNodePhase == null) return unavailable();

        return new DspLifecycleSnapshot(
                true,
                planPhase,
                graphNodePhase,
                parseInt(values.get("buildStage"), -1),
                parseInt(values.get("executionCount"), -1),
                parseInt(values.get("postFreezeExecutionCount"), -1),
                parseInt(values.get("pointersStableCount"), -1),
                parseBoolean(values.get("compilationDone"), false),
                parseInt(values.get("segmentCount"), -1),
                parseInt(values.get("buildingSegments"), -1),
                parseInt(values.get("sealedSegments"), -1),
                parseInt(values.get("failedSegments"), -1));
    }

    private static int parseInt(String value, int defaultValue) {
        if (value == null) return defaultValue;
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static boolean parseBoolean(String value, boolean defaultValue) {
        return value == null ? defaultValue : Boolean.parseBoolean(value);
    }

    public boolean isValid() { return valid; }
    public PlanPhase getPlanPhase() { return planPhase; }
    public GraphNodePhase getGraphNodePhase() { return graphNodePhase; }
    public int getBuildStage() { return buildStage; }
    public int getExecutionCount() { return executionCount; }
    public int getPostFreezeExecutionCount() { return postFreezeExecutionCount; }
    public int getPointersStableCount() { return pointersStableCount; }
    public boolean isCompilationDone() { return compilationDone; }
    public int getSegmentCount() { return segmentCount; }
    public int getBuildingSegmentCount() { return buildingSegmentCount; }
    public int getSealedSegmentCount() { return sealedSegmentCount; }
    public int getFailedSegmentCount() { return failedSegmentCount; }

    /** Shapes are stable in both SHAPES_FROZEN and REPLAYING states. */
    public boolean isShapesFrozenOrReplaying() {
        return valid && planPhase != null && planPhase.isAtLeast(PlanPhase.SHAPES_FROZEN);
    }

    /** Return a read-only diagnostic map for callers that need generic fields. */
    public Map<String, Object> asMap() {
        Map<String, Object> values = new HashMap<>();
        values.put("valid", valid);
        values.put("planPhase", planPhase);
        values.put("graphNodePhase", graphNodePhase);
        values.put("buildStage", buildStage);
        values.put("executionCount", executionCount);
        values.put("postFreezeExecutionCount", postFreezeExecutionCount);
        values.put("pointersStableCount", pointersStableCount);
        values.put("compilationDone", compilationDone);
        values.put("segmentCount", segmentCount);
        values.put("buildingSegments", buildingSegmentCount);
        values.put("sealedSegments", sealedSegmentCount);
        values.put("failedSegments", failedSegmentCount);
        return Collections.unmodifiableMap(values);
    }

    @Override
    public String toString() {
        return "DspLifecycleSnapshot{" +
                "valid=" + valid +
                ", planPhase=" + planPhase +
                ", graphNodePhase=" + graphNodePhase +
                ", buildStage=" + buildStage +
                ", executionCount=" + executionCount +
                ", postFreezeExecutionCount=" + postFreezeExecutionCount +
                ", pointersStableCount=" + pointersStableCount +
                ", compilationDone=" + compilationDone +
                ", segments=" + segmentCount +
                ", building=" + buildingSegmentCount +
                ", sealed=" + sealedSegmentCount +
                ", failed=" + failedSegmentCount +
                '}';
    }
}
