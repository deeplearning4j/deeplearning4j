/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.ui.model.stats;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.core.storage.StatsStorageRouter;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsInitReport;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsReport;
import org.deeplearning4j.ui.model.stats.dsp.SegmentSnapshot;
import org.deeplearning4j.ui.model.stats.dsp.SlotSnapshot;
import org.deeplearning4j.ui.model.stats.dsp.MemoryEventSnapshot;
import org.deeplearning4j.ui.model.storage.impl.JavaStorageMetaData;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.*;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.autodiff.samediff.internal.Variable;

import java.lang.management.ManagementFactory;
import java.util.*;
import java.util.stream.Collectors;

/**
 * SameDiff listener that captures DSP (Dynamic Shape Plan) diagnostics
 * and writes them to a {@link StatsStorageRouter} for visualization
 * in the DSP Diagnostics UI module.
 *
 * Captures: plan phase, segment replay state, op histogram,
 * memory timeline, DOT graph, risky ops, and device placement.
 *
 * Works for both training and inference operations.
 *
 * Usage:
 * <pre>
 *     InMemoryStatsStorage storage = new InMemoryStatsStorage();
 *     UIServer uiServer = UIServer.getInstance();
 *     uiServer.attach(storage);
 *
 *     DspStatsListener listener = new DspStatsListener(storage);
 *     sd.setListeners(listener);
 *     sd.output(placeholders, "output");
 * </pre>
 */
@Slf4j
public class DspStatsListener extends BaseListener {

    private final StatsStorageRouter router;
    private final int reportFrequency;
    private final String sessionID;
    private final String workerID;

    private boolean initialized = false;
    private int iterationCount = 0;
    private long lastIterationStartMs = 0;

    public DspStatsListener(StatsStorageRouter router) {
        this(router, 1);
    }

    public DspStatsListener(StatsStorageRouter router, int reportFrequency) {
        this.router = router;
        this.reportFrequency = Math.max(1, reportFrequency);
        this.sessionID = UUID.randomUUID().toString();
        this.workerID = getJvmUid() + "_dsp_" + Thread.currentThread().getId();
    }

    @Override
    public boolean isActive(Operation operation) {
        return true; // Active for both training and inference
    }

    @Override
    public void operationStart(SameDiff sd, Operation op) {
        lastIterationStartMs = System.currentTimeMillis();
        if (!initialized) {
            doInit(sd);
        }
    }

    @Override
    public void iterationDone(SameDiff sd, At at, org.nd4j.linalg.dataset.api.MultiDataSet dataSet, Loss loss) {
        iterationCount++;
        if (iterationCount % reportFrequency != 0) return;
        captureAndReport(sd);
    }

    @Override
    public void opExecution(SameDiff sd, At at, org.nd4j.linalg.dataset.api.MultiDataSet batch,
                            org.nd4j.autodiff.samediff.internal.SameDiffOp op,
                            org.nd4j.linalg.api.ops.OpContext opContext, INDArray[] outputs) {
        // For inference (no iterationDone callback), capture on each op execution
        // but only report at frequency intervals
        if (!initialized) {
            doInit(sd);
        }
    }

    /**
     * Manually trigger a DSP diagnostics capture. Call this after inference
     * operations that don't fire iterationDone.
     */
    public void captureInference(SameDiff sd) {
        iterationCount++;
        if (iterationCount % reportFrequency != 0) return;
        captureAndReport(sd);
    }

    private void captureAndReport(SameDiff sd) {
        long iterTimeMs = System.currentTimeMillis() - lastIterationStartMs;
        lastIterationStartMs = System.currentTimeMillis();

        try {
            InferenceSession session = sd.getOrCreateSession();
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            if (executor == null) return;

            DynamicShapePlan plan = executor.getCurrentPlan();
            if (plan == null) return;

            org.bytedeco.javacpp.Pointer nativeHandle = executor.getNativePlanHandle();

            DspDiagnosticsReport report = new DspDiagnosticsReport();
            report.setSessionID(sessionID);
            report.setWorkerID(workerID);
            report.setTimeStamp(System.currentTimeMillis());
            report.setIteration(iterationCount);
            report.setIterationTimeMs(iterTimeMs);

            // Plan-level state via DspDebugger
            DspDebugger.PlanReport planReport = DspDebugger.attach(sd).analyzePlan();
            report.setNumSlots(planReport.numSlots);
            report.setNumSegments(planReport.numSegments);
            report.setPlanPhase(planReport.planPhase != null ? planReport.planPhase.name() : "UNKNOWN");
            report.setGraphNodePhase(planReport.graphNodePhase != null ? planReport.graphNodePhase.name() : "UNKNOWN");

            // Execution mode
            GraphExecutionMode mode = executor.getConfiguredGraphExecutionMode();
            report.setExecutionMode(mode != null ? mode.name() : "UNKNOWN");

            // Segments with replay state
            List<PlanIntrospection.SegmentInfo> segInfos;
            if (nativeHandle != null && !nativeHandle.isNull()) {
                segInfos = PlanIntrospection.getSegmentsWithReplayState(plan, nativeHandle);
            } else {
                segInfos = PlanIntrospection.getSegments(plan);
            }
            List<SegmentSnapshot> segSnaps = new ArrayList<>();
            for (PlanIntrospection.SegmentInfo si : segInfos) {
                segSnaps.add(new SegmentSnapshot(
                        si.getStartSlot(), si.getEndSlot(), si.isCapturable(),
                        si.getDeviceId(), si.getSlotCount(), si.getOpNames(),
                        si.getReplayStateName(), si.getReplayCount(),
                        si.getExecutionCount(),
                        si.getExecutionPhase() != null ? si.getExecutionPhase().name() : "UNKNOWN",
                        si.getReplayBackendName(), si.isCompilationFailed()
                ));
            }
            report.setSegments(segSnaps);

            // Op histogram
            report.setOpHistogram(planReport.getOpHistogram());

            // Risky ops
            List<SlotSnapshot> riskySnaps = new ArrayList<>();
            for (DspDebugger.SlotInfo slot : planReport.getRiskyOps()) {
                riskySnaps.add(new SlotSnapshot(
                        slot.index, slot.opName, slot.flags,
                        slot.state != null ? slot.state.name() : "UNKNOWN", slot.formatFlags()
                ));
            }
            report.setRiskyOps(riskySnaps);
            report.setUnfrozenOpCount(planReport.getUnfrozenOps().size());

            // Memory timeline
            List<PlanIntrospection.MemoryEvent> memEvents = PlanIntrospection.getMemoryTimeline(plan);
            List<MemoryEventSnapshot> memSnaps = new ArrayList<>();
            for (PlanIntrospection.MemoryEvent me : memEvents) {
                memSnaps.add(new MemoryEventSnapshot(me.getStep(), me.getSlotIndex(), me.getEventType().name()));
            }
            report.setMemoryTimeline(memSnaps);

            // DOT graph
            report.setDotGraph(PlanIntrospection.toDot(plan));

            // Device placement
            Map<Integer, List<Integer>> devicePlacement = PlanIntrospection.getDevicePlacement(plan);
            Map<Integer, Integer> deviceCounts = new LinkedHashMap<>();
            for (Map.Entry<Integer, List<Integer>> e : devicePlacement.entrySet()) {
                deviceCounts.put(e.getKey(), e.getValue().size());
            }
            report.setDeviceSlotCounts(deviceCounts);

            router.putUpdate(report);

        } catch (Exception e) {
            log.debug("Failed to capture DSP diagnostics: {}", e.getMessage());
        }
    }

    private void doInit(SameDiff sd) {
        if (initialized) return;
        initialized = true;

        long initTime = System.currentTimeMillis();

        DspDiagnosticsInitReport initReport = new DspDiagnosticsInitReport();
        initReport.setSessionID(sessionID);
        initReport.setWorkerID(workerID);
        initReport.setTimeStamp(initTime);
        initReport.setVariableCount(sd.variables().size());
        initReport.setOpCount(sd.getOps().size());

        List<String> paramNames = new ArrayList<>();
        long totalParams = 0;
        for (SDVariable var : sd.variables()) {
            if (var.getVariableType() == VariableType.VARIABLE) {
                paramNames.add(var.name());
                try {
                    long[] shape = var.getShape();
                    if (shape != null) {
                        long count = 1;
                        for (long s : shape) count *= s;
                        totalParams += count;
                    }
                } catch (Exception e) { /* shape not known yet */ }
            }
        }
        initReport.setParamNames(paramNames);
        initReport.setParamCount(totalParams);

        try {
            initReport.setBackend(org.nd4j.linalg.factory.Nd4j.getBackend().getClass().getSimpleName());
            initReport.setDataType(org.nd4j.linalg.factory.Nd4j.dataType().name());
        } catch (Exception e) { /* ignore */ }

        try {
            initReport.setHostname(java.net.InetAddress.getLocalHost().getHostName());
        } catch (Exception e) {
            initReport.setHostname("unknown");
        }

        JavaStorageMetaData meta = new JavaStorageMetaData(
                initTime, sessionID, DspDiagnosticsReport.TYPE_ID, workerID,
                DspDiagnosticsInitReport.class, DspDiagnosticsReport.class
        );

        router.putStorageMetaData(meta);
        router.putStaticInfo(initReport);
    }

    private static String getJvmUid() {
        String name = ManagementFactory.getRuntimeMXBean().getName();
        int idx = name.indexOf('@');
        return idx > 0 ? name.substring(0, idx) : name;
    }
}
