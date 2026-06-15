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

package org.deeplearning4j.ui.playwright;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsInitReport;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsReport;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsReport.SegmentSnapshot;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsReport.SlotSnapshot;
import org.deeplearning4j.ui.model.stats.dsp.DspDiagnosticsReport.MemoryEventSnapshot;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.model.storage.impl.JavaStorageMetaData;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

/**
 * Standalone server that populates all UI modules with realistic data
 * for Playwright browser tests.
 *
 * - TrainModule: real Iris MLP training (5 epochs) via StatsListener
 * - DspDiagModule: mock DSP plan data
 * - ModelViewer / ModelManager / EntityResolution: pages served (no upload data)
 *
 * Run via: bash e2e/start-test-server.sh
 */
public class DspUiTestServer {

    private static final String DSP_SESSION_ID = "test-session-001";
    private static final String DSP_WORKER_ID = "playwright-worker";

    public static void main(String[] args) throws Exception {
        StatsStorage ss = new InMemoryStatsStorage();
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(ss);

        // ── Training data (real model) ──────────────────────────────────
        populateTrainingData(ss);

        // ── DSP diagnostics (mock) ──────────────────────────────────────
        populateDspData(ss);

        System.out.println("UI test server running at http://localhost:9000");
        System.out.println("  Training:  http://localhost:9000/train/overview");
        System.out.println("  DSP:       http://localhost:9000/dsp");
        System.out.println("  ModelView: http://localhost:9000/modelview");
        System.out.println("  Models:    http://localhost:9000/models");
        System.out.println("  Entities:  http://localhost:9000/entities");
        System.out.println("  SameDiff:  http://localhost:9000/samediff");
        System.out.println("Press Ctrl+C to stop.");

        Thread.currentThread().join();
    }

    private static void populateTrainingData(StatsStorage ss) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.01))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(10)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(8)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder().nIn(8).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new StatsListener(ss, 1));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        for (int i = 0; i < 5; i++) {
            net.fit(iter);
            iter.reset();
        }
    }

    private static void populateDspData(StatsStorage ss) throws Exception {
        JavaStorageMetaData meta = new JavaStorageMetaData(
                System.currentTimeMillis(), DSP_SESSION_ID,
                DspDiagnosticsReport.TYPE_ID, DSP_WORKER_ID,
                DspDiagnosticsInitReport.class, DspDiagnosticsReport.class
        );
        ss.putStorageMetaData(meta);

        DspDiagnosticsInitReport init = new DspDiagnosticsInitReport();
        init.setSessionID(DSP_SESSION_ID);
        init.setWorkerID(DSP_WORKER_ID);
        init.setTimeStamp(System.currentTimeMillis());
        init.setVariableCount(42);
        init.setOpCount(18);
        init.setParamCount(1_200_000);
        init.setParamNames(List.of("weight_0", "bias_0", "weight_1", "bias_1"));
        init.setBackend("Nd4jCuda");
        init.setDataType("FLOAT");
        init.setHostname("playwright-host");
        init.setConfiguredExecutionMode("AUTO");
        init.setRequestedOutputs(List.of("output", "logits"));
        ss.putStaticInfo(init);

        for (int iter = 1; iter <= 5; iter++) {
            DspDiagnosticsReport report = buildDspReport(iter);
            ss.putUpdate(report);
            Thread.sleep(50);
        }
    }

    private static DspDiagnosticsReport buildDspReport(int iteration) {
        DspDiagnosticsReport r = new DspDiagnosticsReport();
        r.setSessionID(DSP_SESSION_ID);
        r.setWorkerID(DSP_WORKER_ID);
        r.setTimeStamp(System.currentTimeMillis());
        r.setIteration(iteration);
        r.setIterationTimeMs(45 + (long) (Math.random() * 20));

        if (iteration <= 2) {
            r.setPlanPhase("SLOT_BY_SLOT");
            r.setGraphNodePhase("BUILDING");
        } else if (iteration <= 3) {
            r.setPlanPhase("SHAPES_FROZEN");
            r.setGraphNodePhase("SEALED");
        } else {
            r.setPlanPhase("REPLAYING");
            r.setGraphNodePhase("SEALED");
        }
        r.setExecutionMode("AUTO");
        r.setNumSlots(18);
        r.setNumSegments(4);
        r.setUnfrozenOpCount(iteration <= 2 ? 6 : 0);

        List<SegmentSnapshot> segments = new ArrayList<>();
        segments.add(new SegmentSnapshot(0, 4, true, 0, 5,
                List.of("matmul", "add", "relu", "dropout", "reshape"),
                iteration <= 2 ? "EMPTY" : iteration == 3 ? "CAPTURING" : "REPLAYING",
                iteration > 3 ? 2 : 0, iteration, iteration > 3 ? "REPLAYING" : "WARMUP",
                "cuBLAS", false));
        segments.add(new SegmentSnapshot(5, 9, true, 0, 5,
                List.of("matmul", "add", "gelu", "layernorm", "residual"),
                iteration <= 2 ? "EMPTY" : iteration == 3 ? "CAPTURED" : "REPLAYING",
                iteration > 3 ? 2 : 0, iteration, iteration > 3 ? "REPLAYING" : "COMPILING",
                "Triton", false));
        segments.add(new SegmentSnapshot(10, 14, false, 0, 5,
                List.of("softmax", "matmul", "transpose", "gather", "concat"),
                "EMPTY", 0, iteration, "SLOT_BY_SLOT", null, false));
        segments.add(new SegmentSnapshot(15, 17, true, 1, 3,
                List.of("matmul", "add", "softmax"),
                iteration >= 4 ? "READY" : "EMPTY",
                0, iteration, iteration >= 4 ? "COMPILED" : "WARMUP",
                "cuBLAS", iteration == 3));
        r.setSegments(segments);

        Map<String, Integer> ops = new LinkedHashMap<>();
        ops.put("matmul", 4); ops.put("add", 3); ops.put("softmax", 2);
        ops.put("relu", 1); ops.put("gelu", 1); ops.put("layernorm", 1);
        ops.put("dropout", 1); ops.put("reshape", 1); ops.put("transpose", 1);
        ops.put("gather", 1); ops.put("concat", 1); ops.put("residual", 1);
        r.setOpHistogram(ops);

        List<SlotSnapshot> risky = new ArrayList<>();
        risky.add(new SlotSnapshot(7, "gather", 5, "UNFROZEN",
                "DATA_DEPENDENT | VALUE_DEPENDENT_SHAPE"));
        risky.add(new SlotSnapshot(12, "reshape", 3, "UNFROZEN",
                "VIEW_CAPABLE | DATA_DEPENDENT"));
        r.setRiskyOps(risky);

        List<MemoryEventSnapshot> memory = new ArrayList<>();
        for (int step = 0; step < 18; step++) {
            memory.add(new MemoryEventSnapshot(step, step, "ALLOCATE"));
            if (step > 3) memory.add(new MemoryEventSnapshot(step, step - 3, "RELEASE"));
        }
        r.setMemoryTimeline(memory);

        r.setDotGraph("digraph plan {\n  rankdir=LR;\n" +
                "  node [shape=box, style=filled, fillcolor=\"#3182ce\", fontcolor=white];\n" +
                "  subgraph cluster_seg0 { label=\"Seg 0 (cuBLAS)\";\n" +
                "    s0 [label=\"matmul\"]; s1 [label=\"add\"]; s2 [label=\"relu\"];\n" +
                "  }\n" +
                "  subgraph cluster_seg1 { label=\"Seg 1 (Triton)\";\n" +
                "    s5 [label=\"matmul\"]; s6 [label=\"gelu\"]; s7 [label=\"layernorm\"];\n" +
                "  }\n" +
                "  s0->s1->s2->s5->s6->s7;\n}");

        Map<Integer, Integer> devices = new LinkedHashMap<>();
        devices.put(0, 15); devices.put(1, 3);
        r.setDeviceSlotCounts(devices);
        return r;
    }
}
