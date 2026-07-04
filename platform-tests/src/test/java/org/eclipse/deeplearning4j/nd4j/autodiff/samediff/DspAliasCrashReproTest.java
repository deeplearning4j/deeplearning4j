/*
 *  ******************************************************************************
 *  *
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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * MICROSCOPIC repro for the DspBufferAliasAccuracyTest JVM crash (host-side
 * SIGSEGV inside cudaGraphLaunch at ~plan #18 / 2nd launch of a merged-capture
 * exec whose pointer has NO destroy event). One fixture (opToReshapeView), ONE
 * mode (system property {@code repro.mode}, default NVRTC_JIT), the exact
 * varying-input + fresh-reference loop of the parent test — small enough for
 * compute-sanitizer.
 */
@Slf4j
public class DspAliasCrashReproTest {

    private static final int REPLAYS = 5;

    /**
     * The parent crash needs ACCUMULATION: every mode passes in isolation, the
     * crash appears ~param #4 of the sequential matrix. Walk the same mode order
     * on one fixture in one JVM.
     */
    @Test
    public void reproModeSequence() {
        GraphExecutionMode[] order = {
                GraphExecutionMode.AUTO, GraphExecutionMode.SLOT_BY_SLOT,
                GraphExecutionMode.CUDA_GRAPHS, GraphExecutionMode.NVRTC_JIT,
                GraphExecutionMode.PTX_JIT, GraphExecutionMode.TRITON,
                GraphExecutionMode.EMULATED_REPLAY};
        String[] fixtures = {"opToReshapeView", "concatToExpandDims", "tripleViewChain"};
        for (String fx : fixtures) {
            for (GraphExecutionMode m : order) {
                log.info("SEQUENCE entering fixture={} mode={}", fx, m);
                runOneConfig(fx, m);
                log.info("SEQUENCE fixture={} mode={} done", fx, m);
            }
        }
        log.info("SEQUENCE COMPLETED CLEAN");
    }

    @Test
    public void reproVaryingInput() {
        GraphExecutionMode mode = GraphExecutionMode.valueOf(
                System.getProperty("repro.mode", "NVRTC_JIT"));
        log.info("REPRO mode={}", mode);
        runOneConfig(System.getProperty("repro.fixture", "opToReshapeView"), mode);
        log.info("REPRO mode={} COMPLETED CLEAN", mode);
    }

    private static SameDiff buildFixture(String name) {
        switch (name) {
            case "concatToExpandDims": return buildConcatToExpandDims();
            case "tripleViewChain": return buildTripleViewChain();
            default: return buildOpToReshapeView();
        }
    }

    private static Map<String, INDArray> inputsFor(String name) {
        if ("concatToExpandDims".equals(name)) {
            Map<String, INDArray> m = new LinkedHashMap<>();
            m.put("a", Nd4j.linspace(DataType.FLOAT, -0.3, 0.005, 8 * 16).reshape(8, 16));
            m.put("b", Nd4j.linspace(DataType.FLOAT, 0.2, 0.004, 4 * 16).reshape(4, 16));
            return m;
        }
        return inputsSmall4x16();
    }

    private static SameDiff buildConcatToExpandDims() {
        SameDiff g = SameDiff.create();
        SDVariable a = g.placeHolder("a", DataType.FLOAT, 8, 16);
        SDVariable b = g.placeHolder("b", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w", Nd4j.linspace(DataType.FLOAT, -0.04, 0.0006, 16 * 8).reshape(16, 8));
        SDVariable c = g.concat("concat", 0, a, b);          // [12,16]
        SDVariable e = g.expandDims("exp", c, 0);            // [1,12,16] view
        SDVariable s = g.squeeze("sq", e, 0);                // [12,16] view-of-view
        g.mmul("final", s, w);                               // [12,8]
        g.setOutputs("final");
        return g;
    }

    private static SameDiff buildTripleViewChain() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w", Nd4j.linspace(DataType.FLOAT, -0.02, 0.0005, 8 * 4).reshape(8, 4));
        SDVariable r1 = g.reshape("r1", x, 8, 8);
        SDVariable p = g.permute("p", r1, 1, 0);
        SDVariable r2 = g.reshape("r2", p, 8, 8);
        g.mmul("final", r2, w);
        g.setOutputs("final");
        return g;
    }

    private void runOneConfig(String fixture, GraphExecutionMode mode) {

        SameDiff sd = buildFixture(fixture);
        sd.setGraphExecutionMode(mode);

        for (int i = 0; i < REPLAYS; i++) {
            Map<String, INDArray> inputs = inputsFor(fixture);
            for (Map.Entry<String, INDArray> e : inputs.entrySet()) {
                e.getValue().addi(0.5 * (i + 1));
            }
            try {
                Map<String, INDArray> raw = sd.output(inputs, "final");
                log.info("REPRO replay#{} main out amean={}", i,
                        raw.get("final").amean().getDouble(0));

                // Fresh reference graph per replay — the churn the parent test does.
                SameDiff ref = buildFixture(fixture);
                ref.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                Map<String, INDArray> refIn = inputsFor(fixture);
                for (Map.Entry<String, INDArray> e : refIn.entrySet()) {
                    e.getValue().addi(0.5 * (i + 1));
                }
                try {
                    Map<String, INDArray> refOut = ref.output(refIn, "final");
                    log.info("REPRO replay#{} ref  out amean={}", i,
                            refOut.get("final").amean().getDouble(0));
                } finally {
                    // Parent's discipline: close reference inputs after use.
                    for (INDArray a : refIn.values()) { try { a.close(); } catch (Throwable ignored) { } }
                    try { ref.close(); } catch (Throwable ignored) { }
                }
            } finally {
                // THE parent ingredient the first repro lacked: close the MAIN plan's
                // input placeholder buffers after every execution while the plan stays
                // live and replays next iteration with fresh inputs.
                for (INDArray a : inputs.values()) { try { a.close(); } catch (Throwable ignored) { } }
            }
        }
        sd.close();
        log.info("REPRO mode={} COMPLETED CLEAN", mode);
    }

    private static SameDiff buildOpToReshapeView() {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w = g.var("w", Nd4j.linspace(DataType.FLOAT, -0.05, 0.000391, 16 * 16).reshape(16, 16));
        SDVariable w2 = g.var("w2", Nd4j.linspace(DataType.FLOAT, -0.03, 0.00188, 8 * 4).reshape(8, 4));
        SDVariable h = g.mmul("mmul1", x, w);
        SDVariable a = g.nn.relu("relu", h, 0);
        SDVariable v = g.reshape("view", a, 8, 8);
        g.mmul("final", v, w2);
        g.setOutputs("final");
        return g;
    }

    private static Map<String, INDArray> inputsSmall4x16() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.4, 0.01, 4 * 16).reshape(4, 16));
        return m;
    }
}
