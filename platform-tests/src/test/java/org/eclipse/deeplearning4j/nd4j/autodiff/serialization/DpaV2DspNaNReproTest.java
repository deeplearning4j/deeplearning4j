/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.serialization;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone isolation for the BGE fused-attention NaN: dot_product_attention_v2
 * under DSP returns NaN on a random layer/iteration (fp32 AND fp16 graphs), with
 * all slot inputs finite. Mirrors the AttentionFusionOptimizations wiring:
 * BSHD reshape views -> dpa_v2(q, v, k, emptyQMask, additive vMask [1,1,1,512]).
 * Fixed inputs across iterations: outputs must be finite AND bit-stable.
 * Pool-poisoning between iterations makes uninitialized-buffer reads deterministic.
 */
@Slf4j
public class DpaV2DspNaNReproTest extends BaseND4JTest {

    private static final int LAYERS = 4;
    private static final int ITERATIONS = 6;

    @Override
    public long getTimeoutMilliseconds() {
        return 10 * 60 * 1000L;
    }

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Override
    public DataType getDefaultFPDataType() {
        return DataType.FLOAT;
    }

    @Test
    public void testDpaV2RepeatedDspExecutionStaysFinite() {
        Map<String, String> prev = new LinkedHashMap<>();
        prev.put(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED,
                System.getProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED));
        prev.put(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE,
                System.getProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE));
        boolean prevDsp = InferenceSession.isDynamicShapePlanEnabled();
        // -Ddpa.repro.nanPanic=true: throw at the first op producing NaN (eager pinpointing)
        if (Boolean.getBoolean("dpa.repro.nanPanic")) {
            Nd4j.getExecutioner().setProfilingConfig(
                    org.nd4j.linalg.profiler.ProfilerConfig.builder()
                            .checkForNAN(true).checkForINF(false).build());
        }
        // Repro knobs: -Ddpa.repro.dsp=false, -Ddpa.repro.mode=SLOT_BY_SLOT, -Ddpa.repro.tritonSkip=true
        boolean dspEnabled = !"false".equalsIgnoreCase(System.getProperty("dpa.repro.dsp", "true"));
        String mode = System.getProperty("dpa.repro.mode", "AUTO");
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, Boolean.toString(dspEnabled));
        System.setProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE, mode);
        if (Boolean.getBoolean("dpa.repro.tritonSkip")) {
            System.setProperty(ND4JSystemProperties.TRITON_SKIP_KERNELS, "true");
        }
        InferenceSession.setDynamicShapePlanEnabled(dspEnabled);

        // Structure knobs: -Ddpa.repro.layers=N, -Ddpa.repro.direct4d=true (placeholders
        // already BSHD, no in-graph reshape views), -Ddpa.repro.nomask=true (flash-fast path),
        // -Ddpa.repro.fetchInternal=true (also scan the scores/logits op outputs)
        int layers = Integer.getInteger("dpa.repro.layers", LAYERS);
        boolean direct4d = Boolean.getBoolean("dpa.repro.direct4d");
        boolean noMask = Boolean.getBoolean("dpa.repro.nomask");
        boolean fetchInternal = Boolean.getBoolean("dpa.repro.fetchInternal");

        SameDiff sd = SameDiff.create();
        INDArray xArr = null;
        INDArray maskArr = null;
        INDArray reference = null;
        List<String> internalOutNames = new ArrayList<>();
        try {
            SDVariable x = direct4d
                    ? sd.placeHolder("x", DataType.FLOAT, 1, 512, 12, 64)
                    : sd.placeHolder("x", DataType.FLOAT, 1, 512, 768);
            SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, 1, 1, 1, 512);
            SDVariable cur = x;
            for (int l = 0; l < layers; l++) {
                SDVariable q = direct4d ? cur : cur.reshape(1, 512, 12, 64);
                SDVariable k = direct4d ? cur : cur.reshape(1, 512, 12, 64);
                SDVariable v = direct4d ? cur : cur.reshape(1, 512, 12, 64);
                SDVariable emptyQ = sd.constant("attn_masked_empty_qmask_l" + l,
                        Nd4j.empty(DataType.FLOAT));
                DotProductAttentionV2 op = new DotProductAttentionV2(sd, q, v, k, emptyQ,
                        noMask ? null : mask,
                        null, null, null, null, 0.125, 0.0, false, false);
                SDVariable[] outs = op.outputVariables();
                SDVariable attnOut = outs[0];
                if (fetchInternal) {
                    internalOutNames.add(outs[0].name());
                    internalOutNames.add(outs[1].name());
                    internalOutNames.add(outs[2].name());
                }
                // residual keeps magnitudes BGE-like across layers
                cur = direct4d
                        ? attnOut.add("layer_out_" + l, cur)
                        : attnOut.reshape(1, 512, 768).add("layer_out_" + l, cur);
            }

            Nd4j.getRandom().setSeed(12345);
            xArr = direct4d
                    ? Nd4j.rand(DataType.FLOAT, 1, 512, 12, 64).muli(4.0).subi(2.0)
                    : Nd4j.rand(DataType.FLOAT, 1, 512, 768).muli(4.0).subi(2.0);
            maskArr = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 512);
            List<String> outNameList = new ArrayList<>();
            for (int l = 0; l < layers; l++) {
                outNameList.add("layer_out_" + l);
            }
            outNameList.addAll(internalOutNames);
            String[] outNames = outNameList.toArray(new String[0]);

            Map<String, INDArray> references = new LinkedHashMap<>();
            boolean anyMismatch = false;
            StringBuilder mismatchReport = new StringBuilder();
            boolean nanPanic = Boolean.getBoolean("dpa.repro.nanPanic");
            for (int i = 0; i < ITERATIONS; i++) {
                // The poison fill itself trips checkForNAN (its input is the recycled
                // pool block still holding prior poison) — suspend profiling around it.
                if (nanPanic) {
                    Nd4j.getExecutioner().setProfilingConfig(
                            org.nd4j.linalg.profiler.ProfilerConfig.builder()
                                    .checkForNAN(false).checkForINF(false).build());
                }
                if (!Boolean.getBoolean("dpa.repro.skipPoison")) {
                    dirtyPoolWithNaNs();
                }
                if (nanPanic) {
                    Nd4j.getExecutioner().setProfilingConfig(
                            org.nd4j.linalg.profiler.ProfilerConfig.builder()
                                    .checkForNAN(true).checkForINF(false).build());
                }
                Map<String, INDArray> out = noMask
                        ? sd.output(Map.of("x", xArr), outNames)
                        : sd.output(Map.of("x", xArr, "mask", maskArr), outNames);
                for (String name : outNames) {
                    INDArray result = out.get(name);
                    assertNotNull(result, "iteration " + i + ": missing " + name);
                    boolean nan = result.isNaN().any();
                    boolean inf = result.isInfinite().any();
                    INDArray ref = references.get(name);
                    double maxDiff = ref == null ? 0.0
                            : ref.sub(result).amaxNumber().doubleValue();
                    log.info("iteration {} out {}: hasNaN={} hasInf={} maxDiffVsIter0={}",
                            i, name, nan, inf, maxDiff);
                    if (nan || inf || maxDiff != 0.0) {
                        anyMismatch = true;
                        mismatchReport.append("iteration ").append(i).append(" out ").append(name)
                                .append(": hasNaN=").append(nan).append(" hasInf=").append(inf)
                                .append(" maxDiffVsIter0=").append(maxDiff).append('\n');
                    }
                    if (ref == null) {
                        references.put(name, result.dup());
                    }
                }
            }
            for (INDArray ref : references.values()) {
                if (ref != null) ref.close();
            }
            assertFalse(anyMismatch,
                    "dpa_v2 DSP results non-finite or unstable across iterations:\n" + mismatchReport);
        } finally {
            sd.close();
            if (xArr != null) xArr.close();
            if (maskArr != null) maskArr.close();
            if (reference != null) reference.close();
            for (Map.Entry<String, String> e : prev.entrySet()) {
                if (e.getValue() == null) System.clearProperty(e.getKey());
                else System.setProperty(e.getKey(), e.getValue());
            }
            InferenceSession.setDynamicShapePlanEnabled(prevDsp);
        }
    }

    /**
     * Fill the device pool with NaN patterns sized like the attention intermediates
     * ([1,12,512,512] scores + BSHD tensors) so any kernel reading memory it never
     * wrote picks up NaN deterministically instead of leftover lucky values.
     */
    private static void dirtyPoolWithNaNs() {
        List<INDArray> poison = new ArrayList<>();
        try {
            for (int i = 0; i < 2; i++) {
                poison.add(Nd4j.valueArrayOf(new long[]{1, 12, 512, 512}, Double.NaN, DataType.FLOAT));
                poison.add(Nd4j.valueArrayOf(new long[]{1, 512, 12, 64}, Double.NaN, DataType.FLOAT));
                poison.add(Nd4j.valueArrayOf(new long[]{1, 512, 768}, Double.NaN, DataType.FLOAT));
            }
        } finally {
            for (INDArray p : poison) {
                p.close();
            }
        }
    }
}
