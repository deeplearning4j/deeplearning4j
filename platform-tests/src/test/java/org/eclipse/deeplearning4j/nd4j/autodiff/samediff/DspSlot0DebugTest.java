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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Minimal standalone debug test for the slot-0 null invariant bug in
 * longViewChain / EMULATED_REPLAY mode.
 *
 * Reproduces: DspSlotLifecycleAuditTest#testSlotNonNullInvariantAcrossFreeze
 * with fixture=longViewChain, mode=EMULATED_REPLAY.
 */
@Slf4j
public class DspSlot0DebugTest {

    private static final long SEED = 0xD5CA11B10L;
    private static final AtomicLong WEIGHT_IDX = new AtomicLong(0L);

    private SameDiff sd;

    @BeforeEach
    public void setUp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
        Nd4j.getRandom().setSeed(SEED);
    }

    @AfterEach
    public void tearDown() {
        if (sd != null) {
            try { sd.close(); } catch (Throwable t) { log.warn("sd.close()", t); }
            sd = null;
        }
        Nd4j.getExecutioner().commit();
    }

    @Test
    public void testSlot0NullDebug() throws Exception {
        // Build longViewChain graph
        WEIGHT_IDX.set(0L);
        Nd4j.getRandom().setSeed(SEED);
        sd = SameDiff.create();
        SDVariable x  = sd.placeHolder("x", DataType.FLOAT, 1, 2, 3, 4, 5);
        SDVariable w  = sd.var("w", randFloat(0.05, 5, 5));
        SDVariable p0 = sd.permute("p0", x, 0, 4, 1, 2, 3);   // [1,5,2,3,4]
        SDVariable r0 = sd.reshape("r0", p0, 1, 5, 24);        // [1,5,24]
        SDVariable p1 = sd.permute("p1", r0, 0, 2, 1);         // [1,24,5]
        SDVariable r1 = sd.reshape("r1", p1, 120, 1);          // [120,1]
        SDVariable r2 = sd.reshape("r2", r1, 24, 5);           // [24,5]
        SDVariable r3 = sd.reshape("r3", r2, 4, 30);           // [4,30]
        SDVariable r4 = sd.reshape("r4", r3, 120, 1);          // [120,1]
        SDVariable r5 = sd.reshape("r5", r4, 24, 5);           // [24,5]
        sd.mmul("out", r5, w);                                  // [24,5]
        sd.setOutputs("out");

        // Configure EMULATED_REPLAY
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.EMULATED_REPLAY);

        // 3 warmup executions
        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> inputs = makeInputs();
            System.out.println("[exec " + i + "] BEFORE output()");
            sd.output(inputs, "out");
            DspHandle h = sd.dsp();
            INDArray slot0 = safeGetSlot(h, 0);
            Integer phase = h.isCompiled() ? h.planPhase() : null;
            System.out.println("[exec " + i + "] slot0=" + (slot0 == null ? "NULL" : "non-null")
                    + "  phase=" + phase);
            closeAll(inputs);
            System.out.println("[exec " + i + "] slot0 after closeAll(inputs)="
                    + (slot0 == null ? "N/A" : (slot0.wasClosed() ? "CLOSED" : "open")));
        }

        // Snapshot: which slots were non-null after warmup
        DspHandle handle = sd.dsp();
        if (!handle.isCompiled()) {
            System.out.println("Handle not compiled after warmup — skipping slot check.");
            return;
        }
        int totalSlots = handle.totalSlots();
        boolean[] wasNonNull = new boolean[totalSlots];
        for (int s = 0; s < totalSlots; s++) {
            wasNonNull[s] = (safeGetSlot(handle, s) != null);
        }
        System.out.println("[snapshot] slot0 wasNonNull=" + wasNonNull[0]
                + "  totalSlots=" + totalSlots
                + "  phase=" + handle.planPhase());

        // 1 post-warmup execution with detailed slot-0 tracking
        Map<String, INDArray> inputs = makeInputs();
        INDArray slot0Before = safeGetSlot(handle, 0);
        System.out.println("[exec 3] slot0 BEFORE output()="
                + (slot0Before == null ? "NULL" : "non-null")
                + "  phase=" + handle.planPhase());

        sd.output(inputs, "out");

        INDArray slot0After = safeGetSlot(handle, 0);
        System.out.println("[exec 3] slot0 AFTER output()="
                + (slot0After == null ? "NULL" : "non-null")
                + "  phase=" + handle.planPhase());

        closeAll(inputs);
        System.out.println("[exec 3] slot0 after closeAll(inputs)="
                + (slot0After == null ? "N/A" : (slot0After.wasClosed() ? "CLOSED" : "open")));

        // Assert slot 0 is non-null (mirrors the invariant the failing test checks)
        assertNotNull(slot0After,
                "slot 0 must remain non-null after post-warmup execution in EMULATED_REPLAY"
                + " — phase=" + handle.planPhase());
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static Map<String, INDArray> makeInputs() {
        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put("x", Nd4j.linspace(DataType.FLOAT, -0.2, 0.005, 1 * 2 * 3 * 4 * 5)
                .reshape(1, 2, 3, 4, 5));
        return m;
    }

    private static INDArray safeGetSlot(DspHandle h, int slot) {
        try {
            return h.getSlotOutput(slot);
        } catch (Throwable t) {
            return null;
        }
    }

    private static void closeAll(Map<String, INDArray> arrays) {
        if (arrays == null) return;
        for (INDArray arr : arrays.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
    }

    private static INDArray randFloat(double scale, long... shape) {
        long total = Arrays.stream(shape).reduce(1L, (a, b) -> a * b);
        long callIdx = WEIGHT_IDX.getAndIncrement();
        long state = (callIdx + 1L) * 0x9E3779B97F4A7C15L;
        state ^= (state >>> 30); state *= 0xBF58476D1CE4E5B9L;
        state ^= (state >>> 27); state *= 0x94D049BB133111EBL;
        state ^= (state >>> 31);
        if (state == 0L) state = 0xDEADBEEFCAFEBABEL;
        float[] data = new float[(int) total];
        for (int i = 0; i < total; i++) {
            state ^= state << 13; state ^= state >>> 7; state ^= state << 17;
            double u = ((state >>> 11) & 0x1FFFFFFFFFFFFFL) / (double) (1L << 53);
            data[i] = (float) ((u * 2.0 - 1.0) * scale);
        }
        return Nd4j.create(data, shape);
    }
}
