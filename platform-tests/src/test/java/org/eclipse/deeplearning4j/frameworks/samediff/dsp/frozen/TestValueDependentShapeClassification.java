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
package org.eclipse.deeplearning4j.frameworks.samediff.dsp.frozen;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Regression tests for value-dependent-shape classification in
 * {@link org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler}.
 *
 * <p>Background: the Java plan compiler previously stamped
 * {@code outputShapeDependsOnInputValues=true} on every slot whose inputs
 * included an {@code INT} or {@code LONG} variable. That heuristic
 * misclassified ops like {@code gather}, {@code gather_nd}, {@code concat},
 * etc., whose output shape is derivable from input <em>shapes</em> alone,
 * and made the C++ SHAPES_FROZEN lifecycle check fire
 * {@code LIFECYCLE_ERROR: value-dependent output shape changed at slot N
 * (gather) during SHAPES_FROZEN phase} whenever the gather output grew or
 * shrank between executions (e.g. growing KV cache, variable batch size).</p>
 *
 * <p>The fix consults the C++ {@code OpTraitTable} via
 * {@code NativeOps.getOpTraits(opName)} and only stamps the VALDEP flag
 * when the authoritative trait bitmask carries
 * {@code OP_TRAIT_VALUE_DEPENDENT_SHAPE} (1 &lt;&lt; 8) or
 * {@code OP_TRAIT_DATA_DEPENDENT} (1 &lt;&lt; 9). Ops missing from the
 * trait table fall back to the legacy heuristic so unclassified ops stay
 * safe.</p>
 *
 * <p>These tests build minimal SameDiff graphs, compile the plan without
 * executing it, and inspect the per-slot flag. They do not depend on any
 * external model download.</p>
 *
 * <p>Run from platform-tests:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=TestValueDependentShapeClassification \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2&gt;&amp;1 | tee /tmp/valdep-classification.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP Compiler — value-dependent-shape classification")
public class TestValueDependentShapeClassification {

    /** Mirrors {@code OP_TRAIT_VALUE_DEPENDENT_SHAPE} in libnd4j. */
    private static final int OP_TRAIT_VALUE_DEPENDENT_SHAPE = 1 << 8;
    /** Mirrors {@code OP_TRAIT_DATA_DEPENDENT} in libnd4j. */
    private static final int OP_TRAIT_DATA_DEPENDENT = 1 << 9;
    private static final int VALDEP_MASK =
            OP_TRAIT_VALUE_DEPENDENT_SHAPE | OP_TRAIT_DATA_DEPENDENT;

    // ───────────────────────────────────────────────────────────────── NON-VALDEP

    @Test
    @DisplayName("gather(params, indices) is NOT value-dependent — INT indices alone don't imply VALDEP")
    public void gatherIsNotValueDependent() {
        assertOpValdepFlag("gather", false, sd -> {
            SDVariable params = sd.placeHolder("params", DataType.FLOAT, 10, 4);
            SDVariable indices = sd.placeHolder("indices", DataType.INT32, -1);
            SDVariable out = sd.gather("out", params, indices, 0);
            sd.setOutputs("out");
        });
    }

    @Test
    @DisplayName("gather_nd(params, indices) is NOT value-dependent")
    public void gatherNdIsNotValueDependent() {
        assertOpValdepFlag("gather_nd", false, sd -> {
            SDVariable params = sd.placeHolder("params", DataType.FLOAT, 10, 4);
            SDVariable indices = sd.placeHolder("indices", DataType.INT32, -1, 2);
            SDVariable out = sd.gatherNd("out", params, indices);
            sd.setOutputs("out");
        });
    }

    @Test
    @DisplayName("concat is NOT value-dependent — output shape comes from input shapes")
    public void concatIsNotValueDependent() {
        assertOpValdepFlag("concat", false, sd -> {
            SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 4);
            SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 4);
            SDVariable out = sd.concat("out", 0, a, b);
            sd.setOutputs("out");
        });
    }

    // ───────────────────────────────────────────────────────────────── VALDEP

    @Test
    @DisplayName("fill(shape, value) IS value-dependent — output shape is read from input buffer")
    public void fillIsValueDependent() {
        // fill's shape input is an INT vector whose *values* define the output rank/shape.
        // The trait table must stamp this op as VALDEP so shape drift triggers recompilation
        // rather than being silently accepted.
        assertOpValdepFlag("fill", true, sd -> {
            SDVariable shape = sd.placeHolder("shape", DataType.INT32, -1);
            SDVariable out = sd.fill("out", shape, DataType.FLOAT, 0.0);
            sd.setOutputs("out");
        });
    }

    @Test
    @DisplayName("reshape(x, shape) IS value-dependent when shape comes from an SDVariable")
    public void reshapeDynamicIsValueDependent() {
        assertOpValdepFlag("reshape", true, sd -> {
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 12);
            SDVariable shape = sd.placeHolder("shape", DataType.LONG, -1);
            SDVariable out = sd.reshape("out", x, shape);
            sd.setOutputs("out");
        });
    }

    // ────────────────────────────────────────────────── trait-table / JNI contract

    @Test
    @DisplayName("NativeOps.getOpTraits matches the Java compiler's VALDEP decision")
    public void getOpTraitsAgreesWithCompiler() {
        // The compiler must not drift from the native trait table. If this test fails,
        // the Java compiler is using stale constants or a different code path.
        int gatherTraits = NativeOpsHolder.getInstance().getDeviceNativeOps().getOpTraits("gather");
        int fillTraits = NativeOpsHolder.getInstance().getDeviceNativeOps().getOpTraits("fill");

        assertTrue(gatherTraits != 0,
                "gather must have a non-zero trait bitmask in OpTraitTable.cpp. " +
                        "Missing entry breaks value-dependent-shape classification.");
        assertEquals(0, gatherTraits & VALDEP_MASK,
                "gather must NOT carry VALUE_DEPENDENT_SHAPE or DATA_DEPENDENT traits — " +
                        "its output shape is derivable from input shapes alone.");

        assertTrue(fillTraits != 0, "fill must have a non-zero trait bitmask in OpTraitTable.cpp");
        assertTrue((fillTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE) != 0,
                "fill MUST carry VALUE_DEPENDENT_SHAPE — its shape input's values define the output shape.");
    }

    // ──────────────────────────────────────────────────────────── helpers

    @FunctionalInterface
    private interface GraphBuilder {
        void build(SameDiff sd);
    }

    /**
     * Build a SameDiff with the given graph, compile a DynamicShapePlan, locate the
     * slot whose opName matches {@code opName}, and assert its
     * {@code outputShapeDependsOnInputValues} flag equals {@code expected}.
     */
    private static void assertOpValdepFlag(String opName, boolean expected, GraphBuilder build) {
        SameDiff sd = SameDiff.create();
        try {
            build.build(sd);

            DynamicShapePlan plan = sd.compileDynamicShapePlan("out");
            assertNotNull(plan, "DynamicShapePlan should compile for a minimal " + opName + " graph");

            DynamicShapeSlot match = findSlotByOpName(plan, opName);
            assertNotNull(match,
                    "Could not locate a slot for op '" + opName + "' in the compiled plan. " +
                            "Slots=" + slotOpNames(plan));

            boolean actual = match.isOutputShapeDependsOnInputValues();
            if (actual != expected) {
                fail(opName + " outputShapeDependsOnInputValues=" + actual +
                        " expected=" + expected + ". " +
                        "This indicates DynamicShapePlanCompiler drifted from the C++ OpTraitTable — " +
                        "ops are being classified by the legacy hasIntLongInputs heuristic rather than " +
                        "the authoritative trait bitmask.");
            }
        } finally {
            try {
                sd.close();
            } catch (Exception ignored) {
            }
        }
    }

    private static DynamicShapeSlot findSlotByOpName(DynamicShapePlan plan, String opName) {
        DynamicShapeSlot[] slots = plan.getSlots();
        if (slots == null) return null;
        for (DynamicShapeSlot slot : slots) {
            if (slot == null) continue;
            if (opName.equalsIgnoreCase(slot.getOpName())) {
                return slot;
            }
        }
        return null;
    }

    private static String slotOpNames(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        if (slots == null) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < slots.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(slots[i] == null ? "null" : slots[i].getOpName());
        }
        sb.append("]");
        return sb.toString();
    }
}
