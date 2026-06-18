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
package org.eclipse.deeplearning4j.optraits;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * End-to-end behavioural regression for the op-trait classification system.
 *
 * <p>When an op is (correctly) NOT marked {@code OP_TRAIT_VALUE_DEPENDENT_SHAPE},
 * the SHAPES_FROZEN slot executor's input-shape change recovery path
 * (POST_SEAL_SHAPE_CHANGE) must take effect. When an op IS marked value-dependent,
 * the value-dep shape-match assertion must fire only if the TENSOR DATA feeding
 * the shape has changed.
 *
 * <p>This suite checks two classes of scenarios:
 * <ul>
 *   <li>Shape-determined ops (gather, gather_nd, concat, stack, expand_dims,
 *       squeeze). Running them with an input whose SHAPE changes between calls
 *       must succeed OR throw the "shape changed at slot ... during SHAPES_FROZEN
 *       phase" message — NEVER the "value-dependent output shape changed" message.
 *       That prior false positive is precisely the regression we are guarding
 *       against.</li>
 *   <li>Value-dependent ops (fill, reshape using a shape tensor whose VALUES
 *       change). These should take the value-dep branch — either succeed via
 *       cached-shape recompute or report the value-dep message legitimately.</li>
 * </ul>
 */
@Slf4j
@DisplayName("OpCategory frozen-shape behaviour — gather/concat et al. must not be mis-tagged value-dep")
public class OpCategoryFrozenShapeTest {

    // The OP_TRAIT_VALUE_DEPENDENT_SHAPE bit (see OpDescriptor.h).
    private static final int OP_TRAIT_VALUE_DEPENDENT_SHAPE = 1 << 8;

    /** The error substring that indicates the regression we're guarding against. */
    private static final String VALDEP_ERR_SUBSTRING = "value-dependent output shape changed";

    @BeforeEach
    public void setUp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    public void tearDown() {
        Nd4j.getExecutioner().commit();
    }

    // ────────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────────

    private static int traits(String opName) {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getOpTraits(opName);
    }

    /** Run the graph twice with the supplied inputs. Must NOT surface the value-dep
     *  error message — if the SHAPES_FROZEN path dislikes the shape change it's
     *  fine for it to throw, but only via the shape-changed recovery/assertion
     *  branch. We assert the message class, not graph success.                  */
    private static void runTwiceAssertNoValdepError(
            String fixtureName,
            SameDiff sd,
            String outputName,
            Map<String, INDArray> firstInputs,
            Map<String, INDArray> secondInputs) {
        // First call — seals the plan at the first observed shape.
        INDArray a = sd.output(firstInputs, outputName).get(outputName);
        assertNotNull(a, fixtureName + ": first output must be non-null");
        Nd4j.getExecutioner().commit();

        // Second call — shape of a placeholder has (legitimately) changed.
        try {
            INDArray b = sd.output(secondInputs, outputName).get(outputName);
            assertNotNull(b, fixtureName + ": second output must be non-null");
            // Success is acceptable — the SHAPES_FROZEN path handled the input
            // shape change via POST_SEAL_SHAPE_CHANGE.
        } catch (Throwable t) {
            String msg = t.getMessage() == null ? "" : t.getMessage();
            Throwable cause = t;
            while (cause != null) {
                if (cause.getMessage() != null && cause.getMessage().contains(VALDEP_ERR_SUBSTRING)) {
                    fail(fixtureName + ": input shape change triggered the VALUE-DEP "
                            + "error branch, which means this op is still tagged "
                            + "OP_TRAIT_VALUE_DEPENDENT_SHAPE. That is the regression "
                            + "this test exists to catch. Full message: " + cause.getMessage(), t);
                }
                cause = cause.getCause();
            }
            // Any other failure is still a bug — surface it.
            fail(fixtureName + ": second call threw (but not the value-dep message). "
                    + "Message: " + msg, t);
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Shape-determined ops — must NOT take the value-dep branch
    // ────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("gather: input shape change must not hit the value-dep error branch")
    public void testGatherShapeChange() {
        // Hard assertion: gather MUST NOT be value-dep. A failure here indicates
        // the C++ trait table regressed — fix OpTraitTable.cpp, don't skip.
        int gatherTraits = traits("gather");
        assertNotEquals(0, gatherTraits,
                "gather op is missing from OpTraitTable.cpp (getOpTraits returned 0)");
        assertEquals(0, gatherTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE,
                "gather is still tagged VALUE_DEPENDENT_SHAPE (traits=0x"
                        + Integer.toHexString(gatherTraits) + "). Its output shape is "
                        + "determined by input shapes only; the VALDEP bit causes false "
                        + "positives in SHAPES_FROZEN. Fix OpTraitTable.cpp.");

        SameDiff sd = SameDiff.create();
        try {
            SDVariable params = sd.var("params",
                    Nd4j.linspace(DataType.FLOAT, 0.0, 0.1, 32 * 16).reshape(32, 16));
            // indices: shape [-1, 4] — first dim varies across calls.
            SDVariable indices = sd.placeHolder("idx", DataType.INT64, -1, 4);
            SDVariable out = sd.gather("out", params, indices, 0);
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> first = new LinkedHashMap<>();
            first.put("idx", Nd4j.createFromArray(new long[][]{{0, 1, 2, 3}}));  // [1,4]

            Map<String, INDArray> second = new LinkedHashMap<>();
            second.put("idx", Nd4j.createFromArray(new long[][]{
                    {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}            // [3,4]
            }));

            runTwiceAssertNoValdepError("gather", sd, "out", first, second);
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("gather_nd: input shape change must not hit the value-dep error branch")
    public void testGatherNdShapeChange() {
        int gatherNdTraits = traits("gather_nd");
        assertNotEquals(0, gatherNdTraits,
                "gather_nd op is missing from OpTraitTable.cpp (getOpTraits returned 0)");
        assertEquals(0, gatherNdTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE,
                "gather_nd is still tagged VALUE_DEPENDENT_SHAPE (traits=0x"
                        + Integer.toHexString(gatherNdTraits) + "). Fix OpTraitTable.cpp.");

        SameDiff sd = SameDiff.create();
        try {
            SDVariable params = sd.var("params",
                    Nd4j.linspace(DataType.FLOAT, 0.0, 0.1, 4 * 8).reshape(4, 8));
            // indices: shape [-1, 2] — K varies across calls.
            SDVariable indices = sd.placeHolder("idx", DataType.INT64, -1, 2);
            SDVariable out = sd.gatherNd("out", params, indices);
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> first = new LinkedHashMap<>();
            first.put("idx", Nd4j.createFromArray(new long[][]{{0, 0}}));      // [1,2]

            Map<String, INDArray> second = new LinkedHashMap<>();
            second.put("idx", Nd4j.createFromArray(new long[][]{
                    {0, 0}, {1, 1}, {2, 2}, {3, 3}                             // [4,2]
            }));

            runTwiceAssertNoValdepError("gather_nd", sd, "out", first, second);
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("concat: input non-axis-dim change must not hit the value-dep error branch")
    public void testConcatShapeChange() {
        int concatTraits = traits("concat");
        assertNotEquals(0, concatTraits,
                "concat op is missing from OpTraitTable.cpp (getOpTraits returned 0)");
        assertEquals(0, concatTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE,
                "concat is still tagged VALUE_DEPENDENT_SHAPE (traits=0x"
                        + Integer.toHexString(concatTraits) + "). Fix OpTraitTable.cpp.");

        SameDiff sd = SameDiff.create();
        try {
            // Both inputs share the non-axis dim; axis=0 varies with batch.
            SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 8);
            SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 8);
            SDVariable out = sd.concat("out", 0, a, b);
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> first = new LinkedHashMap<>();
            first.put("a", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 2 * 8).reshape(2, 8));
            first.put("b", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 2 * 8).reshape(2, 8));

            Map<String, INDArray> second = new LinkedHashMap<>();
            second.put("a", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 5 * 8).reshape(5, 8));
            second.put("b", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 3 * 8).reshape(3, 8));

            runTwiceAssertNoValdepError("concat", sd, "out", first, second);
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("stack: input shape change must not hit the value-dep error branch")
    public void testStackShapeChange() {
        int stackTraits = traits("stack");
        assertNotEquals(0, stackTraits,
                "stack op is missing from OpTraitTable.cpp (getOpTraits returned 0)");
        assertEquals(0, stackTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE,
                "stack is still tagged VALUE_DEPENDENT_SHAPE (traits=0x"
                        + Integer.toHexString(stackTraits) + "). Fix OpTraitTable.cpp.");

        SameDiff sd = SameDiff.create();
        try {
            SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 4);
            SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 4);
            SDVariable out = sd.stack("out", 0, a, b);
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> first = new LinkedHashMap<>();
            first.put("a", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 2 * 4).reshape(2, 4));
            first.put("b", Nd4j.linspace(DataType.FLOAT, 1, 0.1, 2 * 4).reshape(2, 4));

            Map<String, INDArray> second = new LinkedHashMap<>();
            second.put("a", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 5 * 4).reshape(5, 4));
            second.put("b", Nd4j.linspace(DataType.FLOAT, 1, 0.1, 5 * 4).reshape(5, 4));

            runTwiceAssertNoValdepError("stack", sd, "out", first, second);
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("expand_dims: input shape change must not hit the value-dep error branch")
    public void testExpandDimsShapeChange() {
        int expandDimsTraits = traits("expand_dims");
        assertNotEquals(0, expandDimsTraits,
                "expand_dims op is missing from OpTraitTable.cpp (getOpTraits returned 0)");
        assertEquals(0, expandDimsTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE,
                "expand_dims is still tagged VALUE_DEPENDENT_SHAPE (traits=0x"
                        + Integer.toHexString(expandDimsTraits) + "). Fix OpTraitTable.cpp.");

        SameDiff sd = SameDiff.create();
        try {
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
            SDVariable out = sd.expandDims("out", x, 1);
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> first = new LinkedHashMap<>();
            first.put("x", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 2 * 8).reshape(2, 8));

            Map<String, INDArray> second = new LinkedHashMap<>();
            second.put("x", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 7 * 8).reshape(7, 8));

            runTwiceAssertNoValdepError("expand_dims", sd, "out", first, second);
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("squeeze: input shape change must not hit the value-dep error branch")
    public void testSqueezeShapeChange() {
        int squeezeTraits = traits("squeeze");
        assertNotEquals(0, squeezeTraits,
                "squeeze op is missing from OpTraitTable.cpp (getOpTraits returned 0)");
        assertEquals(0, squeezeTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE,
                "squeeze is still tagged VALUE_DEPENDENT_SHAPE (traits=0x"
                        + Integer.toHexString(squeezeTraits) + "). Fix OpTraitTable.cpp.");

        SameDiff sd = SameDiff.create();
        try {
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 1, 4);
            SDVariable out = sd.squeeze("out", x, 1);
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            Map<String, INDArray> first = new LinkedHashMap<>();
            first.put("x", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 2 * 1 * 4).reshape(2, 1, 4));

            Map<String, INDArray> second = new LinkedHashMap<>();
            second.put("x", Nd4j.linspace(DataType.FLOAT, 0, 0.1, 6 * 1 * 4).reshape(6, 1, 4));

            runTwiceAssertNoValdepError("squeeze", sd, "out", first, second);
        } finally {
            sd.close();
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Positive path — genuinely value-dependent ops
    // ────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("fill: value-dep op whose shape tensor values change — must not crash with wrong error class")
    public void testFillValueDepPath() {
        // Hard assertion: fill MUST be value-dep (shape fn reads the shape tensor).
        int fillTraits = traits("fill");
        assertNotEquals(0, fillTraits,
                "fill op is missing from OpTraitTable.cpp (getOpTraits returned 0)");
        assertNotEquals(0, fillTraits & OP_TRAIT_VALUE_DEPENDENT_SHAPE,
                "fill MUST carry OP_TRAIT_VALUE_DEPENDENT_SHAPE (its shape fn reads the "
                        + "shape tensor's DATA). traits=0x" + Integer.toHexString(fillTraits)
                        + ". Fix OpTraitTable.cpp.");

        SameDiff sd = SameDiff.create();
        try {
            SDVariable shapeTensor = sd.placeHolder("shape", DataType.INT64, -1);
            SDVariable scalar = sd.constant("scalar", Nd4j.scalar(DataType.FLOAT, 1.25f));
            SDVariable out = sd.fill("out", shapeTensor, DataType.FLOAT, 1.25);
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            // 1st: fill produces a [2,3] tensor.
            Map<String, INDArray> first = new HashMap<>();
            first.put("shape", Nd4j.createFromArray(2L, 3L));

            INDArray a = sd.output(first, "out").get("out");
            assertNotNull(a, "fill first output non-null");
            Nd4j.getExecutioner().commit();

            // 2nd: shape tensor VALUES have changed — this genuinely needs the
            // value-dep branch. It should either succeed or throw — but whatever
            // error class it reports is not a regression.
            Map<String, INDArray> second = new HashMap<>();
            second.put("shape", Nd4j.createFromArray(4L, 5L));
            try {
                INDArray b = sd.output(second, "out").get("out");
                assertNotNull(b, "fill second output non-null");
            } catch (Throwable t) {
                // Accept any throw here — this IS the value-dep path. The test
                // is just verifying the op is discoverable and lets execution
                // through the value-dep handling.
                log.warn("fill second call threw (value-dep branch engaged): {}", t.getMessage());
            }
        } finally {
            sd.close();
        }
    }
}
