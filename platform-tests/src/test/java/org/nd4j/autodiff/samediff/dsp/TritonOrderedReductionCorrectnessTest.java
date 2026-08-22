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
package org.nd4j.autodiff.samediff.dsp;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Correctness gate for the Triton ordered-reduction emission path
 * (emitOrderedReductionValue in TritonIRBuilder_module.cpp).
 *
 * <p>Before that path existed, both live REDUCTION emission copies (the 1D
 * buildModule path and the sectioned path) special-cased only
 * mean/max/min/prod and silently emitted a plain <b>sum</b> for every other
 * reduction routed to a compiled REDUCTION section — norm1, norm2, norm_max,
 * variance, stdev, logsumexp, argmax and argmin all produced sums whenever
 * REDUCTION sections compiled (tritonCompileAll / section-fusion configs).
 *
 * <p>This test forces REDUCTION sections to compile (tritonCompileAll +
 * tritonIncludeTypes=REDUCTION) and checks every reduction kind against the
 * native eager reference. If Triton compilation of a case fails, the segment
 * executes natively and the comparison still passes — the test's regression
 * value is that a compiled-but-wrong kernel can no longer hide.
 */
public class TritonOrderedReductionCorrectnessTest {

    private static final int ROWS = 4;
    private static final int COLS = 37;   // odd, spans multiple lanes of the ordered tree
    private static final int WARMUP_EXECUTIONS = 6;
    private static final double EPS = 1e-2;

    private static boolean isTritonAvailable() {
        try {
            return Nd4j.getNativeOps().isTritonAvailable();
        } catch (Throwable t) {
            return false;
        }
    }

    /** Deterministic input with sign changes and distinct per-row extrema. */
    private static INDArray testInput() {
        INDArray base = Nd4j.linspace(1, ROWS * COLS, ROWS * COLS, DataType.FLOAT)
                .reshape(ROWS, COLS);
        return Transforms.sin(base, true);
    }

    static final class ReductionCase {
        final String name;
        final BiFunction<SameDiff, SDVariable, SDVariable> graph;
        final Function<INDArray, INDArray> expected;
        final boolean exactLongCompare;

        ReductionCase(String name,
                      BiFunction<SameDiff, SDVariable, SDVariable> graph,
                      Function<INDArray, INDArray> expected,
                      boolean exactLongCompare) {
            this.name = name;
            this.graph = graph;
            this.expected = expected;
            this.exactLongCompare = exactLongCompare;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    static Stream<ReductionCase> reductionCases() {
        List<ReductionCase> cases = new ArrayList<>();
        cases.add(new ReductionCase("reduce_sum",
                (sd, x) -> x.sum("out", 1), in -> in.sum(1), false));
        cases.add(new ReductionCase("reduce_mean",
                (sd, x) -> x.mean("out", 1), in -> in.mean(1), false));
        cases.add(new ReductionCase("reduce_max",
                (sd, x) -> x.max("out", 1), in -> in.max(1), false));
        cases.add(new ReductionCase("reduce_min",
                (sd, x) -> x.min("out", 1), in -> in.min(1), false));
        cases.add(new ReductionCase("reduce_prod",
                (sd, x) -> x.prod("out", 1), in -> in.prod(1), false));
        cases.add(new ReductionCase("reduce_norm1",
                (sd, x) -> x.norm1("out", 1), in -> in.norm1(1), false));
        cases.add(new ReductionCase("reduce_norm2",
                (sd, x) -> x.norm2("out", 1), in -> in.norm2(1), false));
        cases.add(new ReductionCase("norm_max",
                (sd, x) -> x.normmax("out", 1), in -> in.normmax(1), false));
        cases.add(new ReductionCase("reduce_variance_biased",
                (sd, x) -> sd.variance("out", x, false, 1), in -> in.var(false, 1), false));
        cases.add(new ReductionCase("reduce_variance_corrected",
                (sd, x) -> sd.variance("out", x, true, 1), in -> in.var(true, 1), false));
        cases.add(new ReductionCase("reduce_stdev_biased",
                (sd, x) -> sd.standardDeviation("out", x, false, 1), in -> in.std(false, 1), false));
        cases.add(new ReductionCase("reduce_logsumexp",
                (sd, x) -> sd.math().logSumExp("out", x, 1),
                in -> {
                    INDArray rowMax = in.max(1);
                    INDArray shifted = in.subColumnVector(rowMax.reshape(ROWS, 1));
                    return Transforms.log(Transforms.exp(shifted, true).sum(1), true).add(rowMax);
                }, false));
        cases.add(new ReductionCase("argmax",
                (sd, x) -> sd.argmax("out", x, 1), in -> Nd4j.argMax(in, 1), true));
        cases.add(new ReductionCase("argmin",
                (sd, x) -> sd.argmin("out", x, 1), in -> Nd4j.argMin(in, 1), true));
        return cases.stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("reductionCases")
    public void testTritonCompiledReductionMatchesNative(ReductionCase c) {
        // Triton is multi-backend (NVIDIA PTX, AMD/ROCm AMDGCN, ZLUDA, and it
        // coexists with the Vulkan backend), so never sniff backend/executioner
        // class names here — DeviceAwareOpExecutioner hides them anyway. Asking
        // the loaded native backend whether its Triton JIT is present is the
        // only correct availability gate.
        assumeTrue(isTritonAvailable(), "Triton unavailable on this build");

        final boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        final String prevIncludeTypes = Nd4j.getEnvironment().tritonIncludeTypes();
        try {
            Nd4j.getEnvironment().setTritonCompileAll(true);
            Nd4j.getEnvironment().setTritonIncludeTypes("REDUCTION");

            INDArray input = testInput();

            SameDiff sd = SameDiff.create();
            SDVariable ph = sd.placeHolder("in", DataType.FLOAT, ROWS, COLS);
            // Elementwise producer in front of the reduction so the segment
            // contains an ELEMENTWISE section feeding the REDUCTION section.
            SDVariable pre = ph.mul(2.0);
            c.graph.apply(sd, pre);
            sd.setGraphExecutionMode(GraphExecutionMode.TRITON);

            Map<String, INDArray> placeholders = Collections.singletonMap("in", input);
            INDArray actual = null;
            for (int i = 0; i < WARMUP_EXECUTIONS; i++) {
                actual = sd.output(placeholders, "out").get("out");
            }

            INDArray expected = c.expected.apply(input.mul(2.0));

            if (c.exactLongCompare) {
                assertArrayEquals(expected.toLongVector(), actual.toLongVector(),
                        c.name + ": Triton-compiled index reduction diverged from native. expected="
                                + expected + " actual=" + actual);
            } else {
                INDArray actualF = actual.castTo(expected.dataType());
                assertTrue(expected.equalsWithEps(actualF, EPS),
                        c.name + ": Triton-compiled reduction diverged from native. expected="
                                + expected + " actual=" + actualF);
            }
        } finally {
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
            Nd4j.getEnvironment().setTritonIncludeTypes(prevIncludeTypes);
        }
    }

    /**
     * SmolDocling patch-mask regression: the imported graph reduces both 16-wide
     * patch dimensions of an INT64 tensor. Treating only iArgs[0] as the axis
     * silently sums 16 values instead of 16*16 and corrupts the downstream
     * greater/not-equals/Where coordinate count.
     */
    @Test
    public void testMultiAxisInt64SumMatchesNative() {
        assumeTrue(isTritonAvailable(), "Triton unavailable on this build");

        final boolean prevCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        final String prevIncludeTypes = Nd4j.getEnvironment().tritonIncludeTypes();
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.ones(DataType.INT64, 1, 32, 32, 16, 16);
        try {
            Nd4j.getEnvironment().setTritonCompileAll(true);
            Nd4j.getEnvironment().setTritonIncludeTypes("REDUCTION");

            SDVariable ph = sd.placeHolder("in", DataType.INT64, 1, 32, 32, 16, 16);
            SDVariable out = ph.sum("out", false, -1, -2);
            sd.setOutputs(out.name());
            sd.setGraphExecutionMode(GraphExecutionMode.TRITON);

            INDArray actual = null;
            Map<String, INDArray> placeholders = Collections.singletonMap("in", input);
            for (int i = 0; i < WARMUP_EXECUTIONS; i++) {
                actual = sd.output(placeholders, out.name()).get(out.name());
            }

            assertArrayEquals(new long[]{1, 32, 32}, actual.shape());
            long[] values = actual.toLongVector();
            for (int i = 0; i < values.length; i++) {
                assertEquals(256L, values[i], "multi-axis sum diverged at output " + i);
            }
        } finally {
            input.close();
            sd.close();
            Nd4j.getEnvironment().setTritonCompileAll(prevCompileAll);
            Nd4j.getEnvironment().setTritonIncludeTypes(prevIncludeTypes);
        }
    }
}
