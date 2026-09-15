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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNormLinear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.BaseNd4jTestWithBackends;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the fused RMSNorm + Linear op.
 * Verifies that helpers::rmsNormLinear produces identical results to
 * the decomposed path: matmul(rmsNorm(x, gamma, eps), W).
 */
public class RmsNormLinearTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testRmsNormBfloatInputFloatGamma() {
        // Qwen uses BF16 activations and FLOAT one-centered normalization scales.
        // This ordinary finite scale also detects reading FLOAT bytes as BF16.
        float scale = Float.intBitsToFloat(0x3f807fc1);
        float eps = 1e-6f;
        try (INDArray input = Nd4j.ones(DataType.BFLOAT16, 2, 16);
             INDArray gamma = Nd4j.valueArrayOf(new long[]{16}, scale, DataType.FLOAT);
             INDArray output = Nd4j.create(DataType.BFLOAT16, 2, 16)) {
            Nd4j.exec(new RmsNorm(input, gamma, output, eps));
            double expected = scale / Math.sqrt(1.0 + eps);
            for (int row = 0; row < 2; row++) {
                for (int col = 0; col < 16; col++) {
                    assertEquals(expected, output.getDouble(row, col), 0.008,
                            "BF16 RMS normalization must consume gamma in its own FLOAT dtype");
                }
            }
        }
    }

    @Test
    public void testRmsNormFloatInputHalfGammaRank4() {
        float eps = 1e-6f;
        long[] shape = {1, 17, 16, 128};
        long length = 1L * 17 * 16 * 128;

        INDArray input = Nd4j.linspace(DataType.FLOAT, -0.5, 0.001, length).reshape(shape);
        INDArray gamma = Nd4j.linspace(DataType.FLOAT, 0.5, 0.004, 128).castTo(DataType.HALF);
        INDArray expected = input.mul(computeInvRms(input, eps)).mul(gamma.castTo(DataType.FLOAT));
        INDArray actual = Nd4j.create(DataType.FLOAT, shape);

        Nd4j.exec(new RmsNorm(input, gamma, actual, eps));

        assertEquals(DataType.FLOAT, actual.dataType());
        assertFalse(actual.isNaN().any(), "FLOAT input + HALF gamma produced NaN");
        assertFalse(actual.isInfinite().any(), "FLOAT input + HALF gamma produced Inf");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
                "FLOAT input + HALF gamma RMSNorm max diff too large: " + maxDiff);
    }

    /**
     * Compute invRms = 1 / sqrt(mean(x^2, axis=-1) + eps), shape [M, 1]
     */
    private static INDArray computeInvRms(INDArray x, float eps) {
        INDArray xSq = x.mul(x);
        INDArray meanSq = xSq.mean(true, -1);
        INDArray rms = Transforms.sqrt(meanSq.add(eps));
        return rms.rdiv(1.0);
    }

    private static final DataType[] NORM_TYPES = {
            DataType.HALF, DataType.BFLOAT16, DataType.FLOAT, DataType.DOUBLE
    };

    static Stream<Arguments> normalizationCases() {
        List<Arguments> cases = new ArrayList<>();
        for (DataType inputType : NORM_TYPES) {
            for (DataType gammaType : NORM_TYPES) {
                for (String layout : new String[]{"c", "f", "view"}) {
                    for (int width : new int[]{1, 17, 257}) {
                        cases.add(Arguments.of(inputType, gammaType, layout, width));
                    }
                }
            }
        }
        return cases.stream();
    }

    private static double normTolerance(DataType type) {
        return type == DataType.BFLOAT16 ? 0.012 : type == DataType.HALF ? 0.0015
                : type == DataType.FLOAT ? 3e-6 : 2e-12;
    }

    private static double[] logicalValues(INDArray array) {
        try (INDArray copy = array.dup('c')) {
            if (copy.dataType() == DataType.DOUBLE) return copy.data().asDouble();
            try (INDArray doubles = copy.castTo(DataType.DOUBLE)) {
                return doubles.data().asDouble();
            }
        }
    }

    @Test
    public void testDoubleReadbackPreservesNormalizationReference() {
        // Reference extraction must neither round DOUBLE to FLOAT nor overflow
        // finite DOUBLE fixtures. Check the same readback path used below.
        double[] values = {1.0 + Math.scalb(1.0, -40), 1e100};
        try (INDArray input = Nd4j.createFromArray(values)) {
            assertArrayEquals(values, logicalValues(input), 0.0,
                    "normalization reference readback must preserve DOUBLE storage");
        }
    }

    private static void assertNormReference(INDArray x, INDArray gamma, INDArray skip, INDArray bias,
                                            INDArray actual, INDArray hidden, double eps) {
        double[] xv = logicalValues(x), gv = gamma == null ? null : logicalValues(gamma);
        double[] sv = skip == null ? null : logicalValues(skip);
        double[] bv = bias == null ? null : logicalValues(bias);
        double[] av = logicalValues(actual), hv = hidden == null ? null : logicalValues(hidden);
        int width = (int) x.size(-1);
        double tolerance = normTolerance(x.dataType());
        assertEquals(x.dataType(), actual.dataType());
        for (int row = 0; row < xv.length / width; row++) {
            double sum = 0;
            for (int j = 0; j < width; j++) {
                double v = xv[row * width + j] + (sv == null ? 0 : sv[row * width + j])
                        + (bv == null ? 0 : bv[j]);
                sum += v * v;
            }
            double inv = 1 / Math.sqrt(sum / width + eps);
            for (int j = 0; j < width; j++) {
                int i = row * width + j;
                double v = xv[i] + (sv == null ? 0 : sv[i]) + (bv == null ? 0 : bv[j]);
                double expected = v * inv * (gv == null ? 1 : gv[j]);
                assertEquals(expected, av[i], tolerance * Math.max(1, Math.abs(expected)),
                        "normalization at row=" + row + " feature=" + j);
                if (hv != null) assertEquals(v, hv[i], tolerance * Math.max(1, Math.abs(v)),
                        "residual hidden output at " + i);
            }
        }
    }

    @ParameterizedTest(name = "rms_{0}_gamma{1}_{2}_K{3}")
    @MethodSource("normalizationCases")
    public void testRmsNormDtypeAndLayoutContract(DataType inputType, DataType gammaType,
                                                String layout, int width) {
        boolean view = layout.equals("view");
        char order = layout.equals("f") ? 'f' : 'c';
        long[] shape = view ? new long[]{3, 4, 2L * width + 2} : new long[]{2, 3, width};
        try (INDArray inputBase = Nd4j.create(inputType, shape, order);
             INDArray gammaBase = Nd4j.create(gammaType, view ? 2L * width + 3 : width);
             INDArray outputBase = Nd4j.create(inputType, shape, 'c')) {
            inputBase.assign(-99);
            outputBase.assign(-77);
            gammaBase.assign(-55);
            INDArray input = view ? inputBase.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 4),
                    NDArrayIndex.interval(1, 2, 2L * width + 1)) : inputBase;
            INDArray output = view ? outputBase.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 4),
                    NDArrayIndex.interval(1, 2, 2L * width + 1)) : outputBase;
            INDArray gamma = view ? gammaBase.get(NDArrayIndex.interval(1, 2, 2L * width + 1)) : gammaBase;
            double[] values = new double[6 * width], scales = new double[width];
            for (int i = 0; i < values.length; i++) values[i] = (i % 29 - 14) * 0.125;
            for (int i = 0; i < width; i++) scales[i] = Float.intBitsToFloat(0x3f807fc1) + (i % 7) * 0.125;
            try (INDArray source = Nd4j.createFromArray(values).reshape('c', 2, 3, width);
                 INDArray scalesArray = Nd4j.createFromArray(scales)) {
                input.assign(source);
                gamma.assign(scalesArray);
            }
            // Repeated execution also checks prepare/register memory ownership.
            for (int repeat = 0; repeat < 2; repeat++) {
                Nd4j.exec(new RmsNorm(input, gamma, output, 1e-6));
                assertNormReference(input, gamma, null, null, output, null, 1e-6);
            }
            try (INDArray allocated = Nd4j.exec(new RmsNorm(input, gamma, 1e-6))[0]) {
                assertNormReference(input, gamma, null, null, allocated, null, 1e-6);
            }
            if (view) {
                double[] parent = logicalValues(outputBase);
                int pos = 0;
                for (int a = 0; a < 3; a++) for (int b = 0; b < 4; b++) {
                    for (int c = 0; c < 2 * width + 2; c++, pos++) {
                        boolean written = a >= 1 && b >= 1 && c >= 1 && c < 2 * width + 1 && c % 2 == 1;
                        if (!written) assertEquals(-77, parent[pos], 0, "output view wrote outside its region");
                    }
                }
            }
        }
    }

    @ParameterizedTest(name = "skip_{0}_gamma{1}_{2}_K{3}")
    @MethodSource("normalizationCases")
    public void testSkipRmsNormDtypeAndLayoutContract(DataType inputType, DataType gammaType,
                                                    String layout, int width) {
        char order = layout.equals("f") ? 'f' : 'c';
        boolean view = layout.equals("view");
        long[] shape = {2, 3, view ? 2L * width + 2 : width};
        try (INDArray xb = Nd4j.create(inputType, shape, order);
             INDArray sb = Nd4j.create(inputType, shape, order);
             INDArray zb = Nd4j.create(inputType, shape, 'c');
             INDArray hb = Nd4j.create(inputType, shape, order);
             INDArray gb = Nd4j.create(gammaType, view ? 2L * width + 2 : width);
             INDArray bb = Nd4j.create(inputType, view ? 2L * width + 2 : width)) {
            INDArray x = view ? xb.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2L * width + 1)) : xb;
            INDArray skip = view ? sb.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2L * width + 1)) : sb;
            INDArray z = view ? zb.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2L * width + 1)) : zb;
            INDArray hidden = view ? hb.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2L * width + 1)) : hb;
            INDArray gamma = view ? gb.get(NDArrayIndex.interval(1, 2, 2L * width + 1)) : gb;
            INDArray bias = view ? bb.get(NDArrayIndex.interval(1, 2, 2L * width + 1)) : bb;
            double[] values = new double[6 * width];
            for (int i = 0; i < values.length; i++) values[i] = (i % 19 - 9) * 0.125;
            try (INDArray source = Nd4j.createFromArray(values).reshape('c', 2, 3, width)) { x.assign(source); }
            skip.assign(0.375);
            gamma.assign(Float.intBitsToFloat(0x3f807fc1));
            bias.assign(-0.125);
            Nd4j.exec(DynamicCustomOp.builder("skip_rms_norm").addInputs(x, skip, gamma, bias)
                    .addOutputs(z, hidden).addFloatingPointArguments(1e-6).build());
            assertNormReference(x, gamma, skip, bias, z, hidden, 1e-6);
            Nd4j.exec(DynamicCustomOp.builder("skip_rms_norm").addInputs(x, skip, gamma)
                    .addOutputs(z).addFloatingPointArguments(1e-6).build());
            assertNormReference(x, gamma, skip, null, z, null, 1e-6);
        }
    }

    static Stream<Arguments> normBoundaryCases() {
        List<Arguments> cases = new ArrayList<>();
        for (DataType type : NORM_TYPES) {
            cases.add(Arguments.of(type, 4097, 300.0, 1e-6)); // low-precision square/partial overflow
            cases.add(Arguments.of(type, 17, 0.0, 1e-12));   // epsilon must not narrow to HALF zero
            cases.add(Arguments.of(type, 1, 2.0, 0.0));
            cases.add(Arguments.of(type, 33, 1e-3, 1e-6));
        }
        cases.add(Arguments.of(DataType.DOUBLE, 257, 1e100, 1e-6));
        return cases.stream();
    }

    @ParameterizedTest
    @MethodSource("normBoundaryCases")
    public void testRmsNormAccumulationAndNoGamma(DataType type, int width, double value, double eps) {
        try (INDArray input = Nd4j.valueArrayOf(new long[]{width}, value, type);
             INDArray output = Nd4j.create(type, width)) {
            Nd4j.exec(DynamicCustomOp.builder("rms_norm").addInputs(input).addOutputs(output)
                    .addFloatingPointArguments(eps).build());
            assertNormReference(input, null, null, null, output, null, eps);
            try (INDArray skip = Nd4j.zeros(type, width);
                 INDArray gamma = Nd4j.ones(DataType.FLOAT, width);
                 INDArray weight = Nd4j.ones(DataType.FLOAT, width, 1);
                 INDArray linear = Nd4j.create(type, 1, 1)) {
                Nd4j.exec(DynamicCustomOp.builder("skip_rms_norm").addInputs(input, skip, gamma)
                        .addOutputs(output).addFloatingPointArguments(eps).build());
                assertNormReference(input, gamma, skip, null, output, null, eps);
                INDArray matrix = input.reshape(1, width);
                Nd4j.exec(new RmsNormLinear(matrix, gamma, weight, linear, eps));
                double stored = logicalValues(input)[0];
                double expected = width * (stored / Math.sqrt(stored * stored + eps));
                assertEquals(expected, linear.getDouble(0), normTolerance(type) * Math.max(1, Math.abs(expected)));
            }
        }
    }

    static Stream<Arguments> linearDtypeCases() {
        List<Arguments> cases = new ArrayList<>();
        for (DataType type : NORM_TYPES) for (DataType gamma : NORM_TYPES) {
            for (DataType weight : NORM_TYPES) for (int rows : new int[]{1, 3}) {
                cases.add(Arguments.of(type, gamma, weight, rows));
            }
        }
        return cases.stream();
    }

    @ParameterizedTest(name = "linear_{0}_gamma{1}_weight{2}_M{3}")
    @MethodSource("linearDtypeCases")
    public void testRmsNormLinearMixedTypesAndViews(DataType type, DataType gammaType, DataType weightType, int rows) {
        int width = 17, columns = 5;
        try (INDArray xb = Nd4j.create(type, rows, 2 * width + 2);
             INDArray gb = Nd4j.create(gammaType, 2 * width + 2);
             INDArray wb = Nd4j.create(weightType, columns, width);
             INDArray zb = Nd4j.create(type, rows, 2 * columns + 2)) {
            INDArray x = xb.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2 * width + 1));
            INDArray gamma = gb.get(NDArrayIndex.interval(1, 2, 2 * width + 1));
            INDArray w = wb.transpose();
            INDArray z = zb.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2 * columns + 1));
            double[] values = new double[rows * width];
            double[] weights = new double[width * columns];
            for (int i = 0; i < values.length; i++) values[i] = (i % 19 - 9) * 0.125;
            for (int i = 0; i < weights.length; i++) weights[i] = (i % 11 - 5) * 0.0625;
            try (INDArray xv = Nd4j.createFromArray(values).reshape(rows, width);
                 INDArray wv = Nd4j.createFromArray(weights).reshape(width, columns)) {
                x.assign(xv);
                w.assign(wv);
            }
            gamma.assign(Float.intBitsToFloat(0x3f807fc1));
            zb.assign(-77);
            Nd4j.exec(new RmsNormLinear(x, gamma, w, z, 1e-6));
            double[] xv = logicalValues(x), gv = logicalValues(gamma), wv = logicalValues(w), zv = logicalValues(z);
            for (int row = 0; row < rows; row++) {
                double sum = 0;
                for (int k = 0; k < width; k++) sum += xv[row * width + k] * xv[row * width + k];
                double inv = 1 / Math.sqrt(sum / width + 1e-6);
                for (int col = 0; col < columns; col++) {
                    double expected = 0;
                    for (int k = 0; k < width; k++) expected += xv[row * width + k] * inv * gv[k] * wv[k * columns + col];
                    assertEquals(expected, zv[row * columns + col], normTolerance(type) * Math.max(1, Math.abs(expected)));
                }
            }
            double[] parent = logicalValues(zb);
            for (int row = 0; row < rows; row++) for (int col = 0; col < 2 * columns + 2; col++) {
                if (col % 2 == 0 || col == 2 * columns + 1)
                    assertEquals(-77, parent[row * (2 * columns + 2) + col], 0, "linear output view overrun");
            }
            assertEquals(type, z.dataType());
        }
    }

    static Stream<Arguments> gammaRoundingCases() {
        return Stream.of(Arguments.of(DataType.BFLOAT16, Float.intBitsToFloat(0x3f807fc1)),
                Arguments.of(DataType.HALF, 1.0003f));
    }

    @ParameterizedTest
    @MethodSource("gammaRoundingCases")
    public void testGammaIsNotRoundedBeforeNormalization(DataType type, float scale) {
        double inv = 1 / Math.sqrt(2.5 + 1e-6);
        // Both fixtures have inv * scale in [0.5, 1), so the storage ULP is
        // 2^-8 for BF16 and 2^-11 for HALF. Round independently to nearest-even:
        // Nd4j.create(double[], ..., type) goes through ArrayTypeConverters,
        // whose HALF/BF16 conversion truncates and is not an output-rounding oracle.
        double quantum = Math.scalb(1.0, type == DataType.BFLOAT16 ? -8 : -11);
        double first = Math.rint(inv * scale / quantum) * quantum;
        double[] expected = {first, 2 * first};
        // Gamma is in [1, 2), where its ULP is twice as large. Prove these
        // fixtures distinguish retaining FLOAT gamma from prematurely rounding it.
        double roundedGamma = Math.rint(scale / (2 * quantum)) * (2 * quantum);
        assertNotEquals(first, Math.rint(inv * roundedGamma / quantum) * quantum,
                "fixture must detect premature gamma rounding");
        try (INDArray input = Nd4j.create(new double[]{1, 2}, new long[]{1, 2}, type);
             INDArray gamma = Nd4j.valueArrayOf(new long[]{2}, scale, DataType.FLOAT);
             INDArray output = Nd4j.create(type, 1, 2)) {
            Nd4j.exec(new RmsNorm(input, gamma, output, 1e-6));
            assertArrayEquals(expected, logicalValues(output), 0.0,
                    "gamma must remain FLOAT until the final output store");
        }
    }

    @ParameterizedTest
    @MethodSource("normTypes")
    public void testRank3LinearFortranAndPermutedViews(DataType type) {
        for (String layout : new String[]{"c", "f", "permute"}) {
            boolean permuted = layout.equals("permute");
            long[] sourceShape = permuted ? new long[]{3, 17, 2} : new long[]{2, 3, 17};
            try (INDArray base = Nd4j.create(type, sourceShape, layout.equals("f") ? 'f' : 'c');
                 INDArray outputBase = Nd4j.create(type, new long[]{17, 2, 3}, 'c');
                 INDArray gamma = Nd4j.valueArrayOf(new long[]{17}, Float.intBitsToFloat(0x3f807fc1), DataType.FLOAT);
                 INDArray weight = Nd4j.create(type, 17, 17)) {
                INDArray input = permuted ? base.permute(2, 0, 1) : base;
                INDArray output = outputBase.permute(1, 2, 0);
                double[] values = new double[2 * 3 * 17];
                double[] identity = new double[17 * 17];
                for (int i = 0; i < values.length; i++) values[i] = (i % 23 - 11) * 0.125;
                for (int i = 0; i < 17; i++) identity[i * 17 + i] = 1;
                try (INDArray source = Nd4j.createFromArray(values).reshape(2, 3, 17);
                     INDArray eye = Nd4j.createFromArray(identity).reshape(17, 17)) {
                    input.assign(source);
                    weight.assign(eye);
                }
                Nd4j.exec(new RmsNormLinear(input, gamma, weight, output, 1e-6));
                assertNormReference(input, gamma, null, null, output, null, 1e-6);
            }
        }
    }

    static Stream<DataType> normTypes() {
        return Stream.of(NORM_TYPES);
    }

    @ParameterizedTest
    @MethodSource("normTypes")
    public void testNormalizationEmptyBatch(DataType type) {
        try (INDArray input = Nd4j.create(type, 0, 17);
             INDArray gamma = Nd4j.ones(DataType.FLOAT, 17);
             INDArray weight = Nd4j.ones(type, 17, 5);
             INDArray output = Nd4j.create(type, 0, 17);
             INDArray linear = Nd4j.create(type, 0, 5)) {
            Nd4j.exec(new RmsNorm(input, gamma, output, 1e-6));
            Nd4j.exec(new RmsNormLinear(input, gamma, weight, linear, 1e-6));
            Nd4j.exec(DynamicCustomOp.builder("skip_rms_norm").addInputs(input, input, gamma)
                    .addOutputs(output).addFloatingPointArguments(1e-6).build());
            assertTrue(output.isEmpty());
            assertTrue(linear.isEmpty());
            assertEquals(type, output.dataType());
            assertArrayEquals(new long[]{0, 17}, output.shape());
        }
        try (INDArray input = Nd4j.create(type, 2, 0);
             INDArray gamma = Nd4j.create(DataType.FLOAT, 0);
             INDArray output = Nd4j.create(type, 2, 0)) {
            Nd4j.exec(new RmsNorm(input, gamma, output, 1e-6));
            assertTrue(output.isEmpty());
        }
    }

    static Stream<String> invalidNormalizationCases() {
        return Stream.of("inputType", "gammaType", "gammaLength", "gammaRank", "inputRank", "outputType",
                "negativeEpsilon", "nanEpsilon", "infiniteEpsilon", "skipType", "skipShape", "biasType",
                "weightType", "weightShape", "emptyLinearFeatures");
    }

    @ParameterizedTest
    @MethodSource("invalidNormalizationCases")
    public void testNormalizationRejectsInvalidContracts(String invalid) {
        DataType inputType = invalid.equals("inputType") ? DataType.INT : DataType.FLOAT;
        DataType gammaType = invalid.equals("gammaType") ? DataType.INT : DataType.FLOAT;
        long[] inputShape = invalid.equals("inputRank") ? new long[0]
                : invalid.equals("emptyLinearFeatures") ? new long[]{2, 0} : new long[]{2, 17};
        long[] gammaShape = invalid.equals("gammaRank") ? new long[]{1, 17}
                : new long[]{invalid.equals("gammaLength") ? 16 : invalid.equals("emptyLinearFeatures") ? 0 : 17};
        double eps = invalid.equals("negativeEpsilon") ? -1 : invalid.equals("nanEpsilon") ? Double.NaN
                : invalid.equals("infiniteEpsilon") ? Double.POSITIVE_INFINITY : 1e-6;
        try (INDArray x = Nd4j.create(inputType, inputShape);
             INDArray gamma = Nd4j.create(gammaType, gammaShape);
             INDArray z = Nd4j.create(invalid.equals("outputType") ? DataType.DOUBLE : DataType.FLOAT, 2, 17)) {
            if (invalid.startsWith("skip") || invalid.equals("biasType")) {
                try (INDArray skip = Nd4j.create(invalid.equals("skipType") ? DataType.DOUBLE : DataType.FLOAT,
                        2, invalid.equals("skipShape") ? 16 : 17);
                     INDArray bias = Nd4j.create(invalid.equals("biasType") ? DataType.DOUBLE : DataType.FLOAT, 17)) {
                    assertThrows(RuntimeException.class, () -> Nd4j.exec(DynamicCustomOp.builder("skip_rms_norm")
                            .addInputs(x, skip, gamma, bias).addOutputs(z).addFloatingPointArguments(eps).build()));
                }
            } else if (invalid.startsWith("weight") || invalid.equals("emptyLinearFeatures")) {
                int rows = invalid.equals("weightShape") ? 16 : invalid.equals("emptyLinearFeatures") ? 0 : 17;
                try (INDArray weight = Nd4j.create(invalid.equals("weightType") ? DataType.INT : DataType.FLOAT, rows, 17)) {
                    assertThrows(RuntimeException.class, () -> Nd4j.exec(DynamicCustomOp.builder("rms_norm_linear")
                            .addInputs(x, gamma, weight).addOutputs(z).addFloatingPointArguments(eps).build()));
                }
            } else {
                assertThrows(RuntimeException.class, () -> Nd4j.exec(DynamicCustomOp.builder("rms_norm")
                        .addInputs(x, gamma).addOutputs(z).addFloatingPointArguments(eps).build()));
            }
        }
    }

    static Stream<Arguments> shapes() {
        return Stream.of(
            // M=1 decode path (fused kernel)
            Arguments.of(1, 64, 64),
            Arguments.of(1, 128, 256),
            Arguments.of(1, 1142, 1142),    // SmolDocling hidden dim
            Arguments.of(1, 1142, 3072),    // SmolDocling MLP up-proj
            // M>1 prefill path (rmsNorm + cuBLAS)
            Arguments.of(4, 64, 64),
            Arguments.of(16, 128, 256),
            Arguments.of(32, 512, 1024)
        );
    }

    /**
     * Compare fused rms_norm_linear against manual decomposition:
     *   normalized = x / sqrt(mean(x^2) + eps) * gamma
     *   output = normalized @ W
     */
    @ParameterizedTest
    @MethodSource("shapes")
    public void testRmsNormLinearMatchesDecomposed(int M, int K, int N) {
        float eps = 1e-5f;

        INDArray x = Nd4j.randn(DataType.FLOAT, M, K);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray W = Nd4j.randn(DataType.FLOAT, K, N).muli(0.02);

        // Reference: manual decomposition
        INDArray invRms = computeInvRms(x, eps);
        INDArray normalized = x.mul(invRms).mul(gamma);
        INDArray expected = normalized.mmul(W);

        // Fused op
        INDArray output = Nd4j.create(DataType.FLOAT, M, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, W, output, eps));

        // Tolerance: fused path uses float32 accumulation throughout
        double rtol = 1e-3;
        double atol = 1e-4;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float exp = expected.getFloat(i, j);
                float act = output.getFloat(i, j);
                float diff = Math.abs(exp - act);
                float tol = (float)(atol + rtol * Math.abs(exp));
                assertTrue(diff < tol,
                    String.format("Mismatch at [%d,%d]: expected=%f actual=%f diff=%f tol=%f (M=%d K=%d N=%d)",
                        i, j, exp, act, diff, tol, M, K, N));
            }
        }
    }

    static Stream<Arguments> mixedGammaShapes() {
        return Stream.of(
                Arguments.of(1, 128, 256),  // fused decode path
                Arguments.of(4, 128, 256)   // general prefill path
        );
    }

    @ParameterizedTest(name = "halfInputFloatGammaHalfWeight_M{0}")
    @MethodSource("mixedGammaShapes")
    public void testHalfInputFloatGammaHalfWeightMatchesFloatReference(int M, int K, int N) {
        float eps = 1e-6f;
        Nd4j.getRandom().setSeed(12345);

        INDArray x = Nd4j.randn(DataType.FLOAT, M, K).muli(0.25).castTo(DataType.HALF);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray W = Nd4j.randn(DataType.FLOAT, K, N).muli(0.02).castTo(DataType.HALF);

        // Build the reference from the exact HALF-rounded input and weights.
        INDArray xFloat = x.castTo(DataType.FLOAT);
        INDArray wFloat = W.castTo(DataType.FLOAT);
        INDArray invRms = computeInvRms(xFloat, eps);
        INDArray expected = xFloat.mul(invRms).mul(gamma).mmul(wFloat);

        INDArray output = Nd4j.create(DataType.HALF, M, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, W, output, eps));

        assertEquals(DataType.HALF, output.dataType());
        assertFalse(output.isNaN().any(),
                "HALF input + FLOAT gamma + HALF weight produced NaN for M=" + M);
        assertFalse(output.isInfinite().any(),
                "HALF input + FLOAT gamma + HALF weight produced Inf for M=" + M);

        double maxDiff = expected.sub(output.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
        assertTrue(maxDiff < 5e-3,
                "Mixed-dtype rms_norm_linear max diff too large for M=" + M + ": " + maxDiff);
    }

    @Test
    public void testDecodePathM1() {
        float eps = 1e-6f;
        int K = 256;
        int N = 512;

        INDArray x = Nd4j.randn(DataType.FLOAT, 1, K);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, K);
        INDArray W = Nd4j.randn(DataType.FLOAT, K, N).muli(0.01);

        // Reference
        INDArray invRms = computeInvRms(x, eps);
        INDArray expected = x.mul(invRms).mul(gamma).mmul(W);

        // Fused
        INDArray output = Nd4j.create(DataType.FLOAT, 1, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, W, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
            "M=1 decode path max diff too large: " + maxDiff);
    }

    @Test
    public void testGammaOnes() {
        float eps = 1e-5f;
        int M = 2, K = 64, N = 32;

        INDArray x = Nd4j.randn(DataType.FLOAT, M, K);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, K);
        INDArray W = Nd4j.randn(DataType.FLOAT, K, N).muli(0.02);

        // With gamma=1, rmsNorm just normalizes
        INDArray invRms = computeInvRms(x, eps);
        INDArray expected = x.mul(invRms).mmul(W);

        INDArray output = Nd4j.create(DataType.FLOAT, M, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, W, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3, "Gamma=ones max diff: " + maxDiff);
    }

    @Test
    public void testNonContiguousProjectionWeight() {
        float eps = 1e-6f;
        int M = 7, K = 64, N = 128;

        INDArray x = Nd4j.randn(DataType.FLOAT, M, K);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray weightRaw = Nd4j.randn(DataType.FLOAT, N, K).muli(0.02);
        INDArray weightTransposed = weightRaw.transpose();

        INDArray invRms = computeInvRms(x, eps);
        INDArray expected = x.mul(invRms).mul(gamma).mmul(weightTransposed);

        INDArray output = Nd4j.create(DataType.FLOAT, M, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, weightTransposed, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3, "Non-contiguous projection weight max diff: " + maxDiff);
    }

    @Test
    public void testRank3InputWithNonContiguousProjectionWeight() {
        float eps = 1e-6f;
        int batch = 1, seqLen = 7, K = 64, N = 128;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seqLen, K);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray weightRaw = Nd4j.randn(DataType.FLOAT, N, K).muli(0.02);
        INDArray weightTransposed = weightRaw.transpose();

        INDArray x2d = x.reshape('c', batch * seqLen, K);
        INDArray invRms = computeInvRms(x2d, eps);
        INDArray expected = x2d.mul(invRms).mul(gamma)
                .mmul(weightTransposed)
                .reshape('c', batch, seqLen, N);

        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, weightTransposed, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Rank-3 non-contiguous projection weight max diff: " + maxDiff);
    }

    @Test
    public void testSameDiffRank3InputWithPermuteProjectionWeight() {
        assertSameDiffRank3InputWithPermuteProjectionWeight(true);
    }

    @Test
    public void testSameDiffRank3InputWithPermuteProjectionWeightDspDisabled() {
        assertSameDiffRank3InputWithPermuteProjectionWeight(false);
    }

    private static void assertSameDiffRank3InputWithPermuteProjectionWeight(boolean dspEnabled) {
        float eps = 1e-6f;
        int batch = 1, seqLen = 7, K = 64, N = 128;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seqLen, K);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray weightRaw = Nd4j.randn(DataType.FLOAT, N, K).muli(0.02);
        INDArray weightTransposed = weightRaw.transpose();

        INDArray x2d = x.reshape('c', batch * seqLen, K);
        INDArray invRms = computeInvRms(x2d, eps);
        INDArray expected = x2d.mul(invRms).mul(gamma)
                .mmul(weightTransposed)
                .reshape('c', batch, seqLen, N);

        SameDiff sd = SameDiff.create();
        SDVariable xVar = sd.placeHolder("x", DataType.FLOAT, batch, seqLen, K);
        SDVariable gammaVar = sd.var("gamma", gamma);
        SDVariable weightRawVar = sd.var("lm_head", weightRaw);
        SDVariable weightTransposedVar = sd.permute("lm_head_t", weightRawVar, 1, 0);
        sd.nn().rmsNormLinear("lm_logits", xVar, gammaVar, weightTransposedVar, eps);
        sd.setOutputs("lm_logits");
        sd.setDspAutoCompileEnabled(dspEnabled);
        sd.setDspNativeAutoCompileEnabled(dspEnabled);
        sd.resetSession();

        Map<String, INDArray> ph = Collections.singletonMap("x", x);
        INDArray actual = sd.outputSingle(ph, "lm_logits");

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "SameDiff rank-3 permuted projection weight max diff: " + maxDiff);
    }
}
