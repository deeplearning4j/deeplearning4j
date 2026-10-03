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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The tensormmul gradients against their definition: C = tensordot(A, B) sums A[.., k] * B[k, ..] over the contracted
 * axes, so for the loss sum(w * C) an element of A receives the sum over every B element that shares its contracted
 * indices of w[C index] * B, and the same holds for B. The reference below sums that directly over every pair of
 * elements, with no axis bookkeeping that could repeat the implementation's mistakes. The contractions differ in how
 * many axes stay free on each side, in the number of contracted axes and in the order their pairs are listed.
 */
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TensorMmulGradientTest extends BaseNd4jTestWithBackends {

    /** The shapes of the two operands and the axes of each that are summed over, paired by position. */
    private static final class Contraction {
        final String name;
        final long[] aShape;
        final long[] bShape;
        final int[] axesA;
        final int[] axesB;

        Contraction(String name, long[] aShape, long[] bShape, int[] axesA, int[] axesB) {
            this.name = name;
            this.aShape = aShape;
            this.bShape = bShape;
            this.axesA = axesA;
            this.axesB = axesB;
        }

        List<Integer> freeA() {
            return free(aShape.length, axesA);
        }

        List<Integer> freeB() {
            return free(bShape.length, axesB);
        }

        /** The shape of the result: the free axes of A, then those of B. */
        long[] cShape() {
            List<Integer> freeA = freeA();
            List<Integer> freeB = freeB();
            long[] shape = new long[freeA.size() + freeB.size()];
            for (int i = 0; i < freeA.size(); i++) shape[i] = aShape[freeA.get(i)];
            for (int i = 0; i < freeB.size(); i++) shape[freeA.size() + i] = bShape[freeB.get(i)];
            return shape;
        }
    }

    private static final Contraction[] CONTRACTIONS = {
            new Contraction("cubes, one axis each, listed differently", new long[]{2, 2, 2}, new long[]{2, 2, 2},
                    new int[]{0}, new int[]{1}),
            new Contraction("matrix product", new long[]{3, 4}, new long[]{4, 5}, new int[]{1}, new int[]{0}),
            new Contraction("two free axes on the left, one on the right", new long[]{2, 3, 4}, new long[]{4, 5},
                    new int[]{2}, new int[]{0}),
            new Contraction("one free axis on the left, two on the right", new long[]{3, 4}, new long[]{2, 3, 5},
                    new int[]{0}, new int[]{1}),
            new Contraction("two contracted pairs, listed against the axis order", new long[]{2, 3, 4},
                    new long[]{3, 5, 2}, new int[]{0, 1}, new int[]{2, 0}),
    };

    /** A different weight for each element of the result, so a gradient sent to the wrong element shows. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void weightedLoss(Nd4jBackend backend) {
        for (Contraction contraction : CONTRACTIONS) {
            check(contraction, DataType.DOUBLE, 1e-12, false);
        }
    }

    /**
     * The result itself is the loss variable, as in a plain gradient check of the op: the gradient at it is the scalar
     * one, which the backward op spreads over every element of the result.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void scalarGradientAtTheResult(Nd4jBackend backend) {
        for (Contraction contraction : CONTRACTIONS) {
            check(contraction, DataType.DOUBLE, 1e-12, true);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void singlePrecision(Nd4jBackend backend) {
        // Single precision sums of a few dozen terms of order one: far inside 1e-3, far outside any mixed-up gradient.
        check(CONTRACTIONS[2], DataType.FLOAT, 1e-3, false);
        check(CONTRACTIONS[4], DataType.FLOAT, 1e-3, false);
        check(CONTRACTIONS[4], DataType.FLOAT, 1e-3, true);
    }

    private static void check(Contraction contraction, DataType type, double tolerance, boolean scalarSeed) {
        double[] a = values(product(contraction.aShape), 1);
        double[] b = values(product(contraction.bShape), 2);
        double[] w = scalarSeed ? ones(product(contraction.cShape())) : values(product(contraction.cShape()), 3);

        double[][] expected = closedForm(contraction, a, b, w);
        String label = contraction.name + (scalarSeed ? ", scalar gradient" : ", weighted") + ", " + type;

        // The forward result first, so a gradient that fails below is not just the gradient of a wrong product.
        assertClose(label + " forward", expected[2], contraction.cShape(), forward(contraction, a, b, type), tolerance);

        Map<String, INDArray> gradients = gradients(contraction, a, b, w, type, scalarSeed);
        assertClose(label + " d/dA", expected[0], contraction.aShape, gradients.get("a"), tolerance);
        assertClose(label + " d/dB", expected[1], contraction.bShape, gradients.get("b"), tolerance);
    }

    private static INDArray forward(Contraction contraction, double[] a, double[] b, DataType type) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("a", Nd4j.createFromArray(a).reshape('c', contraction.aShape).castTo(type));
        SDVariable y = sd.var("b", Nd4j.createFromArray(b).reshape('c', contraction.bShape).castTo(type));
        return sd.tensorMmul(x, y, contraction.axesA, contraction.axesB, false, false, false).eval();
    }

    private static Map<String, INDArray> gradients(Contraction contraction, double[] a, double[] b, double[] w,
                                                   DataType type, boolean scalarSeed) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("a", Nd4j.createFromArray(a).reshape('c', contraction.aShape).castTo(type));
        SDVariable y = sd.var("b", Nd4j.createFromArray(b).reshape('c', contraction.bShape).castTo(type));
        SDVariable result = sd.tensorMmul(x, y, contraction.axesA, contraction.axesB, false, false, false);
        if (scalarSeed) {
            result.markAsLoss();
        } else {
            SDVariable weights = sd.constant("w", Nd4j.createFromArray(w).reshape('c', contraction.cShape()).castTo(type));
            SDVariable loss = result.mul(weights).sum();
            loss.markAsLoss();
        }
        return sd.calculateGradients(Collections.emptyMap(), "a", "b");
    }

    /**
     * C and the gradients of sum(w * C), by summing over every pair of elements of A and B that agree on the
     * contracted indices: the pair adds A * B to the element of C at their free indices, so w[that element] * B goes to
     * the element of A and w[that element] * A to the element of B. [0] is the gradient of A, [1] that of B, [2] is C.
     */
    private static double[][] closedForm(Contraction contraction, double[] a, double[] b, double[] w) {
        List<Integer> freeA = contraction.freeA();
        List<Integer> freeB = contraction.freeB();
        double[] dA = new double[a.length];
        double[] dB = new double[b.length];
        double[] c = new double[product(contraction.cShape())];
        long[] aIndex = new long[contraction.aShape.length];
        long[] bIndex = new long[contraction.bShape.length];
        for (int ia = 0; ia < a.length; ia++) {
            unravel(ia, contraction.aShape, aIndex);
            for (int ib = 0; ib < b.length; ib++) {
                unravel(ib, contraction.bShape, bIndex);
                boolean sharesContractedIndices = true;
                for (int k = 0; k < contraction.axesA.length; k++) {
                    sharesContractedIndices &= aIndex[contraction.axesA[k]] == bIndex[contraction.axesB[k]];
                }
                if (!sharesContractedIndices) continue;

                long position = 0;
                for (int axis : freeA) position = position * contraction.aShape[axis] + aIndex[axis];
                for (int axis : freeB) position = position * contraction.bShape[axis] + bIndex[axis];
                c[(int) position] += a[ia] * b[ib];
                dA[ia] += w[(int) position] * b[ib];
                dB[ib] += w[(int) position] * a[ia];
            }
        }
        return new double[][]{dA, dB, c};
    }

    private static List<Integer> free(int rank, int[] contracted) {
        List<Integer> axes = new ArrayList<>();
        for (int axis = 0; axis < rank; axis++) {
            boolean isContracted = false;
            for (int c : contracted) isContracted |= c == axis;
            if (!isContracted) axes.add(axis);
        }
        return axes;
    }

    /** Neither zero nor repeating quickly: |value| in [0.4, 1.52], alternating in sign. */
    private static double[] values(int count, int seed) {
        double[] values = new double[count];
        for (int i = 0; i < count; i++) {
            double magnitude = 0.4 + ((i * 7 + seed * 13) % 11) / 9.0;
            values[i] = (i + seed) % 2 == 0 ? magnitude : -magnitude;
        }
        return values;
    }

    private static double[] ones(int count) {
        double[] ones = new double[count];
        Arrays.fill(ones, 1.0);
        return ones;
    }

    private static int product(long[] shape) {
        int product = 1;
        for (long dim : shape) product *= (int) dim;
        return product;
    }

    /** The C-order multi-index of a flat position. */
    private static void unravel(int flat, long[] shape, long[] index) {
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            index[axis] = flat % shape[axis];
            flat /= shape[axis];
        }
    }

    private static void assertClose(String label, double[] expected, long[] shape, INDArray actual, double tolerance) {
        assertArrayEquals(shape, actual.shape(), label + " shape");
        long[] index = new long[shape.length];
        for (int flat = 0; flat < expected.length; flat++) {
            unravel(flat, shape, index);
            assertEquals(expected[flat], actual.getDouble(index), tolerance * (1 + Math.abs(expected[flat])),
                    label + " " + Arrays.toString(index));
        }
    }
}
