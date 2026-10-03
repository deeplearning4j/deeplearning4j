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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The scatter ops' gradients against their closed forms where indices repeat and values tie, cases a numerical
 * gradient check cannot settle. The loss is sum(w * scatter(ref, indices, updates)) with a different weight for each
 * output element, so a gradient sent to the wrong element shows. Index 1 takes updates 0, 2 and 4, index 3 updates 1
 * and 5, index 4 update 3; slices 0 and 2 are untouched.
 */
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class ScatterGradientTest extends BaseNd4jTestWithBackends {

    private static final long[] INDEX = {1, 3, 1, 4, 1, 3};
    private static final int SLICES = 5;
    private static final int WIDTH = 3;

    private static final double[][] REF = {{1.5, -2, 2.5}, {-1.25, 3, 0.75}, {2, -0.5, 1}, {0.6, 1.1, -1.7},
            {-0.8, 2.2, 1.3}};
    private static final double[][] UPDATES = {{0.9, -1.4, 1.2}, {-0.7, 1.6, -1.1}, {1.3, 0.8, -0.6},
            {-1.5, 0.55, 1.45}, {0.65, -1.2, 0.95}, {1.05, -0.85, 1.35}};
    private static final double[][] WEIGHTS = {{0.3, -0.7, 1.1}, {0.9, 0.2, -0.4}, {-1.3, 0.6, 0.5},
            {0.8, -0.9, 1.7}, {-0.25, 1.4, 0.35}};

    /** Ties: at index 1, column 0 has ref and updates 0 and 4 at 5; column 1 updates 0 and 2 at 7; and so on. */
    private static final double[][] TIED_REF = {{2, 2, 2}, {5, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}};
    private static final double[][] TIED_UPDATES = {{5, 7, 1}, {9, -9, 0.5}, {2, 7, 0}, {0, 0, 10}, {5, 3, 1},
            {9, 2, 0.5}};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void repeatedIndices(Nd4jBackend backend) {
        for (String op : new String[]{"add", "sub", "mul", "div", "update", "max", "min", "ndAdd", "ndSub",
                "ndUpdate"}) {
            check(op, REF, UPDATES, WEIGHTS);
        }
    }

    /** Where ref and updates tie for the maximum (minimum), each of them gets an equal share of the gradient. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tiesShareTheGradientEvenly(Nd4jBackend backend) {
        check("max", TIED_REF, TIED_UPDATES, WEIGHTS);
        check("min", negate(TIED_REF), negate(TIED_UPDATES), WEIGHTS);
    }

    /** A zero factor at a repeated index: no division by it, and a second zero there zeroes every factor's gradient. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void zeroFactorsAtARepeatedIndex(Nd4jBackend backend) {
        double[][] updates = {{0, 2, 0}, {-0.7, 1.6, -1.1}, {3, 0, 0}, {-1.5, 0.55, 1.45}, {4, 5, 6},
                {1.05, -0.85, 1.35}};
        check("mul", REF, updates, WEIGHTS);
    }

    private static void check(String op, double[][] ref, double[][] updates, double[][] weights) {
        Map<String, INDArray> gradients = gradients(op, ref, updates, weights);
        double[][][] expected = closedForm(op, ref, updates, weights);
        assertClose(op + " d/dref", expected[0], gradients.get("ref"));
        assertClose(op + " d/dupdates", expected[1], gradients.get("updates"));
    }

    private static Map<String, INDArray> gradients(String op, double[][] ref, double[][] updates,
                                                   double[][] weights) {
        SameDiff sd = SameDiff.create();
        SDVariable r = sd.var("ref", Nd4j.createFromArray(ref));
        SDVariable u = sd.var("updates", Nd4j.createFromArray(updates));
        INDArray index = Nd4j.createFromArray(INDEX);
        SDVariable i = sd.constant("indices", op.startsWith("nd") ? index.reshape(INDEX.length, 1) : index);
        SDVariable out;
        switch (op) {
            case "add":
                out = sd.scatterAdd(r, i, u);
                break;
            case "sub":
                out = sd.scatterSub(r, i, u);
                break;
            case "mul":
                out = sd.scatterMul(r, i, u);
                break;
            case "div":
                out = sd.scatterDiv(r, i, u);
                break;
            case "update":
                out = sd.scatterUpdate(r, i, u);
                break;
            case "max":
                out = sd.scatterMax(r, i, u);
                break;
            case "min":
                out = sd.scatterMin(r, i, u);
                break;
            case "ndAdd":
                out = sd.scatterNdAdd(r, i, u);
                break;
            case "ndSub":
                out = sd.scatterNdSub(r, i, u);
                break;
            case "ndUpdate":
                out = sd.scatterNdUpdate(r, i, u);
                break;
            default:
                throw new IllegalArgumentException(op);
        }
        SDVariable loss = out.mul(sd.constant("weights", Nd4j.createFromArray(weights))).sum();
        loss.markAsLoss();
        return sd.calculateGradients(Collections.emptyMap(), "ref", "updates");
    }

    /** The gradients of sum(w * out) by each op's definition: [0] with respect to ref, [1] to the updates. */
    private static double[][][] closedForm(String op, double[][] ref, double[][] u, double[][] w) {
        double[][] dRef = new double[SLICES][WIDTH];
        double[][] dUpdates = new double[INDEX.length][WIDTH];
        for (int s = 0; s < SLICES; s++) {
            List<Integer> at = new ArrayList<>();
            for (int k = 0; k < INDEX.length; k++) {
                if (INDEX[k] == s) at.add(k);
            }
            for (int p = 0; p < WIDTH; p++) {
                switch (op) {
                    case "add":
                    case "ndAdd":
                        dRef[s][p] = w[s][p];
                        for (int k : at) dUpdates[k][p] = w[s][p];
                        break;
                    case "sub":
                    case "ndSub":
                        dRef[s][p] = w[s][p];
                        for (int k : at) dUpdates[k][p] = -w[s][p];
                        break;
                    case "mul": {
                        double product = 1;
                        for (int k : at) product *= u[k][p];
                        dRef[s][p] = w[s][p] * product;
                        for (int k : at) {
                            double others = 1;
                            for (int j : at) {
                                if (j != k) others *= u[j][p];
                            }
                            dUpdates[k][p] = w[s][p] * ref[s][p] * others;
                        }
                        break;
                    }
                    case "div": {
                        double out = ref[s][p];
                        double product = 1;
                        for (int k : at) {
                            out /= u[k][p];
                            product *= u[k][p];
                        }
                        dRef[s][p] = w[s][p] / product;
                        for (int k : at) dUpdates[k][p] = -w[s][p] * out / u[k][p];
                        break;
                    }
                    case "update":
                    case "ndUpdate":
                        dRef[s][p] = at.isEmpty() ? w[s][p] : 0;
                        if (!at.isEmpty()) dUpdates[at.get(at.size() - 1)][p] = w[s][p];
                        break;
                    case "max":
                    case "min": {
                        double extreme = ref[s][p];
                        for (int k : at) {
                            extreme = op.equals("max") ? Math.max(extreme, u[k][p]) : Math.min(extreme, u[k][p]);
                        }
                        int count = ref[s][p] == extreme ? 1 : 0;
                        for (int k : at) {
                            if (u[k][p] == extreme) count++;
                        }
                        dRef[s][p] = ref[s][p] == extreme ? w[s][p] / count : 0;
                        for (int k : at) dUpdates[k][p] = u[k][p] == extreme ? w[s][p] / count : 0;
                        break;
                    }
                    default:
                        throw new IllegalArgumentException(op);
                }
            }
        }
        return new double[][][]{dRef, dUpdates};
    }

    private static double[][] negate(double[][] values) {
        double[][] negated = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            negated[i] = new double[values[i].length];
            for (int j = 0; j < values[i].length; j++) {
                negated[i][j] = -values[i][j];
            }
        }
        return negated;
    }

    private static void assertClose(String label, double[][] expected, INDArray actual) {
        assertArrayEquals(new long[]{expected.length, expected[0].length}, actual.shape(), label + " shape");
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[i].length; j++) {
                assertEquals(expected[i][j], actual.getDouble(i, j), 1e-12 * (1 + Math.abs(expected[i][j])),
                        label + " [" + i + ", " + j + "]");
            }
        }
    }
}
