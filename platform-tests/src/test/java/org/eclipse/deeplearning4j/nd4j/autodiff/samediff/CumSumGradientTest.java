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

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The gradient of loss = sum(w * cumsum(x)) along the rows: x_k reaches every output whose sum holds it, so dL/dx_k is
 * the sum of w over those outputs. The gradient graph is built from serialized copies of the ops, and the copy of
 * CumSum once lost its reverse flag (and read its axes from the wrong argument), so a reverse scan was differentiated
 * as a forward one.
 */
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class CumSumGradientTest extends BaseNd4jTestWithBackends {

    private static final double[][] X = {{1, 2, 3, 4}, {-1, 0.5, 2, -3}};
    private static final double[][] W = {{1, 10, 100, 1000}, {2, 20, 200, 2000}};

    /** The sum of w over the outputs i whose (exclusive or inclusive, forward or reverse) sum contains x_k. */
    private static INDArray expectedGradient(boolean exclusive, boolean reverse) {
        double[][] out = new double[W.length][W[0].length];
        for (int r = 0; r < W.length; r++) {
            for (int k = 0; k < W[r].length; k++) {
                double sum = 0;
                for (int i = 0; i < W[r].length; i++) {
                    boolean holds = reverse ? (exclusive ? i < k : i <= k) : (exclusive ? i > k : i >= k);
                    if (holds) sum += W[r][i];
                }
                out[r][k] = sum;
            }
        }
        return Nd4j.createFromArray(out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyScanDirectionDifferentiatesItsOwnSums(Nd4jBackend backend) {
        for (boolean exclusive : new boolean[]{false, true}) {
            for (boolean reverse : new boolean[]{false, true}) {
                SameDiff sd = SameDiff.create();
                SDVariable x = sd.var("x", Nd4j.createFromArray(X));
                SDVariable w = sd.constant("w", Nd4j.createFromArray(W));
                sd.cumsum(x, exclusive, reverse, 1).mul(w).sum().markAsLoss();

                INDArray grad = sd.calculateGradients(Collections.emptyMap(), "x").get("x");
                assertEquals(expectedGradient(exclusive, reverse), grad,
                        "exclusive=" + exclusive + ", reverse=" + reverse);
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
