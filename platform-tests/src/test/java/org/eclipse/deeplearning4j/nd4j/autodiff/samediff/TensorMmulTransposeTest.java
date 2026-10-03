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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * tensorMmul's transposeX, transposeY and transposeZ reverse the axes of the first input, the second input and the
 * result (numpy's .T), and the contracted axes refer to the transposed inputs. The native op once ignored the flags
 * (and the gradient never received them), so each combination must equal the same contraction of explicitly permuted
 * arrays, in value and in both gradients.
 */
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TensorMmulTransposeTest extends BaseNd4jTestWithBackends {

    private static long[] reversed(int rank) {
        long[] axes = new long[rank];
        for (int i = 0; i < rank; i++) axes[i] = rank - 1 - i;
        return axes;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void flagsTransposeInputsAndResult(Nd4jBackend backend) {
        INDArray x = Nd4j.linspace(DataType.DOUBLE, -1.2, 0.1, 24).reshape(2, 3, 4);
        INDArray y = Nd4j.linspace(DataType.DOUBLE, 0.7, -0.05, 60).reshape(4, 3, 5);
        for (int flags = 0; flags < 8; flags++) {
            boolean tx = (flags & 1) != 0, ty = (flags & 2) != 0, tz = (flags & 4) != 0;
            // Contract the size-3 axis and the size-4 axis of the (transposed) inputs; the result is [2, 5] or [5, 2].
            int[] dimsX = {1, tx ? 0 : 2};
            int[] dimsY = {1, ty ? 2 : 0};
            String label = "transposeX=" + tx + ", transposeY=" + ty + ", transposeZ=" + tz;

            Map<String, INDArray> flagged = run(x, y, dimsX, dimsY, tx, ty, tz, false);
            Map<String, INDArray> permuted = run(x, y, dimsX, dimsY, tx, ty, tz, true);
            for (String name : new String[]{"z", "x", "y"}) {
                assertEquals(permuted.get(name), flagged.get(name), label + ": " + name);
            }
        }
    }

    /**
     * The output z and the gradients of x and y for loss = sum(w * z): with the flags, or with permute ops around a
     * contraction without them.
     */
    private static Map<String, INDArray> run(INDArray xValue, INDArray yValue, int[] dimsX, int[] dimsY, boolean tx,
                                             boolean ty, boolean tz, boolean explicitPermutes) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xValue.dup());
        SDVariable y = sd.var("y", yValue.dup());
        SDVariable z;
        if (explicitPermutes) {
            SDVariable xIn = tx ? sd.permute(x, reversed(3)) : x;
            SDVariable yIn = ty ? sd.permute(y, reversed(3)) : y;
            SDVariable product = sd.tensorMmul(xIn, yIn, dimsX, dimsY, false, false, false);
            z = tz ? sd.permute(product, reversed(2)) : product;
        } else {
            z = sd.tensorMmul(x, y, dimsX, dimsY, tx, ty, tz);
        }
        z = sd.identity("z", z);
        long[] zShape = tz ? new long[]{5, 2} : new long[]{2, 5};
        SDVariable w = sd.constant("w", Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 10).reshape(zShape));
        z.mul(w).sum().markAsLoss();

        Map<String, INDArray> result = new HashMap<>(sd.calculateGradients(Collections.emptyMap(), "x", "y"));
        result.put("z", sd.output(Collections.emptyMap(), "z").get("z"));
        return result;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
