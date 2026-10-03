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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A broadcast op applies y along x's TADs: with one broadcast dimension d, z[..., i, ...] = x[..., i, ...] op y[i]
 * where i indexes d, whatever y's rank (y may keep x's rank with every other axis 1). The CPU loops for rank 3, 4
 * and 5 once ranged over z's leading axes with y's own strides: they read past y's single element and left z
 * partly unwritten.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class BroadcastTadContractTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void broadcastAlongOneAxisOfEveryRank(Nd4jBackend backend) {
        long[][] shapes = {{3, 3}, {2, 3, 4}, {1, 2, 3, 3}, {2, 3, 4, 5}, {2, 3, 2, 2, 2}};
        for (long[] shape : shapes) {
            for (int axis = 0; axis < shape.length; axis++) {
                for (boolean keepRank : new boolean[]{false, true}) {
                    Nd4j.getRandom().setSeed(11 + axis);
                    INDArray x = Nd4j.rand(DataType.DOUBLE, shape).addi(0.5);
                    long[] yShape = new long[keepRank ? shape.length : 1];
                    Arrays.fill(yShape, 1);
                    yShape[keepRank ? axis : 0] = shape[axis];
                    INDArray y = Nd4j.rand(DataType.DOUBLE, yShape).addi(0.5);
                    double[] ys = y.dup().data().asDouble();

                    String what = Arrays.toString(shape) + " along " + axis + ", y " + Arrays.toString(yShape);
                    INDArray divided = Nd4j.createUninitialized(DataType.DOUBLE, shape);
                    Nd4j.getExecutioner().exec(new BroadcastDivOp(x, y, divided, axis));
                    INDArray added = Nd4j.createUninitialized(DataType.DOUBLE, shape);
                    Nd4j.getExecutioner().exec(new BroadcastAddOp(x, y, added, axis));

                    for (long i = 0; i < shape[axis]; i++) {
                        INDArray xSlice = slice(x, shape.length, axis, i);
                        INDArray dividedSlice = slice(divided, shape.length, axis, i);
                        INDArray addedSlice = slice(added, shape.length, axis, i);
                        double[] xs = xSlice.dup().data().asDouble();
                        double[] ds = dividedSlice.dup().data().asDouble();
                        double[] as = addedSlice.dup().data().asDouble();
                        for (int k = 0; k < xs.length; k++) {
                            assertEquals(xs[k] / ys[(int) i], ds[k], 1e-12, "div " + what + " index " + i + "/" + k);
                            assertEquals(xs[k] + ys[(int) i], as[k], 1e-12, "add " + what + " index " + i + "/" + k);
                        }
                    }
                }
            }
        }
    }

    private static INDArray slice(INDArray array, int rank, int axis, long index) {
        INDArrayIndex[] indices = new INDArrayIndex[rank];
        for (int d = 0; d < rank; d++) {
            indices[d] = d == axis ? NDArrayIndex.point(index) : NDArrayIndex.all();
        }
        return array.get(indices);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
