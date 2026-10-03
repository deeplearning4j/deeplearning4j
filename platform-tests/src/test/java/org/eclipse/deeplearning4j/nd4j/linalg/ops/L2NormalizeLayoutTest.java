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
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * DL4J's L2NormalizeVertex on a [minibatch, channels, h, w] activation: the norm over every axis but
 * the first, kept as [minibatch, 1, 1, 1], then a broadcast division along axis 0. A minibatch of 1
 * makes that norm cover every element.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class L2NormalizeLayoutTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void normOverEveryAxisButTheFirst(Nd4jBackend backend) {
        for (int minibatch : new int[]{1, 3}) {
            Nd4j.getRandom().setSeed(12345);
            INDArray x = Nd4j.rand(DataType.DOUBLE, minibatch, 2, 3, 3).subi(0.5);
            INDArray norm = x.norm2(true, 1, 2, 3);
            assertArrayEquals(new long[]{minibatch, 1, 1, 1}, norm.shape(), "minibatch " + minibatch + " norm shape");
            for (int m = 0; m < minibatch; m++) {
                double sum = 0;
                for (double v : x.get(NDArrayIndex.point(m)).dup().data().asDouble()) {
                    sum += v * v;
                }
                assertEquals(Math.sqrt(sum), norm.getDouble(m, 0, 0, 0), 1e-12, "minibatch " + minibatch + " norm " + m);
            }

            Transforms.max(norm, 1e-8, false);
            INDArray out = Nd4j.createUninitialized(DataType.DOUBLE, x.shape(), x.ordering());
            Nd4j.getExecutioner().exec(new BroadcastDivOp(x, norm, out, 0));
            for (int m = 0; m < minibatch; m++) {
                double[] row = x.get(NDArrayIndex.point(m)).dup().data().asDouble();
                double[] normalized = out.get(NDArrayIndex.point(m)).dup().data().asDouble();
                double n = norm.getDouble(m, 0, 0, 0);
                for (int i = 0; i < row.length; i++) {
                    assertEquals(row[i] / n, normalized[i], 1e-12,
                            "minibatch " + minibatch + " element " + m + "/" + i + " of " + Arrays.toString(x.shape()));
                }
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
