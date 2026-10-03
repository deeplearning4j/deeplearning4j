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
import org.nd4j.linalg.api.ops.impl.shape.MergeMax;
import org.nd4j.linalg.api.ops.impl.shape.MergeMaxIndex;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The steps of DL4J's ElementWiseVertex(Max) on its layouts: activations and the index array in
 * 'f' order, the masks in 'c' order. Each step is compared element by element through logical
 * indices.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class MergeMaxLayoutTest extends BaseNd4jTestWithBackends {

    private static final long ROWS = 150;
    private static final long COLUMNS = 5;

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void mergeMaxAndItsIndexAgreeAcrossOrders(Nd4jBackend backend) {
        for (char inputOrder : new char[]{'c', 'f'}) {
            for (char outputOrder : new char[]{'c', 'f'}) {
                Nd4j.getRandom().setSeed(7);
                INDArray a = Nd4j.rand(DataType.DOUBLE, ROWS, COLUMNS).dup(inputOrder);
                INDArray b = Nd4j.rand(DataType.DOUBLE, ROWS, COLUMNS).dup(inputOrder);
                String layout = "inputs '" + inputOrder + "', output '" + outputOrder + "'";

                INDArray max = Nd4j.createUninitialized(DataType.DOUBLE, new long[]{ROWS, COLUMNS}, outputOrder);
                MergeMax mergeMax = new MergeMax(a, b);
                mergeMax.addOutputArgument(max);
                Nd4j.getExecutioner().exec(mergeMax);

                INDArray index = Nd4j.createUninitialized(DataType.INT, new long[]{ROWS, COLUMNS}, outputOrder);
                MergeMaxIndex mergeMaxIndex = new MergeMaxIndex(a, b);
                mergeMaxIndex.addOutputArgument(index);
                Nd4j.getExecutioner().exec(mergeMaxIndex);

                for (long r = 0; r < ROWS; r++) {
                    for (long c = 0; c < COLUMNS; c++) {
                        double va = a.getDouble(r, c);
                        double vb = b.getDouble(r, c);
                        String at = layout + " at [" + r + ", " + c + "]";
                        assertEquals(Math.max(va, vb), max.getDouble(r, c), 0.0, "mergemax " + at);
                        assertEquals(vb > va ? 1.0 : 0.0, index.getDouble(r, c), 0.0, "mergemaxindex " + at);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void matchConditionMaskFollowsLogicalIndices(Nd4jBackend backend) {
        for (char xOrder : new char[]{'c', 'f'}) {
            for (char zOrder : new char[]{'c', 'f'}) {
                INDArray index = Nd4j.createUninitialized(DataType.INT, new long[]{ROWS, COLUMNS}, xOrder);
                for (long r = 0; r < ROWS; r++) {
                    for (long c = 0; c < COLUMNS; c++) {
                        index.putScalar(new long[]{r, c}, (r * 7 + c * 3) % 2);
                    }
                }
                for (int value = 0; value < 2; value++) {
                    INDArray mask = Nd4j.create(DataType.BOOL, new long[]{ROWS, COLUMNS}, zOrder);
                    Nd4j.getExecutioner().exec(new MatchConditionTransform(index, mask, Conditions.equals(value)));
                    for (long r = 0; r < ROWS; r++) {
                        for (long c = 0; c < COLUMNS; c++) {
                            boolean expected = index.getDouble(r, c) == value;
                            assertEquals(expected, mask.getDouble(r, c) != 0.0,
                                    "x '" + xOrder + "', z '" + zOrder + "', value " + value + " at [" + r + ", " + c + "]");
                        }
                    }
                }
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
