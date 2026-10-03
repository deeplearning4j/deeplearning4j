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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Concatenation copies whole buffers only for arrays whose elements fill them densely in their
 * own order. An in-place permute keeps an array's buffer and view flag under reordered strides,
 * so the flag cannot stand in for that check.
 */
@Tag(TagNames.NDARRAY_INDEXING)
@NativeTag
public class ConcatLayoutTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void concatReadsPermutedInputsThroughTheirStrides(Nd4jBackend backend) {
        for (int axis = 0; axis < 3; axis++) {
            Nd4j.getRandom().setSeed(3 + axis);
            // [4, 2, 3] in memory as [2, 3, 4]: permuted in place, so not a view
            INDArray permuted = Nd4j.rand(DataType.DOUBLE, 2, 3, 4).permutei(2, 0, 1);
            long[] otherShape = permuted.shape().clone();
            otherShape[axis] = 2;
            INDArray other = Nd4j.rand(DataType.DOUBLE, otherShape);
            for (boolean permutedFirst : new boolean[]{true, false}) {
                INDArray first = permutedFirst ? permuted : other;
                INDArray second = permutedFirst ? other : permuted;
                INDArray joined = Nd4j.concat(axis, first, second);
                long[] expectedShape = first.shape().clone();
                expectedShape[axis] += second.size(axis);
                String what = "axis " + axis + ", permuted " + (permutedFirst ? "first" : "second");
                assertArrayEquals(expectedShape, joined.shape(), what);
                for (long i = 0; i < expectedShape[0]; i++) {
                    for (long j = 0; j < expectedShape[1]; j++) {
                        for (long k = 0; k < expectedShape[2]; k++) {
                            long[] at = {i, j, k};
                            long split = first.size(axis);
                            INDArray source = at[axis] < split ? first : second;
                            long[] sourceAt = at.clone();
                            if (at[axis] >= split) sourceAt[axis] -= split;
                            assertEquals(source.getDouble(sourceAt), joined.getDouble(at), 0.0,
                                    what + " at " + Arrays.toString(at));
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
