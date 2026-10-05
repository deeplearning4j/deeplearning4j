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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.factory.ops.NDBase;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * repeat(input, repeats, axis) repeats each entry along the axis by its count (one count for all of them, or one for
 * each). The Java constructors handed the counts to the native op as a second input array and gave it no integer
 * arguments, which are where the op reads its counts and its axis from (the counts first, the axis last).
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class RepeatOpTest extends BaseNd4jTestWithBackends {

    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void arraysAreRepeatedByTheirCounts(Nd4jBackend backend) {
        NDBase base = new NDBase();
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT32}) {
            INDArray vector = Nd4j.createFromArray(1.0, 2.0, 3.0).castTo(type);
            // one count for every entry
            assertArrayEquals(new double[]{1, 1, 2, 2, 3, 3},
                    logical(base.repeat(vector, Nd4j.createFromArray(2L), 0)), 0.0, type + " vector by 2");
            // one count for each
            assertArrayEquals(new double[]{1, 2, 2, 3, 3, 3},
                    logical(base.repeat(vector, Nd4j.createFromArray(1L, 2L, 3L), 0)), 0.0, type + " vector by 1, 2, 3");
            // the counts in another integer type
            assertArrayEquals(new double[]{1, 1, 1, 2, 2, 2, 3, 3, 3},
                    logical(base.repeat(vector, Nd4j.createFromArray(3), -1)), 0.0, type + " vector by 3, int counts");

            INDArray matrix = Nd4j.createFromArray(new double[][]{{1, 2, 3}, {4, 5, 6}}).castTo(type);
            assertArrayEquals(new double[]{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6},
                    logical(base.repeat(matrix, Nd4j.createFromArray(2L), 0)), 0.0, type + " rows by 2");
            assertArrayEquals(new double[]{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
                    logical(base.repeat(matrix, Nd4j.createFromArray(3L), 1)), 0.0, type + " columns by 3");
            assertArrayEquals(new double[]{1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6},
                    logical(base.repeat(matrix, Nd4j.createFromArray(1L, 2L, 3L), 1)), 0.0,
                    type + " columns by 1, 2, 3");
            assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 4, 5, 6},
                    logical(base.repeat(matrix, Nd4j.createFromArray(1L, 2L), -2)), 0.0, type + " rows by 1, 2");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void samediffRepeatsByConstantCounts(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.constant("input", Nd4j.createFromArray(new double[][]{{1, 2, 3}, {4, 5, 6}}));
        SDVariable counts = sd.constant("counts", Nd4j.createFromArray(2L, 1L, 2L));
        SDVariable out = sd.repeat("out", input, counts, 1);
        assertArrayEquals(new double[]{1, 1, 2, 3, 3, 4, 4, 5, 6, 6}, logical(out.eval()), 0.0);
    }

    /** The counts become integer arguments, so a count only known when the graph runs cannot be used. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void samediffRefusesCountsWithoutAValue(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.constant("input", Nd4j.createFromArray(1.0, 2.0, 3.0));
        SDVariable counts = sd.placeHolder("counts", DataType.INT64, 1);
        assertThrows(IllegalStateException.class, () -> sd.repeat(input, counts, 0));
    }
}
