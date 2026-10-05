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
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * mergeavg is the sum of its inputs divided by their number, accumulated in at least FLOAT. The CPU helper multiplied
 * the sum by a reciprocal taken in float, so a DOUBLE average was a float reciprocal's worth off, and summed HALF and
 * BFLOAT16 values in their own type.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class MergeAvgTest extends BaseNd4jTestWithBackends {

    private static INDArray mergeAvg(INDArray... inputs) {
        return Nd4j.exec(DynamicCustomOp.builder("mergeavg").addInputs(inputs).build())[0];
    }

    private static double[] logical(INDArray array) {
        return array.dup('c').data().asDouble();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void doubleAveragesAreAccurateToDoublePrecision(Nd4jBackend backend) {
        for (int count : new int[]{2, 3, 5, 7, 11}) {
            Nd4j.getRandom().setSeed(count);
            INDArray[] inputs = new INDArray[count];
            double[] expected = new double[3 * 50];
            for (int i = 0; i < count; i++) {
                inputs[i] = Nd4j.rand(DataType.DOUBLE, 3, 50).muli(100.0);
                // the layout of an input does not change the average
                if (i % 2 == 1)
                    inputs[i] = inputs[i].dup('f');
                double[] values = logical(inputs[i]);
                for (int e = 0; e < expected.length; e++)
                    expected[e] += values[e];
            }
            for (int e = 0; e < expected.length; e++)
                expected[e] /= count;
            assertArrayEquals(expected, logical(mergeAvg(inputs)), 1e-13, count + " DOUBLE arrays");
        }
    }

    /** 2048 + 1 + 1 is 2050, not the 2048 a HALF sum stops at; 256 + 1 + 1 is 258, not the 256 of a BFLOAT16 sum. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void lowPrecisionSumsAreAccumulatedWide(Nd4jBackend backend) {
        INDArray half = mergeAvg(Nd4j.createFromArray(2048.0f).castTo(DataType.HALF),
                Nd4j.createFromArray(1.0f).castTo(DataType.HALF), Nd4j.createFromArray(1.0f).castTo(DataType.HALF));
        assertEquals(683.5, half.getDouble(0), 0.0, "HALF average of 2048, 1 and 1 (2050 / 3 = 683.33)");

        INDArray bfloat = mergeAvg(Nd4j.createFromArray(256.0f).castTo(DataType.BFLOAT16),
                Nd4j.createFromArray(1.0f).castTo(DataType.BFLOAT16),
                Nd4j.createFromArray(1.0f).castTo(DataType.BFLOAT16));
        assertEquals(86.0, bfloat.getDouble(0), 0.0, "BFLOAT16 average of 256, 1 and 1 (258 / 3 = 86)");
    }
}
