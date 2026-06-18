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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountZero;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

@NativeTag
public class ReduceLongExecutionIsolationTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void directDimensionalReduceLongOpsMatchExpected(Nd4jBackend backend) {
        INDArray second = makeInput(DataType.DOUBLE).mul(2.0);
        INDArray expectedNonZero = second.neq(0).castTo(DataType.LONG).sum(2);
        INDArray expectedZero = second.eq(0).castTo(DataType.LONG).sum(2);

        INDArray actualNonZero = Nd4j.getExecutioner().exec(new CountNonZero(second.dup(), 2));
        INDArray actualZero = Nd4j.getExecutioner().exec(new CountZero(second.dup(), 2));

        assertTrue(expectedNonZero.equals(actualNonZero),
                "Direct dimensional countNonZero execution returned the wrong values.\nExpected: "
                        + describe(expectedNonZero) + "\nActual: " + describe(actualNonZero));
        assertTrue(expectedZero.equals(actualZero),
                "Direct dimensional countZero execution returned the wrong values.\nExpected: "
                        + describe(expectedZero) + "\nActual: " + describe(actualZero));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sameDiffIntermediateReduceLongOpsMatchExpected(Nd4jBackend backend) {
        SameDiff sd = buildReduceLongGraph();
        INDArray input = makeInput(DataType.DOUBLE);
        INDArray second = input.mul(2.0);
        INDArray expectedNonZero = second.neq(0).castTo(DataType.LONG).sum(2);
        INDArray expectedZero = second.eq(0).castTo(DataType.LONG).sum(2);

        Map<String, INDArray> outputs = sd.output(Map.of("in", input), "countNonZero", "countZero");

        assertTrue(expectedNonZero.equals(outputs.get("countNonZero")),
                "SameDiff dimensional countNonZero on an intermediate value returned the wrong values.\nExpected: "
                        + describe(expectedNonZero) + "\nActual: " + describe(outputs.get("countNonZero")));
        assertTrue(expectedZero.equals(outputs.get("countZero")),
                "SameDiff dimensional countZero on an intermediate value returned the wrong values.\nExpected: "
                        + describe(expectedZero) + "\nActual: " + describe(outputs.get("countZero")));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void serializedSameDiffIntermediateReduceLongOpsMatchExpected(Nd4jBackend backend) throws IOException {
        SameDiff original = buildReduceLongGraph();
        ByteBuffer serialized = original.asFlatBuffers(true);
        SameDiff restored = SameDiff.fromFlatBuffers(serialized);

        INDArray input = makeInput(DataType.DOUBLE);
        INDArray second = input.mul(2.0);
        INDArray expectedNonZero = second.neq(0).castTo(DataType.LONG).sum(2);
        INDArray expectedZero = second.eq(0).castTo(DataType.LONG).sum(2);

        Map<String, INDArray> outputs = restored.output(Map.of("in", input), "countNonZero", "countZero");

        assertTrue(expectedNonZero.equals(outputs.get("countNonZero")),
                "Serialized SameDiff dimensional countNonZero on an intermediate value returned the wrong values.\nExpected: "
                        + describe(expectedNonZero) + "\nActual: " + describe(outputs.get("countNonZero")));
        assertTrue(expectedZero.equals(outputs.get("countZero")),
                "Serialized SameDiff dimensional countZero on an intermediate value returned the wrong values.\nExpected: "
                        + describe(expectedZero) + "\nActual: " + describe(outputs.get("countZero")));
    }

    private static SameDiff buildReduceLongGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.DOUBLE, 3, 4, 5);
        SDVariable second = in.mul(2.0);
        sd.math().countNonZero("countNonZero", second, 2);
        sd.math().countZero("countZero", second, 2);
        return sd;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void directDimensionalCountNonZeroAcrossInputTypes(Nd4jBackend backend) {
        for (DataType inputType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray second = makeInput(inputType).mul(2.0);
            INDArray expected = second.neq(0).castTo(DataType.LONG).sum(2);
            INDArray actual = Nd4j.getExecutioner().exec(new CountNonZero(second.dup(), 2));

            assertTrue(expected.equals(actual),
                    "Direct dimensional countNonZero failed for inputType=" + inputType + ".\nExpected: "
                            + describe(expected) + "\nActual: " + describe(actual));
        }
    }

    private static INDArray makeInput(DataType dataType) {
        INDArray input = Nd4j.linspace(1, 60, 60, DataType.DOUBLE).reshape('c', 3, 4, 5).castTo(dataType);
        input.putScalar(0, 0, 0, 0.0);
        input.putScalar(0, 1, 3, 0.0);
        input.putScalar(1, 2, 4, 0.0);
        input.putScalar(2, 3, 1, 0.0);
        return input;
    }

    private static String describe(INDArray array) {
        return "dtype=" + array.dataType()
                + ", shape=" + Arrays.toString(array.shape())
                + ", values=" + array.toStringFull();
    }
}
