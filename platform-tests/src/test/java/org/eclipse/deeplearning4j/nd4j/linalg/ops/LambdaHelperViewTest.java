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

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Helpers that apply an element function through the NDArray lambda loops (thresholdedrelu and its gradient on the
 * CPU) read and wrote views at twice their offset: bufferAsT() already points at a view's first element, and the
 * loops added the view's offset again, so the last row of a matrix was read past the end of the matrix.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class LambdaHelperViewTest extends BaseNd4jTestWithBackends {

    private static INDArray matrix() {
        return Nd4j.createFromArray(new double[][]{{-1, 2, 0.5, 3}, {4, -5, 0.8, 0.1}, {0.7, 9, -2, 1.5}});
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void thresholdedReluOfTheLastRow(Nd4jBackend backend) {
        INDArray lastRow = matrix().getRow(2);
        assertEquals(8, lastRow.offset());
        INDArray out = Nd4j.createUninitialized(DataType.DOUBLE, 4);

        Nd4j.exec(DynamicCustomOp.builder("thresholdedrelu").addInputs(lastRow).addOutputs(out)
                .addFloatingPointArguments(0.75).build());

        assertEquals(Nd4j.createFromArray(0.0, 9.0, 0.0, 1.5), out);
    }

    /** In place on a row, only that row changes. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void thresholdedReluInPlaceOnTheLastRow(Nd4jBackend backend) {
        INDArray matrix = matrix();
        INDArray lastRow = matrix.getRow(2);

        Nd4j.exec(DynamicCustomOp.builder("thresholdedrelu").addInputs(lastRow).addOutputs(lastRow)
                .addFloatingPointArguments(0.75).build());

        INDArray expected = matrix();
        expected.putRow(2, Nd4j.createFromArray(0.0, 9.0, 0.0, 1.5));
        assertEquals(expected, matrix);
    }

    /** The gradient passes where the input row 2 is above the threshold, taking it from gradient row 1. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void thresholdedReluGradientOfRowViews(Nd4jBackend backend) {
        INDArray inputs = matrix();
        INDArray gradients = Nd4j.createFromArray(new double[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        INDArray out = Nd4j.createUninitialized(DataType.DOUBLE, 4);

        Nd4j.exec(DynamicCustomOp.builder("thresholdedrelu_bp").addInputs(inputs.getRow(2), gradients.getRow(1))
                .addOutputs(out).addFloatingPointArguments(0.75).build());

        assertEquals(Nd4j.createFromArray(0.0, 6.0, 0.0, 8.0), out);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
