/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.helper;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class INDArrayHelperTest {
    @Test
    public void when_inputHasIncorrectShape_expect_outputWithCorrectShape() {
        // Arrange
        INDArray input = Nd4j.create(new double[] { 1.0, 2.0, 3.0});

        // Act
        INDArray output = INDArrayHelper.forceCorrectShape(input);

        // Assert
        assertEquals(2, output.shape().length);
        assertEquals(1, output.shape()[0]);
        assertEquals(3, output.shape()[1]);
    }

    @Test
    public void when_inputHasCorrectShape_expect_outputWithSameShape() {
        // Arrange
        INDArray input = Nd4j.create(new double[] { 1.0, 2.0, 3.0}).reshape(1, 3);

        // Act
        INDArray output = INDArrayHelper.forceCorrectShape(input);

        // Assert
        assertEquals(2, output.shape().length);
        assertEquals(1, output.shape()[0]);
        assertEquals(3, output.shape()[1]);
    }

    @Test
    public void when_inputHasOneDimension_expect_outputWithTwoDimensions() {
        // Arrange
        INDArray input = Nd4j.create(new double[] { 1.0 });

        // Act
        INDArray output = INDArrayHelper.forceCorrectShape(input);

        // Assert
        assertEquals(2, output.shape().length);
        assertEquals(1, output.shape()[0]);
        assertEquals(1, output.shape()[1]);
    }

    @Test
    public void when_callingCreateBatchForShape_expect_INDArrayWithCorrectShapeAndOriginalShapeUnchanged() {
        // Arrange
        long[] shape = new long[] { 1, 3, 4};

        // Act
        INDArray output = INDArrayHelper.createBatchForShape(2, shape);

        // Assert
        // Output shape
        assertArrayEquals(new long[] { 2, 3, 4 }, output.shape());

        // Input should remain unchanged
        assertArrayEquals(new long[] { 1, 3, 4 }, shape);

    }

    @Test
    public void when_callingCreateRnnBatchForShape_expect_INDArrayWithCorrectShapeAndOriginalShapeUnchanged() {
        // Arrange
        long[] shape = new long[] { 1, 3, 1 };

        // Act
        INDArray output = INDArrayHelper.createRnnBatchForShape(5, shape);

        // Assert
        // Output shape
        assertArrayEquals(new long[] { 1, 3, 5 }, output.shape());

        // Input should remain unchanged
        assertArrayEquals(new long[] { 1, 3, 1 }, shape);
    }

}
