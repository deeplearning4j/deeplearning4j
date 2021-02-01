/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.rl4j.observation.transform.operation;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class ArrayToINDArrayTransformTest {

    @Test
    public void when_notUsingShape_expect_transformTo1DINDArray() {
        // Arrange
        ArrayToINDArrayTransform sut = new ArrayToINDArrayTransform();
        double[] data = new double[] { 1.0, 2.0, 3.0 };

        // Act
        INDArray result = sut.transform(data);

        // Assert
        assertArrayEquals(new long[] { 3 }, result.shape());
        assertEquals(1.0, result.getDouble(0), 0.00001);
        assertEquals(2.0, result.getDouble(1), 0.00001);
        assertEquals(3.0, result.getDouble(2), 0.00001);
    }

    @Test
    public void when_usingShape_expect_transformTo1DINDArray() {
        // Arrange
        ArrayToINDArrayTransform sut = new ArrayToINDArrayTransform(1, 3);
        double[] data = new double[] { 1.0, 2.0, 3.0 };

        // Act
        INDArray result = sut.transform(data);

        // Assert
        assertArrayEquals(new long[] { 1, 3 }, result.shape());
        assertEquals(1.0, result.getDouble(0, 0), 0.00001);
        assertEquals(2.0, result.getDouble(0, 1), 0.00001);
        assertEquals(3.0, result.getDouble(0, 2), 0.00001);
    }

}
