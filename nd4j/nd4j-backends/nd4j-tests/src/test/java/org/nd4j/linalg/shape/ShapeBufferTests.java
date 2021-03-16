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

package org.nd4j.linalg.shape;

import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.util.ArrayUtil;

import static org.junit.jupiter.api.Assertions.*;


public class ShapeBufferTests extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testRank(Nd4jBackend backend) {
        long[] shape = {2, 4};
        long[] stride = {1, 2};
        val shapeInfoBuffer = Shape.createShapeInformation(shape, stride, 1, 'c', DataType.DOUBLE, false);
        val buff = shapeInfoBuffer.asNioLong();
        assertEquals(2, Shape.rank(buff));
    }


    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testArrCreationShape(Nd4jBackend backend) {
        val arr = Nd4j.linspace(1, 4, 4, DataType.DOUBLE).reshape(2, 2);
        for (int i = 0; i < 2; i++)
            assertEquals(2, arr.size(i));
        int[] stride = ArrayUtil.calcStrides(new int[] {2, 2});
        for (int i = 0; i < stride.length; i++) {
            assertEquals(stride[i], arr.stride(i));
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testShape(Nd4jBackend backend) {
        long[] shape = {2, 4};
        long[] stride = {1, 2};
        val shapeInfoBuffer = Shape.createShapeInformation(shape, stride, 1, 'c', DataType.DOUBLE, false);
        val buff = shapeInfoBuffer.asNioLong();
        val shapeView = Shape.shapeOf(buff);
        assertTrue(Shape.contentEquals(shape, shapeView));
        val strideView = Shape.stride(buff);
        assertTrue(Shape.contentEquals(stride, strideView));
        assertEquals('c', Shape.order(buff));
        assertEquals(1, Shape.elementWiseStride(buff));
        assertFalse(Shape.isVector(buff));
        assertTrue(Shape.contentEquals(shape, Shape.shapeOf(buff)));
        assertTrue(Shape.contentEquals(stride, Shape.stride(buff)));
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testBuff(Nd4jBackend backend) {
        long[] shape = {1, 2};
        long[] stride = {1, 2};
        val buff = Shape.createShapeInformation(shape, stride, 1, 'c', DataType.DOUBLE, false).asNioLong();
        assertTrue(Shape.isVector(buff));
    }


}
