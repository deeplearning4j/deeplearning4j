/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.shape;

import lombok.val;
import org.nd4j.linalg.primitives.Pair;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class StaticShapeTests extends BaseNd4jTest {

    public StaticShapeTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testShapeInd2Sub() {
        long normalTotal = 0;
        long n = 1000;
        for (int i = 0; i < n; i++) {
            long start = System.nanoTime();
            Shape.ind2subC(new int[] {2, 2}, 1);
            long end = System.nanoTime();
            normalTotal += Math.abs(end - start);
        }

        normalTotal /= n;
        System.out.println(normalTotal);

        System.out.println("C " + Arrays.toString(Shape.ind2subC(new int[] {2, 2}, 1)));
        System.out.println("F " + Arrays.toString(Shape.ind2sub(new int[] {2, 2}, 1)));
    }


    @Test
    public void testBufferToIntShapeStrideMethods() {
        //Specifically: Shape.shape(IntBuffer), Shape.shape(DataBuffer)
        //.isRowVectorShape(DataBuffer), .isRowVectorShape(IntBuffer)
        //Shape.size(DataBuffer,int), Shape.size(IntBuffer,int)
        //Also: Shape.stride(IntBuffer), Shape.stride(DataBuffer)
        //Shape.stride(DataBuffer,int), Shape.stride(IntBuffer,int)

        List<List<Pair<INDArray, String>>> lists = new ArrayList<>();
        lists.add(NDArrayCreationUtil.getAllTestMatricesWithShape(3, 4, 12345));
        lists.add(NDArrayCreationUtil.getAllTestMatricesWithShape(1, 4, 12345));
        lists.add(NDArrayCreationUtil.getAllTestMatricesWithShape(3, 1, 12345));
        lists.add(NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, 3, 4, 5));
        lists.add(NDArrayCreationUtil.getAll4dTestArraysWithShape(12345, 3, 4, 5, 6));
        lists.add(NDArrayCreationUtil.getAll4dTestArraysWithShape(12345, 3, 1, 5, 1));
        lists.add(NDArrayCreationUtil.getAll5dTestArraysWithShape(12345, 3, 4, 5, 6, 7));
        lists.add(NDArrayCreationUtil.getAll6dTestArraysWithShape(12345, 3, 4, 5, 6, 7, 8));

        val shapes = new long[][] {{3, 4}, {1, 4}, {3, 1}, {3, 4, 5}, {3, 4, 5, 6}, {3, 1, 5, 1}, {3, 4, 5, 6, 7},
                        {3, 4, 5, 6, 7, 8}};

        for (int i = 0; i < shapes.length; i++) {
            List<Pair<INDArray, String>> list = lists.get(i);
            val shape = shapes[i];

            for (Pair<INDArray, String> p : list) {
                INDArray arr = p.getFirst();

                assertArrayEquals(shape, arr.shape());

                val thisStride = arr.stride();

                val ib = arr.shapeInfo();
                DataBuffer db = arr.shapeInfoDataBuffer();

                //Check shape calculation
                assertEquals(shape.length, Shape.rank(ib));
                assertEquals(shape.length, Shape.rank(db));

                assertArrayEquals(shape, Shape.shape(ib));
                assertArrayEquals(shape, Shape.shape(db));

                for (int j = 0; j < shape.length; j++) {
                    assertEquals(shape[j], Shape.size(ib, j));
                    assertEquals(shape[j], Shape.size(db, j));

                    assertEquals(thisStride[j], Shape.stride(ib, j));
                    assertEquals(thisStride[j], Shape.stride(db, j));
                }

                //Check base offset
                assertEquals(Shape.offset(ib), Shape.offset(db));

                //Check offset calculation:
                NdIndexIterator iter = new NdIndexIterator(shape);
                while (iter.hasNext()) {
                    val next = iter.next();
                    long offset1 = Shape.getOffset(ib, next);

                    assertEquals(offset1, Shape.getOffset(db, next));

                    switch (shape.length) {
                        case 2:
                            assertEquals(offset1, Shape.getOffset(ib, next[0], next[1]));
                            assertEquals(offset1, Shape.getOffset(db, next[0], next[1]));
                            break;
                        case 3:
                            assertEquals(offset1, Shape.getOffset(ib, next[0], next[1], next[2]));
                            assertEquals(offset1, Shape.getOffset(db, next[0], next[1], next[2]));
                            break;
                        case 4:
                            assertEquals(offset1, Shape.getOffset(ib, next[0], next[1], next[2], next[3]));
                            assertEquals(offset1, Shape.getOffset(db, next[0], next[1], next[2], next[3]));
                            break;
                        case 5:
                        case 6:
                            //No 5 and 6d getOffset overloads
                            break;
                        default:
                            throw new RuntimeException();
                    }
                }
            }
        }
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
