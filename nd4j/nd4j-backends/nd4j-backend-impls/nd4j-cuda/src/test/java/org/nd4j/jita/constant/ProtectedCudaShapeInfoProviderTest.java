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

package org.nd4j.jita.constant;

import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.ShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * Created by raver119 on 30.09.16.
 */
@Slf4j
public class ProtectedCudaShapeInfoProviderTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testPurge1() throws Exception {
        INDArray array = Nd4j.create(10, 10);

        ProtectedCudaShapeInfoProvider provider = (ProtectedCudaShapeInfoProvider) ProtectedCudaShapeInfoProvider.getInstance();

        assertEquals(true, provider.protector.containsDataBuffer(0, new ShapeDescriptor(array.shape(), array.stride(),0, array.elementWiseStride(), array.ordering())));

        Nd4j.getMemoryManager().purgeCaches();

        assertEquals(false, provider.protector.containsDataBuffer(0, new ShapeDescriptor(array.shape(), array.stride(),0, array.elementWiseStride(), array.ordering())));

    //    INDArray array2 = Nd4j.create(10, 10);
    }


    @Test
    public void testPurge2() throws Exception {
        INDArray arrayA = Nd4j.create(10, 10);

        DataBuffer shapeInfoA = arrayA.shapeInfoDataBuffer();

        INDArray arrayE = Nd4j.create(10, 10);

        DataBuffer shapeInfoE = arrayE.shapeInfoDataBuffer();

        int[] arrayShapeA = shapeInfoA.asInt();

        assertTrue(shapeInfoA == shapeInfoE);

        ShapeDescriptor descriptor = new ShapeDescriptor(arrayA.shape(), arrayA.stride(), 0, arrayA.elementWiseStride(), arrayA.ordering());
        ConstantProtector protector = ConstantProtector.getInstance();
        AllocationPoint pointA = AtomicAllocator.getInstance().getAllocationPoint(arrayA.shapeInfoDataBuffer());

        assertEquals(true, protector.containsDataBuffer(0, descriptor));

////////////////////////////////////

        Nd4j.getMemoryManager().purgeCaches();

////////////////////////////////////


        assertEquals(false, protector.containsDataBuffer(0, descriptor));

        INDArray arrayB = Nd4j.create(10, 10);

        DataBuffer shapeInfoB = arrayB.shapeInfoDataBuffer();

        assertFalse(shapeInfoA == shapeInfoB);

        AllocationPoint pointB = AtomicAllocator.getInstance().getAllocationPoint(arrayB.shapeInfoDataBuffer());


        assertArrayEquals(arrayShapeA, shapeInfoB.asInt());

        // pointers should be equal, due to offsets reset
        assertEquals(pointA.getPointers().getDevicePointer().address(), pointB.getPointers().getDevicePointer().address());
    }



    @Test
    public void testPurge3() throws Exception {
        INDArray arrayA = Nd4j.create(10, 10);

        DataBuffer shapeInfoA = arrayA.shapeInfoDataBuffer();
        int[] shapeA = shapeInfoA.asInt();
        log.info("ShapeA: {}", shapeA);


        Nd4j.getMemoryManager().purgeCaches();

        INDArray arrayB = Nd4j.create(20, 20);

        DataBuffer shapeInfoB = arrayB.shapeInfoDataBuffer();

        int[] shapeB = shapeInfoB.asInt();
        log.info("ShapeB: {}", shapeB);
    }

}