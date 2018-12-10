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

package org.nd4j.jita.memory.impl;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.linalg.api.buffer.DataBuffer;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CudaDirectProviderTest {

    @Test
    public void mallocHost() throws Exception {
        CudaDirectProvider provider = new CudaDirectProvider();

        AllocationShape shape = new AllocationShape(100000, 4, DataType.FLOAT);
        AllocationPoint point = new AllocationPoint();
        point.setShape(shape);



        point.setPointers(provider.malloc(shape, point, AllocationStatus.HOST));

        System.out.println("Allocated...");
        Thread.sleep(1000);


        provider.free(point);

        System.out.println("Deallocated...");
        Thread.sleep(1000);
    }

    @Test
    public void mallocDevice() throws Exception {
        CudaDirectProvider provider = new CudaDirectProvider();

        AllocationShape shape = new AllocationShape(300000, 4, DataType.FLOAT);
        AllocationPoint point = new AllocationPoint();
        point.setShape(shape);


        point.setPointers(provider.malloc(shape, point, AllocationStatus.DEVICE));

        System.out.println("Allocated...");
        Thread.sleep(1000);


        point.setAllocationStatus(AllocationStatus.DEVICE);

        provider.free(point);

        System.out.println("Deallocated...");
        Thread.sleep(1000);
    }

}