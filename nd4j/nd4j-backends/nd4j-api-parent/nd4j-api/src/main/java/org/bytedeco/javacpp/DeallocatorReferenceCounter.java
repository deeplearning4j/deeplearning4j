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
package org.bytedeco.javacpp;

import org.nd4j.common.io.ReflectionUtils;
import org.nd4j.common.primitives.Counter;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.DataBufferDouble;
import java.lang.reflect.Field;
import java.sql.Ref;
import java.util.Iterator;

public class DeallocatorReferenceCounter {

    private static Field referentField;

    static {
        try {
            referentField = ReflectionUtils.findField(Pointer.NativeDeallocator.class, "referent");
            ReflectionUtils.makeAccessible(referentField);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }
    public DeallocatorReferenceCounter() {

    }


    public Counter<String> countAllocations() {
        Pointer.DeallocatorReference head = Pointer.DeallocatorReference.head;
        Counter<String> allocationCounts = new Counter<>();
        while(head != null && head.next != null) {
            Pointer pointer = (Pointer) ReflectionUtils.getField(referentField, head.deallocator);
            if(pointer != null)
                allocationCounts.incrementCount(pointer.getClass().getName(),1);
            head = head.next;
        }

        Iterator<PointerScope> scopeIterator = PointerScope.getScopeIterator();
        while(scopeIterator.hasNext()) {
            PointerScope scope = scopeIterator.next();
            for(Pointer pointer : scope.pointerStack) {
                allocationCounts.incrementCount("Pointer stack " + pointer.getClass().getName(),1);
            }
        }

        Nd4j.getDeallocatorService().getReferenceMap().values().forEach(deallocatableReference -> {
            Object pointer = ReflectionUtils.getField(referentField, deallocatableReference);
            if(pointer != null) {
                if(pointer instanceof DataBuffer) {
                    DataBuffer dataBuffer = (DataBuffer) pointer;
                    allocationCounts.incrementCount("Deallocator reference " + dataBuffer.getClass().getName(), dataBuffer.pointer().referenceCount());

                } else {
                    allocationCounts.incrementCount("Deallocator reference " + pointer.getClass().getName(), 1);
                }
            }
        });


        return allocationCounts;

    }
}


