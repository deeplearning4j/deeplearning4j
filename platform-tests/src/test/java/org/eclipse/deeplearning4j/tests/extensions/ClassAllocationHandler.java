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
package org.eclipse.deeplearning4j.tests.extensions;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;

import java.util.List;
import java.util.Map;

/**
 * A class allocation handler is a callback that is invoked
 * when a {@link DeallocatableReference}
 * is deallocated.
 *
 * @author Adam Gibson
 */
public interface ClassAllocationHandler {

    /**
     * Clear the accumulated references
     * in the {@link #passedReferences()}
     *  map
     */
    void clearReferences();
    /**
     * The set of passed references for the specific handler.
     * When a test name is not set, a custom handler
     * can be registered which will capture allocations
     * before a test is set. This is common when dealing with
     * test setup and parameterized tests that do some sort
     * of preloading.
     * @return
     */
    Map<String, List<DeallocatableReference>> passedReferences();



    /**
     * Clear the accumulated references
     * in the {@link #passedReferences()}
     *  map
     */
    void clearDataBuffers();
    /**
     * The set of passed references for the specific handler.
     * When a test name is not set, a custom handler
     * can be registered which will capture allocations
     * before a test is set. This is common when dealing with
     * test setup and parameterized tests that do some sort
     * of preloading.
     * @return
     */
    Map<String, List<DataBuffer>> passedDataBuffers();


    /**
     * Handles {@link DeallocatableReference}
     * deallocation in the context of a specific class.
     * This can be needed when parameters or allocations are loaded
     * before a specific method is instantiated.
     * @param reference
     */
    void handleDeallocatableReference(DeallocatableReference reference);

    /**
     * Handles data buffer deallocation.
     * @param dataBuffer
     */
    void handleDataBuffer(DataBuffer dataBuffer);

}
