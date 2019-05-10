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

package org.nd4j.jita.allocator.garbage;

import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * This class enables Thread tracking via DeallocatorService
 * @author raver119@gmail.com
 */
public class DeallocatableThread implements Deallocatable {
    private long threadId;
    private CudaContext context;

    public DeallocatableThread(Thread thread, CudaContext context) {
        this.threadId = thread.getId();
        this.context = context;
    }

    @Override
    public String getUniqueId() {
        return "thread_" +  threadId;
    }

    @Override
    public Deallocator deallocator() {
        return new ContextDeallocator(context);
    }

    @Override
    public int targetDevice() {
        return context.getDeviceId();
    }
}
