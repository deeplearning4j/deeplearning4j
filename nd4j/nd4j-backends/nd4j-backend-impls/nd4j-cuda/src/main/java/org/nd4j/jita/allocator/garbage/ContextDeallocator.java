/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;

/**
 * This class provides Deallocator implementation for tracking/releasing CudaContexts once thread holding it dies
 * @author raver119@gmail.com
 */
@Slf4j
public class ContextDeallocator implements Deallocator {
    private CudaContext context;

    public ContextDeallocator(@NonNull CudaContext context) {
        this.context = context;
    }

    @Override
    public void deallocate() {
        AtomicAllocator.getInstance().getContextPool().releaseContext(context);
    }
}
