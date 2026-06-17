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

package org.nd4j.autodiff.samediff.config;

import lombok.NonNull;
import org.nd4j.linalg.api.ops.executioner.KernelManager.Engine;
import org.nd4j.linalg.api.ops.executioner.KernelManager.OpKernelInfo;

/**
 * Configuration for a category of operations.
 * Returned by the {@code forXxx()} methods on {@link KernelConfiguration}.
 */
public class CategoryConfiguration {

    private final KernelConfiguration parent;
    private final OperationCategory category;

    CategoryConfiguration(KernelConfiguration parent, OperationCategory category) {
        this.parent = parent;
        this.category = category;
    }

    public CategoryConfiguration useCpu() {
        return useEngine(Engine.CPU);
    }

    public CategoryConfiguration useCuda() {
        return useEngine(Engine.CUDA);
    }

    public CategoryConfiguration useCudnn() {
        return useEngine(Engine.CUDNN);
    }

    public CategoryConfiguration useOneDnn() {
        return useEngine(Engine.ONEDNN);
    }

    public CategoryConfiguration useMps() {
        return useEngine(Engine.MPS);
    }

    public CategoryConfiguration useAccelerate() {
        return useEngine(Engine.ACCELERATE);
    }

    public CategoryConfiguration useEngine(@NonNull Engine engine) {
        parent.addPendingChange(() -> {
            for (String pattern : category.getPatterns()) {
                parent.getKernelManager().disableEngineForPattern(engine, pattern);
                for (OpKernelInfo op : parent.getKernelManager().searchOperations(pattern)) {
                    parent.getKernelManager().setPreferredEngine(op.getOpName(), engine);
                }
            }
        });
        return this;
    }

    public CategoryConfiguration disable(@NonNull Engine engine) {
        parent.addPendingChange(() -> {
            for (String pattern : category.getPatterns()) {
                for (OpKernelInfo op : parent.getKernelManager().searchOperations(pattern)) {
                    parent.getKernelManager().disableKernel(op.getOpName(), engine);
                }
            }
        });
        return this;
    }

    public KernelConfiguration and() {
        return parent;
    }
}
