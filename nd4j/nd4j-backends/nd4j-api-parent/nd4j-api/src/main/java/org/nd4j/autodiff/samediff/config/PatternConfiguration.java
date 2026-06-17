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
 * Configuration for operations matching a wildcard pattern.
 * Returned by {@link KernelConfiguration#forPattern(String)}.
 */
public class PatternConfiguration {

    private final KernelConfiguration parent;
    private final String pattern;

    PatternConfiguration(KernelConfiguration parent, String pattern) {
        this.parent = parent;
        this.pattern = pattern;
    }

    public PatternConfiguration useEngine(@NonNull Engine engine) {
        parent.addPendingChange(() -> {
            for (OpKernelInfo op : parent.getKernelManager().searchOperations(pattern)) {
                parent.getKernelManager().setPreferredEngine(op.getOpName(), engine);
            }
        });
        return this;
    }

    public PatternConfiguration disableEngine(@NonNull Engine engine) {
        parent.addPendingChange(() -> parent.getKernelManager().disableEngineForPattern(engine, pattern));
        return this;
    }

    public KernelConfiguration and() {
        return parent;
    }
}
