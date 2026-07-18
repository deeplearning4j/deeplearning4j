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

package org.nd4j.presets.vulkan;

import org.nd4j.nativeblas.NativeOps;

/**
 * Helper base class for the JavaCPP-generated {@code Nd4jVulkan} bindings.
 *
 * <p>The generated {@code org.nd4j.linalg.vulkan.bindings.Nd4jVulkan}
 * class extends this helper and supplies the shared {@link NativeOps} ABI,
 * matching the CUDA helper boundary without host-BLAS initialization.</p>
 */
public abstract class Nd4jVulkanHelper extends Nd4jVulkanPresets implements NativeOps {

    private static UnsupportedOperationException hostBlasUnsupported() {
        return new UnsupportedOperationException("OpenBLAS controls are not available on the Vulkan backend");
    }

    @Override
    public final void setOpenBlasThreads(int threads) {
        throw hostBlasUnsupported();
    }

    @Override
    public final int getOpenBlasThreads() {
        throw hostBlasUnsupported();
    }

    @Override
    public final boolean isSerializeBlasCalls() {
        throw hostBlasUnsupported();
    }

    @Override
    public final void setSerializeBlasCalls(boolean serialize) {
        throw hostBlasUnsupported();
    }
}
