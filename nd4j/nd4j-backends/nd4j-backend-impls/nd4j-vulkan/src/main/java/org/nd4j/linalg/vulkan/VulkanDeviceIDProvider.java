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

package org.nd4j.linalg.vulkan;

import org.nd4j.jita.constant.DeviceIDProvider;

/**
 * DeviceIDProvider for the Vulkan compute backend.
 *
 * <p>Returns the Vulkan device ID bound to the current thread by querying the
 * AffinityManager. This mirrors {@code CudaDeviceIdProvider}, which delegates
 * to {@code AtomicAllocator.getInstance().getDeviceId()} for the same purpose:
 * providing the currently-active device ID to context providers and constant-cache
 * selectors (e.g. ConstantBuffersCache) that need a device key.</p>
 *
 * <p>Vulkan has no pointer-attributes query (no {@code cudaPointerGetAttributes}
 * equivalent), so thread-local affinity is the only reliable source. The pool
 * pointer-registry (ADR-0111 §3) will provide per-pointer attribution at P2/P3;
 * until then thread-local is the correct authoritative answer.</p>
 *
 * <p>Wired via {@code deviceidprovider} key in {@code nd4j-vulkan.properties}.</p>
 */
public class VulkanDeviceIDProvider implements DeviceIDProvider {

    @Override
    public int getDeviceId() {
        return VulkanRuntime.getInstance().currentDevice();
    }
}
