/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// INTENTIONALLY NO DEFINITIONS.
//
// The graph-baked-address-pinning CPU stubs (platformPinGraphBakedAddress /
// platformFlushGraphBakedPins) were originally placed here, but this file lives
// under graph/impl/ and is therefore compiled in BOTH the CPU and CUDA builds.
// In the CUDA build that collided (multiple definition at link time) with the
// real implementations in graph/impl/NativeDynamicShapePlan_cuda.cu, which nvcc
// also compiles. The CPU stubs now live in their canonical home alongside the
// other platform-dispatch CPU stubs:
//     graph/cpu/NativeDynamicShapePlan_cuda_stubs.cpp   (CPU-only build)
//     graph/impl/NativeDynamicShapePlan_cuda.cu         (CUDA-only build)
//
// This file is left without definitions (a valid empty translation unit) so the
// existing CMake source glob need not be refreshed; it can be removed entirely
// on the next full CMake re-configure.
