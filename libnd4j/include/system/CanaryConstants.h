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

#ifndef LIBND4J_CANARY_CONSTANTS_H
#define LIBND4J_CANARY_CONSTANTS_H

#include <cstdint>
#include <system/common.h>

namespace sd {

/**
 * Sentinel values written into padding regions to detect buffer overruns.
 * Used by DataBuffer (host allocations) and shape buffer creators.
 */
struct CanaryConstants {
    /** Written into HOST_ALLOC_PADDING after data buffers. */
    static constexpr uint64_t DATA_BUFFER_CANARY = 0xDEADBEEFCAFEBABEULL;

    /** Written into SD_SHAPE_ALLOC_PADDING after shape info buffers. */
    static constexpr LongType SHAPE_BUFFER_CANARY = static_cast<LongType>(0x5AFE5AFE5AFE5AFELL);
};

}  // namespace sd

#endif  // LIBND4J_CANARY_CONSTANTS_H
