/* ******************************************************************************
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

#ifndef LIBND4J_OP_TRAIT_TABLE_H
#define LIBND4J_OP_TRAIT_TABLE_H

#include <string>
#include <system/common.h>
#include <ops/declarable/OpDescriptor.h>

namespace sd {
namespace ops {

/**
 * Initialize the op trait table. Safe to call multiple times (idempotent).
 * Registers trait flags on all known ops via OpRegistrator.
 */
SD_LIB_EXPORT void initOpTraits();

/**
 * Look up op trait flags by op name. Returns 0 if the op is not found.
 * Uses the same internal trait table as initOpTraits().
 */
SD_LIB_EXPORT uint32_t getOpTraitsByName(const std::string& opName);

/**
 * Get the number of structural iArgs for an op (args that affect output shape
 * but not the computation itself, like axis for concat).
 * Returns -1 if the op has no structural iArgs entry.
 */
SD_LIB_EXPORT int getStructuralIArgCount(const std::string& opName);

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_OP_TRAIT_TABLE_H
