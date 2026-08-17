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

#ifndef LIBND4J_LEGACYOPTYPECODES_H
#define LIBND4J_LEGACYOPTYPECODES_H

namespace sd {
namespace graph {

// Wire-protocol numeric codes for NativeSlot.legacy.legacyOpType.
// These MUST match the values the Java-side DSP compiler writes. See
// NativeDynamicShapePlan.cpp's legacy-op switch for the one-to-one mapping
// to the LegacyXxxOp C++ classes.
//
// Categorization notes relevant to backend routing:
//   - LEGACY_NOT_SET (0): slot is not a legacy op wrapper.
//   - LEGACY_TRANSFORM_SAME..LEGACY_SCALAR_BOOL (1..7),
//     LEGACY_PAIRWISE_BOOL (18), and LEGACY_TRANSFORM_ANY (19):
//     element-wise ops; Triton-compilable.
//   - LEGACY_REDUCE_FLOAT..LEGACY_REDUCE3 (8..12), LEGACY_INDEX_REDUCE (14),
//     LEGACY_BROADCAST (15): read reduction/broadcast dims from block.getAxis();
//     wrap NativeOpExecutioner C-APIs with no Triton IR emitter — must route
//     to SLOT_BY_SLOT.
//   - LEGACY_STATS (13), LEGACY_BROADCAST_BOOL (16): read dims from iArgs
//     directly; currently treated as capturable.
//   - LEGACY_RANDOM (17): RNG ops.
enum LegacyOpTypeCode : int {
  LEGACY_NOT_SET              = 0,
  LEGACY_TRANSFORM_SAME       = 1,
  LEGACY_TRANSFORM_STRICT     = 2,
  LEGACY_TRANSFORM_FLOAT      = 3,
  LEGACY_TRANSFORM_BOOL       = 4,
  LEGACY_SCALAR               = 5,
  LEGACY_PAIRWISE_TRANSFORM   = 6,
  LEGACY_SCALAR_BOOL          = 7,
  LEGACY_REDUCE_FLOAT         = 8,
  LEGACY_REDUCE_SAME          = 9,
  LEGACY_REDUCE_BOOL          = 10,
  LEGACY_REDUCE_LONG          = 11,
  LEGACY_REDUCE3              = 12,
  LEGACY_STATS                = 13,
  LEGACY_INDEX_REDUCE         = 14,
  LEGACY_BROADCAST            = 15,
  LEGACY_BROADCAST_BOOL       = 16,
  LEGACY_RANDOM               = 17,
  LEGACY_PAIRWISE_BOOL        = 18,
  LEGACY_TRANSFORM_ANY        = 19,
  // Private Vulkan eager-execution identities. The Java DSP wire protocol
  // does not allocate separate integer family codes, but native Vulkan
  // legacy entry points preserve these families explicitly. Negative values
  // keep them outside the stable Java wire range while reusing NativeSlot's
  // family lookup path.
  LEGACY_BROADCAST_INT        = -1,
  LEGACY_PAIRWISE_INT         = -2,
  LEGACY_SCALAR_INT           = -3,
};

// True for legacy op types whose reduction/broadcast dimensions live in
// block.getAxis() rather than block.getIArguments(). The DSP Java compiler
// packs dims into iArgs, so these types need axis mirrored from iArgs before
// validateAndExecute is called.
inline constexpr bool legacyOpReadsAxisFromIArgs(int code) noexcept {
  return (code >= LEGACY_REDUCE_FLOAT && code <= LEGACY_REDUCE3) ||
         code == LEGACY_INDEX_REDUCE ||
         code == LEGACY_BROADCAST;
}

// True for legacy op types that wrap NativeOpExecutioner C-APIs which have no
// Triton IR emitter. These must route to SLOT_BY_SLOT at segment-build time so
// they execute via the native kernel path rather than failing GPU compile.
inline constexpr bool legacyOpRequiresSlotBySlot(int code) noexcept {
  return legacyOpReadsAxisFromIArgs(code);
}

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_LEGACYOPTYPECODES_H
