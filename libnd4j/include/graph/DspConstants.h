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

#ifndef LIBND4J_DSP_CONSTANTS_H
#define LIBND4J_DSP_CONSTANTS_H

#include <cstddef>
#include <cstdint>

namespace sd {
namespace graph {
namespace dsp {

// ─── FNV-1a 64-bit hash constants ───────────────────────────────────────────
constexpr uint64_t FNV1A64_OFFSET_BASIS = 0xcbf29ce484222325ULL;
constexpr uint64_t FNV1A64_PRIME        = 0x100000001b3ULL;

// ─── Shape chain folding thresholds ─────────────────────────────────────────
constexpr int SHAPE_CHAIN_MAX_ELEMENTS = 64;
constexpr int SHAPE_CHAIN_MAX_RANK     = 8;

// ─── Environment variable names ─────────────────────────────────────────────
// Used for platform-independent path resolution. Actual DSP config comes from
// Environment::getInstance().dsp() — these are only for OS-level paths.
constexpr const char* ENV_HOME        = "HOME";
constexpr const char* ENV_USERPROFILE = "USERPROFILE";
constexpr const char* ENV_TMPDIR      = "TMPDIR";
constexpr const char* ENV_TMP         = "TMP";
constexpr const char* ENV_TEMP        = "TEMP";

}  // namespace dsp
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_CONSTANTS_H
