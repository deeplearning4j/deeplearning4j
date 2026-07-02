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

#include <graph/IslandCapturePolicy.h>

// NativeDynamicShapePlan.h defines NativeSlot — pull it in here so we can
// access opTraits_, ident, and isCapturable().  The header guard prevents
// double-inclusion when callers already pulled it in.
#include <graph/NativeDynamicShapePlan.h>
#include <ops/declarable/OpDescriptor.h>   // OpTraits enum values

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace sd {
namespace graph {

// ── Internal helpers ─────────────────────────────────────────────────────────

// Case-insensitive substring search.
static bool containsCI(const std::string& haystack, const char* needle) {
  std::string h = haystack;
  std::transform(h.begin(), h.end(), h.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return h.find(needle) != std::string::npos;
}

// ── IslandCapturePolicy private statics ─────────────────────────────────────

bool IslandCapturePolicy::nameMatchesAttention(const std::string& name) {
  // Match known attention op name patterns (case-insensitive substrings).
  // "attention" covers dot_product_attention, dot_product_attention_v2,
  // flash_attention; "flash" is an additional fallback.
  return containsCI(name, "attention") || containsCI(name, "flash");
}

bool IslandCapturePolicy::nameMatchesDynamicIndex(const std::string& name) {
  // gather/scatter/dynamic — dynamic-index ops whose replay behaviour depends
  // on the runtime content of an index buffer.
  return containsCI(name, "gather")  ||
         containsCI(name, "scatter") ||
         containsCI(name, "dynamic");
}

bool IslandCapturePolicy::isExcluded(const NativeSlot* slots, int i,
                                     const IslandCaptureProfile& profile) {
  const NativeSlot& slot = slots[i];
  const uint32_t traits = slot.opTraits_;
  const std::string& opName = slot.ident.opName;

  // ── Attention ─────────────────────────────────────────────────────────────
  if (profile.excludeAttention) {
    // Prefer the trait bit (OP_TRAIT_ATTENTION = 1 << 17) set by OpTraitTable.
    const bool traitAttention = (traits & sd::ops::OP_TRAIT_ATTENTION) != 0;
    if (traitAttention || nameMatchesAttention(opName)) return true;
  }

  // ── Host-callback / dynamic-output-size ──────────────────────────────────
  if (profile.excludeHostCallback) {
    // OP_TRAIT_EXTERNAL_WORKSPACE: op uses external library workspace (cuBLAS,
    // cuDNN, etc.) that performs host calls during GPU graph capture.
    // OP_TRAIT_DYNAMIC_OUTPUT_SIZE: output size determined by runtime data —
    // requires host sync, which invalidates a capture stream.
    const bool externalWs = (traits & sd::ops::OP_TRAIT_EXTERNAL_WORKSPACE) != 0;
    const bool dynamicOut = (traits & sd::ops::OP_TRAIT_DYNAMIC_OUTPUT_SIZE) != 0;
    if (externalWs || dynamicOut) return true;
  }

  // ── Dynamic-index ops ────────────────────────────────────────────────────
  if (profile.excludeDynamicIndex) {
    // Prefer trait bits for gather/scatter families.
    const bool gatherTrait     = (traits & sd::ops::OP_TRAIT_GATHER)            != 0;
    const bool gatherNdTrait   = (traits & sd::ops::OP_TRAIT_GATHER_ND)         != 0;
    const bool scatterTrait    = (traits & sd::ops::OP_TRAIT_SCATTER_ND)        != 0;
    const bool scatterUpdTrait = (traits & sd::ops::OP_TRAIT_SCATTER_ND_UPDATE) != 0;
    if (gatherTrait || gatherNdTrait || scatterTrait || scatterUpdTrait ||
        nameMatchesDynamicIndex(opName)) {
      return true;
    }
  }

  return false;
}

// ── IslandCapturePolicy::partition ──────────────────────────────────────────

std::vector<IslandRange> IslandCapturePolicy::partition(
    const NativeSlot* slots, int start, int end,
    const IslandCaptureProfile& profile) {

  std::vector<IslandRange> result;

  if (start >= end || slots == nullptr) return result;

  // islandStart: index where the current in-progress island begins (-1 = none).
  int islandStart = -1;
  int islandNodeCount = 0;

  auto flushIsland = [&](int flushEnd) {
    if (islandStart >= 0 && flushEnd > islandStart) {
      result.emplace_back(islandStart, flushEnd, /*capture=*/true);
    }
    islandStart = -1;
    islandNodeCount = 0;
  };

  for (int i = start; i < end; ++i) {
    const NativeSlot& slot = slots[i];

    // ── Check if this slot is capturable at all (non-negotiable) ─────────
    // isCapturable() returns false for:
    //   - control-flow nodes (CF_NONE check fails)
    //   - hasDynamicOutputSize() — always a gap regardless of profile
    const bool capturable = slot.isCapturable(/*mergeViews=*/true);

    // ── Check profile-based exclusion ─────────────────────────────────────
    const bool excluded = capturable && isExcluded(slots, i, profile);

    if (!capturable || excluded) {
      // Close any open island before this gap slot.
      flushIsland(i);

      // Emit the excluded/non-capturable slot as its own eager gap.
      result.emplace_back(i, i + 1, /*capture=*/false);

      // No island is open after a gap; the next slot starts fresh.
      continue;
    }

    // ── Slot is capturable and not excluded ───────────────────────────────

    // If adding this slot would exceed the node budget, close the current
    // island first (silent split — no gap emitted).
    if (islandStart >= 0 && islandNodeCount >= profile.maxIslandNodes) {
      flushIsland(i);
    }

    // Open a new island if none is in progress.
    if (islandStart < 0) {
      islandStart = i;
      islandNodeCount = 0;
    }

    islandNodeCount++;
  }

  // Close any trailing open island.
  flushIsland(end);

  return result;
}

}  // namespace graph
}  // namespace sd
