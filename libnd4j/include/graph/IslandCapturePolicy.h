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

#ifndef LIBND4J_ISLAND_CAPTURE_POLICY_H
#define LIBND4J_ISLAND_CAPTURE_POLICY_H

// system/common.h provides SD_LIB_EXPORT and platform typedefs.
// In standalone test mode (SD_HIP_POLICY_TEST_STANDALONE), the caller
// provides equivalent definitions before including this header so that
// the full libnd4j dependency tree is not required.
#ifndef SD_HIP_POLICY_TEST_STANDALONE
#include <system/common.h>
#endif

#include <string>
#include <vector>

// Forward declaration — avoid pulling in the full NativeDynamicShapePlan.h.
// IslandCapturePolicy only needs opTraits_ (uint32_t) and ident.opName (string).
namespace sd {
namespace graph {
struct NativeSlot;
}  // namespace graph
}  // namespace sd

namespace sd {
namespace graph {

/**
 * IslandRange — one contiguous run of slots with a capturable/gap annotation.
 *
 *   begin  — inclusive slot index (into the parent slot array)
 *   end    — exclusive slot index
 *   capture — true  → the run is safe to capture into a graph island
 *             false → the run must execute eagerly ("gap")
 */
struct IslandRange {
  int begin;   // inclusive
  int end;     // exclusive
  bool capture;

  IslandRange(int b, int e, bool cap) : begin(b), end(e), capture(cap) {}
};

/**
 * IslandCaptureProfile — tuning knobs for island/gap partitioning.
 *
 * Two static factory methods provide calibrated defaults:
 *   - forCuda()  : permissive (large islands, only strict excludes)
 *   - forRocm()  : conservative (small islands ≤128 nodes; attention,
 *                   host-callback, and dynamic-index ops forced into gaps).
 *
 * The conservative ROCm profile mirrors vLLM's PIECEWISE hipGraph strategy:
 * smaller islands avoid the ~200-node stale-pointer threshold seen in early
 * ROCm graph implementations and ensure attention/gather ops—which are the
 * most common source of replay stalls—run eagerly where they are safe.
 */
struct IslandCaptureProfile {
  /**
   * Maximum number of slots (nodes) allowed in a single captured island.
   * When a run would exceed this count the island is closed and a new one
   * is started immediately after (no gap emitted—the split is transparent).
   */
  int maxIslandNodes;

  /**
   * When true, any slot classified as an attention op (ATTENTION trait or
   * name matching "attention"/"dot_product_attention"/"flash") is forced into
   * its own capture=false gap.  Islands are opened fresh after the gap.
   */
  bool excludeAttention;

  /**
   * When true, any slot classified as a host-callback (EXTERNAL_WORKSPACE
   * trait or hasDynamicOutputSize()) is forced into its own gap.
   */
  bool excludeHostCallback;

  /**
   * When true, any slot classified as a dynamic-index op (GATHER, GATHER_ND,
   * SCATTER_ND, SCATTER_ND_UPDATE, or name matching "gather"/"scatter"/
   * "dynamic") is forced into its own gap.
   */
  bool excludeDynamicIndex;

  /**
   * Byte budget hint: if the cumulative output buffer footprint of the current
   * island exceeds this value the island is closed and a new one started.
   * Zero means "no byte limit".  Only an approximation—backends may ignore it.
   */
  size_t maxIslandBytesHint;

  // ── Static factory profiles ───────────────────────────────────────────────

  /**
   * CUDA profile — permissive.
   * Matches today's existing CUDA graph capture behaviour: effectively no node
   * limit, attention/gather ops are capturable, byte hint is the existing
   * per-island safety limit (~128 MiB).
   */
  static IslandCaptureProfile forCuda() {
    IslandCaptureProfile p;
    p.maxIslandNodes       = 1 << 20;  // effectively unlimited
    p.excludeAttention     = false;
    p.excludeHostCallback  = false;
    p.excludeDynamicIndex  = false;
    p.maxIslandBytesHint   = 128ULL * 1024 * 1024;  // 128 MiB
    return p;
  }

  /**
   * ROCm/HIP profile — conservative.
   *
   * Key design choices (matching vLLM's PIECEWISE hipGraph pattern):
   *   - maxIslandNodes=128: keeps islands below the ~200-node stale-pointer
   *     threshold observed in AMD ROCm hipGraph implementations.
   *   - excludeAttention=true: FlashAttention / fused-GQA kernels on ROCm
   *     have historically misfired during graph replay; running them eagerly
   *     is the safe fallback that vLLM and TRT-LLM also adopt.
   *   - excludeHostCallback=true: ops that need host synchronisation to
   *     determine their output size cannot be captured in hipGraph streams.
   *   - excludeDynamicIndex=true: gather/scatter ops with runtime indices
   *     produce address-dependent launch parameters that break graph replay
   *     when the index buffer contents change between steps.
   *   - maxIslandBytesHint=64 MiB: conservative budget to limit per-island
   *     memory pinning overhead on unified-memory AMD GPUs.
   */
  static IslandCaptureProfile forRocm() {
    IslandCaptureProfile p;
    p.maxIslandNodes       = 128;
    p.excludeAttention     = true;
    p.excludeHostCallback  = true;
    p.excludeDynamicIndex  = true;
    p.maxIslandBytesHint   = 64ULL * 1024 * 1024;  // 64 MiB
    return p;
  }

  /**
   * Apple Metal / MLX profile — conservative island size, but attention STAYS
   * captured (the key difference from ROCm).
   *
   * Islands are realized as MTLIndirectCommandBuffer (ICB) recordings and
   * replayed by MetalReplayHandle; gaps run eagerly via the MPS per-op helpers
   * (ops/declarable/platform/mps/*.mm). Design choices:
   *   - maxIslandNodes=64: an ICB has a fixed maxCommandCount (256 default in
   *     MetalReplayHandle) and each slot consumes one command slot; 64 keeps
   *     comfortable headroom and bounds re-record cost.
   *   - excludeAttention=FALSE: unlike ROCm, MLX fuses attention as a first-class
   *     op (mx::fast::scaled_dot_product_attention), so it captures cleanly into
   *     the MLX island graph — no reason to force it eager.
   *   - excludeHostCallback=true: ICB cannot host-readback between replays.
   *   - excludeDynamicIndex=true: gather/scatter/dynamic-shape produce
   *     address-dependent launch params; an ICB bakes MTLBuffer addresses at
   *     record time and cannot mutate them per replay — so these run eager.
   *   - maxIslandBytesHint=64 MiB: unified-memory budget hint.
   * Pointer stability (ICB bakes buffer addresses) is enforced separately by the
   * DSP argTable-stable check, which re-records the ICB when addresses drift.
   */
  static IslandCaptureProfile forMetal() {
    IslandCaptureProfile p;
    p.maxIslandNodes       = 64;
    p.excludeAttention     = false;  // MLX fuses attention — keep it in the island
    p.excludeHostCallback  = true;
    p.excludeDynamicIndex  = true;
    p.maxIslandBytesHint   = 64ULL * 1024 * 1024;  // 64 MiB
    return p;
  }
};

/**
 * IslandCapturePolicy — backend-neutral island/gap partitioner.
 *
 * Given a NativeSlot array and a half-open range [start, end), returns an
 * ordered vector of IslandRange covering exactly [start, end) with no gaps
 * and no overlaps.  Ranges with capture=true are safe to hand to a
 * hipGraph/cudaGraph capture call; ranges with capture=false must be executed
 * eagerly by the caller.
 *
 * This class has NO dependency on CUDA or HIP headers—it inspects only the
 * opTraits_ bitmask and the ident.opName string on each NativeSlot.
 */
class SD_LIB_EXPORT IslandCapturePolicy {
 public:
  /**
   * Partition [start, end) into capture islands and eager gaps.
   *
   * The returned vector covers [start, end) exactly, in order, with no
   * overlapping ranges.  The first range always starts at `start`; the last
   * ends at `end`.
   *
   * Splitting rules (applied left-to-right, highest priority first):
   *   1. If adding the next slot would exceed profile.maxIslandNodes, close
   *      the current island and start a fresh one (no gap—split is silent).
   *   2. If a slot is classified as excluded (attention / host-callback /
   *      dynamic-index, per the profile flags), close the current island,
   *      emit the excluded slot as its own capture=false gap, then start a
   *      new island for the slot after it.
   *   3. If a slot is not capturable (isCapturable() == false: control-flow
   *      node, dynamic-output-size), treat it the same as an excluded slot.
   *
   * Empty ranges (begin == end) are never emitted.
   *
   * @param slots    Pointer to the flat slot array (not owned; must outlive
   *                 the returned ranges).
   * @param start    First slot index to consider (inclusive).
   * @param end      One past the last slot index to consider (exclusive).
   * @param profile  Tuning profile (use forCuda() or forRocm()).
   * @return Ordered, non-overlapping ranges covering [start, end).
   */
  static std::vector<IslandRange> partition(const NativeSlot* slots,
                                            int start, int end,
                                            const IslandCaptureProfile& profile);

 private:
  /**
   * Returns true if the slot at index `i` should be forced into an eager gap
   * under the given profile.  Does NOT check isCapturable()—that is checked
   * separately in partition().
   */
  static bool isExcluded(const NativeSlot* slots, int i,
                         const IslandCaptureProfile& profile);

  /**
   * Classify an op name via substring matching as a fallback when the trait
   * bitmask does not carry the needed information.
   */
  static bool nameMatchesAttention(const std::string& name);
  static bool nameMatchesDynamicIndex(const std::string& name);
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_ISLAND_CAPTURE_POLICY_H
