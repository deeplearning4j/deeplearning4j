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

#ifndef LIBND4J_SECTION_TYPE_CONFIG_H
#define LIBND4J_SECTION_TYPE_CONFIG_H

#include <graph/gpu/TritonIRBuilder.h>

#include <algorithm>
#include <unordered_set>

namespace sd {
namespace graph {

// ── Grid type for section launch configuration ──
enum class SectionGridType {
  LINEAR_1D,    // ceil(N / BLOCK_SIZE) blocks
  TILED_2D,     // gridM * gridN blocks (matmul)
  ATTENTION,    // batchHeads * gridQ blocks (flash attention)
  CUSTOM        // section-specific (convolution)
};

// ── Shared memory strategy ──
enum class SectionSharedMemStrategy {
  NONE,            // Pure register/global memory ops
  TILED_MATMUL,    // A[BM,BK] + B[BK,BN] tiles, multi-buffered
  ATTENTION,       // Q+K+V tiles via chooseFusedAttentionTileConfig
  TREE_REDUCTION   // BLOCK_SIZE * sizeof(float) scratch
};

// ── Per-section-type configuration ──
// All behavioral properties for a KernelSectionType in one place.
struct SectionTypeConfig {
  KernelSectionType type;
  const char* name;

  // Compilation strategy
  bool compiledByDefault;   // Compiled without compileAll (ELEMENTWISE, IDENTITY)
  bool alwaysFallback;      // Always native CUDA (SHAPE_MANIPULATION)
  bool alwaysStandalone;    // Always own kernel (CONVOLUTION)
  bool fusionVerified;      // Safe to merge into multi-section mega-kernels

  // Multi-section kernel behavior
  bool needsGlobalBarrier;  // Cross-block sync when consuming cross-section intermediates

  // Grid and shared memory
  SectionGridType gridType;
  SectionSharedMemStrategy sharedMemStrategy;
};

// ── Static configuration table (19 rows, one per section type) ──
// Order matches KernelSectionType enum for O(1) lookup via static_cast<int>.
//
// fusionVerified = true for GATHER, GATHER_ND, STACK (tested 65% diversity in merged kernels).
// SPLIT, CONCAT, CONST_GEN have subtle accuracy regressions when merged — standalone until proven.
inline constexpr SectionTypeConfig SECTION_TYPE_CONFIGS[] = {
  //  type                            name              compiled fallback standalone fusionOK barrier grid                       sharedMem
  { KernelSectionType::ELEMENTWISE,        "ELEMENTWISE",    true,  false, false, true,  false, SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::MATMUL,             "MATMUL",         false, false, false, false, false, SectionGridType::TILED_2D,   SectionSharedMemStrategy::TILED_MATMUL },
  { KernelSectionType::FUSED_ATTENTION,    "ATTENTION",      false, false, false, false, true,  SectionGridType::ATTENTION,  SectionSharedMemStrategy::ATTENTION },
  { KernelSectionType::REDUCTION,          "REDUCTION",      false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::TREE_REDUCTION },
  { KernelSectionType::NORMALIZATION,      "NORMALIZATION",  false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::TREE_REDUCTION },
  { KernelSectionType::GATHER,             "GATHER",         false, false, false, true,  true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::GATHER_ND,          "GATHER_ND",      false, false, false, true,  true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::CONCAT,             "CONCAT",         false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::SPLIT,              "SPLIT",          false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::SPLIT_V,            "SPLIT_V",        false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::STACK,              "STACK",          false, false, false, true,  true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::STRIDED_SLICE,      "STRIDED_SLICE",  false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::TILE,               "TILE",           false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::SCATTER_ND,         "SCATTER_ND",     false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::SCATTER_ND_UPDATE,  "SCATTER_ND_UPD", false, false, false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::SHAPE_MANIPULATION, "SHAPE_MANIP",    false, true,  false, false, true,  SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::CONSTANT_GENERATION,"CONST_GEN",      false, true,  false, false, false, SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
  { KernelSectionType::CONVOLUTION,        "CONVOLUTION",    false, false, true,  false, false, SectionGridType::CUSTOM,     SectionSharedMemStrategy::NONE },
  { KernelSectionType::IDENTITY,           "IDENTITY",       true,  false, false, true,  false, SectionGridType::LINEAR_1D,  SectionSharedMemStrategy::NONE },
};

static constexpr int SECTION_TYPE_CONFIG_COUNT = sizeof(SECTION_TYPE_CONFIGS) / sizeof(SECTION_TYPE_CONFIGS[0]);

// ── O(1) lookup by enum value ──
inline const SectionTypeConfig& getSectionTypeConfig(KernelSectionType type) {
  int idx = static_cast<int>(type);
  // Bounds check — should never fail if enum and table are in sync
  if (idx >= 0 && idx < SECTION_TYPE_CONFIG_COUNT) {
    return SECTION_TYPE_CONFIGS[idx];
  }
  // Fallback to ELEMENTWISE config for safety
  return SECTION_TYPE_CONFIGS[0];
}

// ── Decision functions ──

// Should this section be compiled as a standalone kernel (not merged with neighbors)?
inline bool shouldBeStandalone(const SectionTypeConfig& cfg,
                               bool sectionFusionEnabled,
                               const KernelSection& section) {
  if (cfg.alwaysStandalone) return true;

  // FUSED_ATTENTION: runtime check — standalone unless single query tile
  if (cfg.type == KernelSectionType::FUSED_ATTENTION) {
    int batchHeads = std::max(1, section.batchSize) * std::max(1, section.numHeads);
    return section.gridRequirement > batchHeads;
  }

  // Tiny axis-0 concat sections are primarily scalar/vector shape bookkeeping in decode.
  // Keeping them standalone blocks the surrounding attention-neighborhood fusion for no
  // real benefit, while the large KV append concats still remain standalone because they
  // use axis 2 and/or require many blocks.
  if (cfg.type == KernelSectionType::CONCAT &&
      sectionFusionEnabled &&
      section.concatAxis == 0 &&
      section.gridRequirement <= 1) {
    return false;
  }

  // DATA_MOVEMENT types that aren't fusion-verified: standalone to avoid accuracy regressions
  // (SPLIT, SPLIT_V, CONCAT, CONST_GEN have subtle offset calculation issues when merged)
  if (sectionFusionEnabled) {
    return !cfg.fusionVerified && !cfg.compiledByDefault;
  }

  // Without section fusion, only compiledByDefault types (EW, IDENTITY) can merge
  return !cfg.compiledByDefault;
}

// Should this section fall back to native CUDA (cuBLAS/native ops)?
inline bool shouldFallback(const SectionTypeConfig& cfg,
                           bool compileAll,
                           const std::unordered_set<KernelSectionType>& includedTypes) {
  if (cfg.alwaysFallback) return true;

  if (!compileAll) {
    // Default: only compiledByDefault types (ELEMENTWISE, IDENTITY) are compiled
    return !cfg.compiledByDefault;
  }

  // compileAll mode with include types filter: compile compiledByDefault + explicitly listed
  if (!includedTypes.empty()) {
    if (!cfg.compiledByDefault && includedTypes.find(cfg.type) == includedTypes.end()) {
      return true;
    }
  }

  return false;
}

// Get human-readable name for a section type
inline const char* sectionTypeName(KernelSectionType type) {
  return getSectionTypeConfig(type).name;
}

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SECTION_TYPE_CONFIG_H
