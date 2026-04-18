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

// GPU graph backend (Triton/NVRTC/PTX) execution methods.
//
// Contains getGpuGraphBackend() which selects the best available GPU compiler
// backend (Triton > NVRTC > PTX) based on the configured GraphExecutionMode,
// and executeSegmentWithGpuGraph() which drives segment compilation, CUDA graph
// capture/replay for Triton fused kernels, and native ordered-range orchestration.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspConstants.h>
#include <graph/DspHashUtils.h>
#include <graph/DspStreamGuard.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/gpu/ViewRecipe.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/MmulHelper.h>
#include <ops/OpTraitTable.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <config.h>
#if defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX
#include <graph/gpu/OpCategoryTable.h>
#endif

// Forward declaration for TritonGraphBackend (full header only available when HAVE_TRITON)
namespace sd { namespace graph { class TritonGraphBackend; } }

// Portable buffer accessor: specialBuffer() on CUDA, buffer() on CPU.
#ifdef SD_CUDA
#define DSP_BUF(arr) ((arr)->specialBuffer())
#else
#define DSP_BUF(arr) ((arr)->buffer())
#endif

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// GPU graph backends (conditional)
#if HAVE_TRITON && defined(SD_CUDA)
#include <graph/gpu/TritonGraphBackend.h>
#endif
#ifdef SD_CUDA
#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/gpu/PtxGraphBackend.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <memory/cuda/CudaMemoryPool.h>
#endif
#ifdef SD_TPU
#include <graph/tpu/TpuGraphBackend.h>
#endif
#ifdef HAVE_HEXAGON_MLIR
#include <graph/hexagon/HexagonGraphBackend.h>
#endif

namespace sd {
namespace graph {

#ifdef SD_CUDA
// Pre-allocated cross-stream sync event, reused across replay calls to avoid
// cudaEventCreate/Destroy overhead (~42 pairs per step × 14 segments).
// Created lazily on first use with cudaEventDisableTiming for minimal overhead.
static thread_local cudaEvent_t tl_crossStreamEvent = nullptr;

static inline cudaEvent_t getCrossStreamEvent() {
  if (tl_crossStreamEvent == nullptr) {
    cudaEventCreateWithFlags(&tl_crossStreamEvent, cudaEventDisableTiming);
  }
  return tl_crossStreamEvent;
}
#endif

// ── Segment bucket classification ──────────────────────────────────────────
// Classifies gap slots into bucket types based on slot traits and
// materialization behavior. Used by DSP segment bucket diagnostics.

// Resolve the TritonOpCategory for a slot's op. Returns UNSUPPORTED if the
// op cannot be identified.
#if defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX
static TritonOpCategory resolveOpCategory(int slotIdx, NativeSlot* slots) {
  if (slots == nullptr) return TritonOpCategory::UNSUPPORTED;
  const std::string& opName = slots[slotIdx].ident.opName;
  if (opName.empty()) return TritonOpCategory::UNSUPPORTED;

  const auto& table = getOpCategoryTable();
  auto it = table.find(opName);
  return (it != table.end()) ? it->second : TritonOpCategory::UNSUPPORTED;
}
#endif

static uint32_t resolveStructuralSlotTraits(int slotIdx, NativeSlot* slots) {
  if (slots == nullptr) return 0;

  uint32_t traits = 0;
  const auto& slot = slots[slotIdx];
  if (slot.ident.op != nullptr && slot.ident.op->getOpDescriptor() != nullptr) {
    traits |= slot.ident.op->getOpDescriptor()->getTraits();
  }
  // Fallback: look up traits by op name from the trait table.
  if (traits == 0 && !slot.ident.opName.empty()) {
    traits |= sd::ops::getOpTraitsByName(slot.ident.opName);
  }

  if (slot.flags.isViewCapableOp) traits |= sd::ops::OP_TRAIT_VIEW_PRODUCING;
  if (slot.flags.isIdentityOp) traits |= sd::ops::OP_TRAIT_IDENTITY;
  if (slot.flags.outputShapeDependsOnInputValues) traits |= sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE;
  if (slot.flags.isDataDependent) traits |= sd::ops::OP_TRAIT_DATA_DEPENDENT;

  return traits;
}

static uint32_t resolveSlotTraits(int slotIdx, NativeSlot* slots) {
  uint32_t traits = resolveStructuralSlotTraits(slotIdx, slots);
  if (traits != 0) {
    return traits;
  }

  // Coarse trait recovery is only for legacy/unknown slots with no descriptor-backed
  // structural traits. It must not invent view semantics from the category table:
  // categories like SHAPE_MANIPULATION include both zero-copy views and materializing
  // ops such as broadcast_to.
#if defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX
  TritonOpCategory cat = resolveOpCategory(slotIdx, slots);
  switch (cat) {
    case TritonOpCategory::IDENTITY:
      traits |= sd::ops::OP_TRAIT_IDENTITY | sd::ops::OP_TRAIT_VIEW_PRODUCING;
      break;
    case TritonOpCategory::CONSTANT_GENERATION:
      traits |= sd::ops::OP_TRAIT_CONSTANT_GENERATION;
      break;
    case TritonOpCategory::DATA_MOVEMENT:
      traits |= sd::ops::OP_TRAIT_DATA_MOVEMENT;
      break;
    case TritonOpCategory::FUSED_ATTENTION:
      traits |= sd::ops::OP_TRAIT_ATTENTION;
      break;
    case TritonOpCategory::NORMALIZATION:
      traits |= sd::ops::OP_TRAIT_NORMALIZATION;
      break;
    case TritonOpCategory::MATMUL:
      traits |= sd::ops::OP_TRAIT_MATMUL;
      break;
    case TritonOpCategory::REDUCTION:
      traits |= sd::ops::OP_TRAIT_REDUCTION;
      break;
    case TritonOpCategory::UNARY_ELEMENTWISE:
      traits |= sd::ops::OP_TRAIT_UNARY_ELEMENTWISE;
      break;
    case TritonOpCategory::BINARY_ELEMENTWISE:
      traits |= sd::ops::OP_TRAIT_BINARY_ELEMENTWISE;
      break;
    case TritonOpCategory::TERNARY:
      traits |= sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE;
      break;
    case TritonOpCategory::COMPARISON:
      traits |= sd::ops::OP_TRAIT_COMPARISON;
      break;
    case TritonOpCategory::LOGICAL:
      traits |= sd::ops::OP_TRAIT_LOGICAL;
      break;
    default:
      break;
  }
#endif  // SD_CUDA || HAVE_MLIR || HAVE_TRITON || HAVE_MLX

  return traits;
}

static bool slotHasAnyTrait(int slotIdx, NativeSlot* slots, uint32_t traits) {
  return (resolveSlotTraits(slotIdx, slots) & traits) != 0;
}

static bool slotIsViewRecipeOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots,
                         sd::ops::OP_TRAIT_VIEW_PRODUCING | sd::ops::OP_TRAIT_IDENTITY);
}


static bool segmentHasInternalValueShapeInputs(const GraphSegment& seg, NativeSlot* slots) {
  return dsp::segmentHasInternalValueShapeInputs(seg, slots);
}

// ── View recipe capture, validation, and installation ──────────────────────

// Map an op name to a ViewRecipeType using the category table for semantic
// classification, with string matching only for sub-type disambiguation.
static ViewRecipeType opToViewRecipeType(const char* opName) {
  std::string lower(opName ? opName : "");
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower == "reshape_no_copy") return ViewRecipeType::RESHAPE_NO_COPY;
  if (lower == "reshape" || lower == "flatten" || lower == "flatten_2d")
    return ViewRecipeType::RESHAPE;
  if (lower == "permute" || lower.find("transpose") != std::string::npos)
    return ViewRecipeType::PERMUTE;
  if (lower == "expand_dims") return ViewRecipeType::EXPAND_DIMS;
  if (lower == "squeeze") return ViewRecipeType::SQUEEZE;
  if (lower == "strided_slice") return ViewRecipeType::STRIDED_SLICE;

  return ViewRecipeType::RESHAPE;  // Default recipe kind for view-compatible ops
}

// Check if the input array is C-contiguous and suitable for view creation.
static bool isViewCompatibleInput(NDArray* input) {
  if (input == nullptr || input->dataBuffer() == nullptr) return false;
  // View requires C-contiguous layout for simple offset + stride math
  return input->ordering() == 'c';
}

static bool resolveViewRecipeOwnerSlot(int outputSlotIdx,
                                       const SlotBufferInfo* slotOwnership,
                                       int totalOutputSlots,
                                       int& outOwnerSlotIdx) {
  outOwnerSlotIdx = outputSlotIdx;
  if (slotOwnership == nullptr || outputSlotIdx < 0 || outputSlotIdx >= totalOutputSlots) {
    return false;
  }

  std::vector<uint8_t> seen(static_cast<size_t>(totalOutputSlots), 0);
  int current = outputSlotIdx;
  while (current >= 0 && current < totalOutputSlots &&
         slotOwnership[current].ownership == BufferOwnership::VIEW_OF_SLOT) {
    if (seen[current]) {
      return false;
    }
    seen[current] = 1;

    int parent = slotOwnership[current].parentSlotIdx;
    if (parent < 0 || parent >= totalOutputSlots) {
      return false;
    }
    current = parent;
  }

  outOwnerSlotIdx = current;
  return current >= 0 && current < totalOutputSlots;
}

// Capture a view recipe for a single view-producing slot.
// Returns true if the recipe was successfully captured, false if the op
// would materialize (hard error path — caller should fail the segment).
static bool captureViewRecipe(int slotIdx, NativeSlot* slots, NDArray** outputSlots,
                              NDArray** externalInputs, int numExternalInputs,
                              SlotBufferInfo* slotOwnership, int totalOutputSlots,
                              ViewRecipe& outRecipe, std::string& outError) {
  outRecipe = ViewRecipe();
  outRecipe.outputSlotIndex = -1;

  if (slots == nullptr) {
    outError = "null slots array";
    return false;
  }

  const std::string& opNameStr3 = slots[slotIdx].ident.opName;
  if (opNameStr3.empty()) {
    outError = "no op name";
    return false;
  }

  if (!slotIsViewRecipeOp(slotIdx, slots)) {
    outError = "op is not view-capable (would materialize)";
    return false;
  }

  outRecipe.type = opToViewRecipeType(opNameStr3.c_str());

  // Find the output slot index for this slot
  if (slots[slotIdx].wiring.numOutputs > 0) {
    outRecipe.outputSlotIndex = slots[slotIdx].wiring.outputSlotIndices[0];
  }

  // Resolve the source input for this view op. For replay we need the
  // actual owning buffer observed during warmup, not just the logical input
  // edge, otherwise view chains can reinstall against an intermediate array
  // whose buffer is smaller than the captured output view.
  int sourceIdx = slots[slotIdx].wiring.inputSourceIndices[0];
  if (outRecipe.outputSlotIndex >= 0 && outRecipe.outputSlotIndex < totalOutputSlots &&
      slotOwnership != nullptr &&
      slotOwnership[outRecipe.outputSlotIndex].ownership == BufferOwnership::VIEW_OF_SLOT) {
    int ownerSlotIdx = -1;
    if (!resolveViewRecipeOwnerSlot(outRecipe.outputSlotIndex, slotOwnership,
                                    totalOutputSlots, ownerSlotIdx)) {
      outError = "view ownership chain is invalid or cyclic";
      return false;
    }
    sourceIdx = ownerSlotIdx;
  }
  outRecipe.sourceSlotIndex = sourceIdx;

  NDArray* source = nullptr;
  if (sourceIdx < 0) {
    int extIdx = -(sourceIdx + 1);
    if (extIdx >= 0 && extIdx < numExternalInputs) {
      source = externalInputs[extIdx];
    }
  } else if (sourceIdx >= 0) {
    source = outputSlots[sourceIdx];
  }

  if (source == nullptr || source->dataBuffer() == nullptr) {
    outError = "source array is null or has no data buffer";
    return false;
  }

  if (!isViewCompatibleInput(source)) {
    outError = "source is not C-contiguous — view not possible, must materialize";
    return false;
  }

  NDArray* out = nullptr;
  if (outputSlots != nullptr && outRecipe.outputSlotIndex >= 0) {
    out = outputSlots[outRecipe.outputSlotIndex];
  }
  if (out == nullptr || out->dataBuffer() == nullptr) {
    outError = "output array is null; cannot capture exact replay view state";
    return false;
  }

  if (out->dataBuffer() != source->dataBuffer()) {
    outError = "observed output did not alias the captured source buffer";
    return false;
  }

  const LongType outputOffset = out->offset();
  const LongType outputLength = out->lengthOf();
  const LongType sourceBufferElems = source->dataBuffer()->getNumElements();
  if (outputOffset < 0 || outputLength < 0 || outputOffset > sourceBufferElems ||
      outputLength > (sourceBufferElems - outputOffset)) {
    outError = "observed output view exceeds source buffer capacity";
    return false;
  }

  // Capture the source buffer address and size only after proving the
  // observed output really was a zero-copy alias of that source buffer.
  outRecipe.capturedSourceAddr = DSP_BUF(source);
  outRecipe.capturedSourceBytes = source->dataBuffer()->getLenInBytes();

  // Capture output shape from the observed output array.
  outRecipe.rank = std::min((int)out->rankOf(), (int)ViewRecipe::MAX_SHAPE_RANK);
  outRecipe.outputOrder = out->ordering();
  // Store contiguous flag as EWS: 1 if contiguous, 0 otherwise.
  // EWS values are unreliable; use stride check instead.
  outRecipe.outputEws = shape::strideDescendingCAscendingF(out->shapeInfo()) ? (LongType)1 : (LongType)0;
  outRecipe.outputOffset = outputOffset;
  outRecipe.outputExtra = ArrayOptions::extra(out->shapeInfo());
  for (int i = 0; i < outRecipe.rank; i++) {
    outRecipe.outputShape[i] = out->sizeAt(i);
    outRecipe.outputStrides[i] = out->strideAt(i);
  }
  // Note: slot-level shape info is not available here — rely on output array

  // For permute: capture the permutation vector from tArgs
  if (outRecipe.type == ViewRecipeType::PERMUTE && slots[slotIdx].args.tArgs != nullptr) {
    int permRank = std::min(slots[slotIdx].args.numTArgs, (int)ViewRecipe::MAX_SHAPE_RANK);
    for (int i = 0; i < permRank; i++) {
      outRecipe.perm[i] = (int)slots[slotIdx].args.tArgs[i];
    }
  }

  // For strided_slice: capture begin/end/stride
  if (outRecipe.type == ViewRecipeType::STRIDED_SLICE) {
    outRecipe.sliceRank = outRecipe.rank;
    for (int i = 0; i < outRecipe.sliceRank; i++) {
      outRecipe.sliceBegin[i] = (slots[slotIdx].args.iArgs != nullptr && i < slots[slotIdx].args.numIArgs)
                                ? slots[slotIdx].args.iArgs[i] : 0;
      outRecipe.sliceEnd[i] = (slots[slotIdx].args.tArgs != nullptr && i < slots[slotIdx].args.numTArgs)
                              ? (LongType)slots[slotIdx].args.tArgs[i] : outRecipe.outputShape[i];
      outRecipe.sliceStride[i] = 1;  // Default stride
    }
  }

  outRecipe.validated = false;  // Will be validated during POINTERS_STABLE
  return true;
}

static LongType* buildCapturedViewShapeInfo(const ViewRecipe& recipe, NDArray* source) {
  if (source == nullptr) return nullptr;

  LongType* shapeInfo = ShapeBuilders::createShapeInfo(
      source->dataType(),
      recipe.outputOrder,
      recipe.rank,
      recipe.outputShape,
      recipe.outputStrides,
      nullptr,
      recipe.outputExtra);
  if (shapeInfo == nullptr) return nullptr;

  auto len = shape::shapeInfoLength(shapeInfo);
  shapeInfo[len - 2] = recipe.outputEws;
  shape::setOrder(shapeInfo, recipe.outputOrder);
  return shapeInfo;
}

// Capture view recipes for all view-capable gap slots in a segment.
// Returns the number of recipes captured.
static int captureViewRecipesForSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots,
                                        SlotBufferInfo* slotOwnership,
                                        int totalOutputSlots) {
  seg.exec.viewRecipes = ViewRecipeChain();
  seg.exec.viewRecipes.segmentStartSlot = seg.def.startSlot;
  seg.exec.viewRecipes.segmentEndSlot = seg.def.endSlot;

  int captured = 0;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    if (!slotIsViewRecipeOp(s, slots)) continue;

    ViewRecipe recipe;
    std::string error;
    if (captureViewRecipe(s, slots, outputSlots, externalInputs, numExternalInputs,
                          slotOwnership, totalOutputSlots,
                          recipe, error)) {
      seg.exec.viewRecipes.recipes.push_back(std::move(recipe));
      captured++;
    } else {
      DSP_DIAG_SEG(FALLBACK, s, "VIEW_RECIPE_FAIL: seg[%d-%d] slot %d: %s",
                   seg.def.startSlot, seg.def.endSlot, s, error.c_str());
    }
  }

  return captured;
}

// Validate view recipes during POINTERS_STABLE phase.
// Checks that source buffer addresses haven't changed since capture.
// Returns true if all recipes are valid, false if any source changed.
static bool validateViewRecipes(GraphSegment& seg, NDArray** outputSlots,
                                NDArray** externalInputs, int numExternalInputs) {
  for (auto& recipe : seg.exec.viewRecipes.recipes) {
    NDArray* source = nullptr;
    int sourceIdx = recipe.sourceSlotIndex;
    if (sourceIdx < 0) {
      int extIdx = -(sourceIdx + 1);
      if (extIdx >= 0 && extIdx < numExternalInputs) {
        source = externalInputs[extIdx];
      }
    } else {
      source = outputSlots[sourceIdx];
    }

    if (source == nullptr || source->dataBuffer() == nullptr) {
      DSP_DIAG(FALLBACK, "VIEW_RECIPE_VALIDATE: seg[%d-%d] slot %d source is null",
               seg.def.startSlot, seg.def.endSlot, recipe.outputSlotIndex);
      return false;
    }

    void* currentAddr = DSP_BUF(source);
    if (currentAddr != recipe.capturedSourceAddr) {
      DSP_DIAG(FALLBACK, "VIEW_RECIPE_VALIDATE: seg[%d-%d] slot %d source addr changed: "
                         "captured=%p current=%p",
               seg.def.startSlot, seg.def.endSlot, recipe.outputSlotIndex,
               recipe.capturedSourceAddr, currentAddr);
      return false;
    }

    LongType currentSourceBytes = source->dataBuffer()->getLenInBytes();
    if (currentSourceBytes != recipe.capturedSourceBytes) {
      DSP_DIAG(EXECUTE, "VIEW_RECIPE_VALIDATE: seg[%d-%d] slot %d source bytes changed: "
                        "captured=%lld current=%lld sourceSlot=%d",
               seg.def.startSlot, seg.def.endSlot, recipe.outputSlotIndex,
               static_cast<long long>(recipe.capturedSourceBytes),
               static_cast<long long>(currentSourceBytes),
               recipe.sourceSlotIndex);
      return false;
    }

    recipe.validated = true;
  }

  return true;
}

// Install view recipes during REPLAYING phase.
// Creates zero-copy views from captured recipes and installs them into
// outputSlots so that consumer replay ops see the correct buffers.
static void installViewRecipes(GraphSegment& seg, NDArray** outputSlots,
                               int totalOutputSlots, NDArray** externalInputs,
                               int numExternalInputs) {
  for (const auto& recipe : seg.exec.viewRecipes.recipes) {
    if (!recipe.validated) continue;

    NDArray* source = nullptr;
    int sourceIdx = recipe.sourceSlotIndex;
    if (sourceIdx < 0) {
      int extIdx = -(sourceIdx + 1);
      if (extIdx >= 0 && extIdx < numExternalInputs) {
        source = externalInputs[extIdx];
      }
    } else {
      source = outputSlots[sourceIdx];
    }

    if (source == nullptr || source->dataBuffer() == nullptr) continue;

    int outputSlotIdx = recipe.outputSlotIndex;
    if (outputSlotIdx < 0 || outputSlotIdx >= totalOutputSlots) continue;

    LongType* shapeInfo = buildCapturedViewShapeInfo(recipe, source);
    if (shapeInfo == nullptr) {
      DSP_DIAG(EXECUTE, "VIEW_RECIPE_INSTALL_FAIL: seg[%d-%d] slot %d shapeInfo build failed",
               seg.def.startSlot, seg.def.endSlot, outputSlotIdx);
      continue;
    }

    LongType viewOffset = recipe.outputOffset;
    LongType viewLength = shape::length(shapeInfo);
    LongType sourceBufferElems = source->dataBuffer()->getNumElements();
    if (viewOffset < 0 || viewOffset > sourceBufferElems ||
        viewLength > (sourceBufferElems - viewOffset)) {
      delete[] shapeInfo;
      std::string msg = "DSP replay phase violation: view recipe for seg[" +
                        std::to_string(seg.def.startSlot) + "-" + std::to_string(seg.def.endSlot) +
                        "] output slot " + std::to_string(outputSlotIdx) +
                        " requires " + std::to_string(viewLength) +
                        " elements at offset " + std::to_string(viewOffset) +
                        " from source slot " + std::to_string(recipe.sourceSlotIndex) +
                        ", but source buffer has only " + std::to_string(sourceBufferElems) +
                        " elements";
      THROW_EXCEPTION(msg.c_str());
    }

    // Create the view array
    NDArray* view = new NDArray(source->dataBuffer(),
                                shapeInfo,
                                LaunchContext::defaultContext(),
                                viewOffset);
    delete[] shapeInfo;
    if (view != nullptr) {
      // Install into output slot
      if (outputSlots[outputSlotIdx] != nullptr &&
          outputSlots[outputSlotIdx] != source) {
        delete outputSlots[outputSlotIdx];
      }
      outputSlots[outputSlotIdx] = view;
    }
  }
}

// Default capture host workspace size for Triton path (32MB, same as non-Triton path).
// Configurable via ND4J_DSP_CAPTURE_HOST_WORKSPACE_MB env var.
#ifdef SD_CUDA
static size_t TRITON_CAPTURE_HOST_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(Environment::getInstance().dsp().captureHostWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();

// Default capture workspace size for Triton graph capture (128MB).
// Configurable via ND4J_DSP_CAPTURE_WORKSPACE_MB env var.
static size_t TRITON_CAPTURE_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(Environment::getInstance().dsp().captureWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();
#endif

// Local helper: convert Status enum to human-readable string for diagnostics.
static const char* statusName_gpu(Status status) {
  switch (status) {
    case Status::OK: return "OK";
    case Status::BAD_INPUT: return "BAD_INPUT";
    case Status::BAD_SHAPE: return "BAD_SHAPE";
    case Status::BAD_RANK: return "BAD_RANK";
    case Status::BAD_PARAMS: return "BAD_PARAMS";
    case Status::BAD_OUTPUT: return "BAD_OUTPUT";
    case Status::BAD_RNG: return "BAD_RNG";
    case Status::BAD_EPSILON: return "BAD_EPSILON";
    case Status::BAD_GRADIENTS: return "BAD_GRADIENTS";
    case Status::BAD_BIAS: return "BAD_BIAS";
    case Status::VALIDATION: return "VALIDATION";
    case Status::BAD_GRAPH: return "BAD_GRAPH";
    case Status::BAD_LENGTH: return "BAD_LENGTH";
    case Status::BAD_DIMENSIONS: return "BAD_DIMENSIONS";
    case Status::BAD_ORDER: return "BAD_ORDER";
    case Status::BAD_ARGUMENTS: return "BAD_ARGUMENTS";
    case Status::DOUBLE_WRITE: return "DOUBLE_WRITE";
    case Status::DOUBLE_READ: return "DOUBLE_READ";
    case Status::KERNEL_FAILURE: return "KERNEL_FAILURE";
    case Status::EQ_TRUE: return "EQ_TRUE";
    case Status::EQ_FALSE: return "EQ_FALSE";
    case Status::MAYBE: return "MAYBE";
    default: return "UNKNOWN";
  }
}

// Helper: extract specialBuffer() device addresses from NDArray** into void** for
// address snapshot diagnostics. Thread-local to avoid repeated allocation.
static void extractDeviceAddrs(NDArray** arrays, int count, std::vector<void*>& out) {
  out.resize(count);
  for (int i = 0; i < count; i++) {
    out[i] = (arrays != nullptr && arrays[i] != nullptr)
             ? DSP_BUF(arrays[i]) : nullptr;
  }
}

/**
* Compute FNV-1a hash of slot output specialBuffer() addresses for a segment.
* Used to verify that output buffers haven't been reallocated between capture
* and replay — stale addresses in a CUDA graph cause SIGSEGV or corruption.
*/
static LongType computeSlotAddrHash(NDArray** outputSlots, int startSlot, int endSlot, int totalSlots) {
  return dsp::computeSlotAddrHash(outputSlots, startSlot, endSlot, totalSlots,
                                  [](NDArray* a) -> void* { return DSP_BUF(a); });
}


// ── GPU CONTEXT PROBE ──────────────────────────────────────────────────────
// Shared helper that dumps multi-device memory state + CUDA context health.
// Called by all error handlers to detect downstream/pre-existing errors.
#ifdef SD_CUDA
static void dumpGpuContextState(int failedDeviceId, const char* errorType) {
  // 1. Check for pre-existing CUDA error (downstream error detection)
  cudaError_t preExisting = cudaPeekAtLastError();
  if (preExisting != cudaSuccess) {
    DSP_DIAG(MEMORY, "%s: PRE-EXISTING CUDA ERROR on device %d: %d (%s) — "
                     "this may be a downstream error from a previous operation",
             errorType, failedDeviceId,
             static_cast<int>(preExisting), cudaGetErrorString(preExisting));
  }

  // 2. Failed device: full detail
  {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, failedDeviceId);
    size_t gpuFree = 0, gpuTotal = 0;
    cudaSetDevice(failedDeviceId);
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    DSP_DIAG(MEMORY, "%s: device %d '%s' (cc=%d.%d): free=%zuMB total=%zuMB used=%zuMB "
                     "multiProcessorCount=%d",
             errorType, failedDeviceId, props.name,
             props.major, props.minor,
             gpuFree / (1024*1024), gpuTotal / (1024*1024),
             (gpuTotal - gpuFree) / (1024*1024),
             props.multiProcessorCount);
  }

  // 3. Other devices: one-line summary each
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  for (int d = 0; d < deviceCount; d++) {
    if (d == failedDeviceId) continue;
    cudaSetDevice(d);
    size_t otherFree = 0, otherTotal = 0;
    cudaMemGetInfo(&otherFree, &otherTotal);
    cudaError_t otherErr = cudaPeekAtLastError();
    DSP_DIAG(MEMORY, "%s: device %d: free=%zuMB total=%zuMB%s",
             errorType, d, otherFree / (1024*1024), otherTotal / (1024*1024),
             otherErr != cudaSuccess ?
             (std::string(" CUDA_ERROR=") + cudaGetErrorString(otherErr)).c_str() : "");
  }

  // 4. Restore original device
  cudaSetDevice(failedDeviceId);

  // 5. Report graph execution active state
  DSP_DIAG(MEMORY, "%s: tl_graphExecutionActive=%d",
           errorType, tl_graphExecutionActive ? 1 : 0);
}

// ── DISTINCT ERROR HANDLERS ────────────────────────────────────────────────

static Status reportOomError(GraphSegment& seg, const char* phase,
                             size_t requestedBytes, int deviceId) {
  dumpGpuContextState(deviceId, "OOM");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(MEMORY,
           "OOM ERROR in seg[%d-%d] during '%s' on device %d: "
           "requested=%zuMB gpuFree=%zuMB gpuTotal=%zuMB gpuUsed=%zuMB "
           "executionCount=%d phase=%d",
           seg.def.startSlot, seg.def.endSlot, phase, deviceId,
           requestedBytes / (1024*1024), gpuFree / (1024*1024),
           gpuTotal / (1024*1024), (gpuTotal - gpuFree) / (1024*1024),
           seg.exec.executionCount, static_cast<int>(seg.exec.currentPhase));
  seg.exec.compilationFailed = true;
  return Status::KERNEL_FAILURE;
}

static Status reportCaptureError(GraphSegment& seg, const char* step,
                                 cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "CAPTURE");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(EXECUTE,
           "CAPTURE ERROR in seg[%d-%d] at step '%s' on device %d: "
           "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
           "executionCount=%d numOps=%d compiledBy='%s'",
           seg.def.startSlot, seg.def.endSlot, step, deviceId,
           static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
           gpuFree / (1024*1024), gpuTotal / (1024*1024),
           seg.exec.executionCount, seg.def.endSlot - seg.def.startSlot + 1,
           seg.exec.compiledByBackend.c_str());
  seg.exec.compilationFailed = true;
  cudaGetLastError(); // clear error state
  return Status::KERNEL_FAILURE;
}

static Status reportReplayError(GraphSegment& seg, const char* step,
                                cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "REPLAY");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(EXECUTE,
           "REPLAY ERROR in seg[%d-%d] at step '%s' on device %d: "
           "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
           "executionCount=%d hasReplayHandle=%d",
           seg.def.startSlot, seg.def.endSlot, step, deviceId,
           static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
           gpuFree / (1024*1024), gpuTotal / (1024*1024),
           seg.exec.executionCount,
           seg.exec.replayHandle != nullptr ? 1 : 0);
  seg.exec.compilationFailed = true;
  cudaGetLastError(); // clear error state
  return Status::KERNEL_FAILURE;
}

#if HAVE_TRITON && defined(SD_CUDA)
static bool findUnsupportedTritonReplayGap(TritonGraphBackend* tritonBackend,
                                          const GraphSegment& seg,
                                          NativeSlot* slots,
                                          int* firstGapSlot,
                                          int* lastCoveredSlot,
                                          int* gapSlotCount) {
 if (firstGapSlot != nullptr) *firstGapSlot = -1;
 if (lastCoveredSlot != nullptr) *lastCoveredSlot = -1;
 if (gapSlotCount != nullptr) *gapSlotCount = 0;
 if (tritonBackend == nullptr) return false;

 auto gapSlots = tritonBackend->getGapSlots(seg, slots);
 if (gapSlotCount != nullptr) *gapSlotCount = static_cast<int>(gapSlots.size());
 if (gapSlots.empty()) return false;

 int maxCoveredSlot = -1;
 for (int slot = seg.def.startSlot; slot <= seg.def.endSlot; slot++) {
   if (gapSlots.find(slot) == gapSlots.end()) {
     maxCoveredSlot = slot;
   }
 }
 if (lastCoveredSlot != nullptr) *lastCoveredSlot = maxCoveredSlot;
 if (maxCoveredSlot < 0) return false;

 int earliestUnsupportedGap = -1;
 for (int slot = seg.def.startSlot; slot <= seg.def.endSlot; slot++) {
   if (gapSlots.find(slot) != gapSlots.end() && slot < maxCoveredSlot) {
     earliestUnsupportedGap = slot;
     break;
   }
 }
 if (firstGapSlot != nullptr) *firstGapSlot = earliestUnsupportedGap;
 return earliestUnsupportedGap >= 0;
}

/**
* Build an ordered replay schedule for a mixed Triton/gap segment.
*
* Given a segment like seg[200-399] with gaps at [298-312] and [347-369],
* this produces:
*   unit 0: TRITON_ISLAND [200-297]  islandIndex=0
*   unit 1: GAP            [298-312] islandIndex=-1
*   unit 2: TRITON_ISLAND [313-346]  islandIndex=1
*   unit 3: GAP            [347-369] islandIndex=-1
*   unit 4: TRITON_ISLAND [370-399]  islandIndex=2
*/
static ReplaySchedule buildCompositeReplaySchedule(const GraphSegment& seg,
                                                  NativeSlot* slots,
                                                  TritonGraphBackend* tritonBackend) {
 ReplaySchedule schedule;
 auto gap_slots = tritonBackend->getGapSlots(seg, slots);

 int islandIdx = 0;
 int rangeStart = seg.def.startSlot;
 bool inIsland = (gap_slots.find(seg.def.startSlot) == gap_slots.end());

 for (int slot = seg.def.startSlot; slot <= seg.def.endSlot + 1; slot++) {
   bool isGap = (slot <= seg.def.endSlot && gap_slots.find(slot) != gap_slots.end());
   bool atBoundary = (slot > seg.def.endSlot) || (inIsland && isGap) || (!inIsland && !isGap);

   if (atBoundary && slot > rangeStart) {
     if (inIsland) {
       DSP_DIAG_SEG(SEGMENT, rangeStart,
                    "compositeReplaySchedule UNIT kind=ISLAND islandBefore=%d startSlot=%d endSlot=%d segParent=[%d-%d]",
                    islandIdx, rangeStart, slot - 1, seg.def.startSlot, seg.def.endSlot);
       schedule.units.emplace_back(REPLAY_UNIT_TRITON_ISLAND, rangeStart, slot - 1, islandIdx++);
     } else {
       DSP_DIAG_SEG(SEGMENT, rangeStart,
                    "compositeReplaySchedule UNIT kind=GAP startSlot=%d endSlot=%d segParent=[%d-%d]",
                    rangeStart, slot - 1, seg.def.startSlot, seg.def.endSlot);
       schedule.units.emplace_back(REPLAY_UNIT_GAP, rangeStart, slot - 1, -1);
     }
     rangeStart = slot;
     inIsland = !isGap;
   }
   // If at seg.def.endSlot+1 boundary with pending range, the loop above handles it
   if (slot == seg.def.endSlot + 1 && rangeStart <= seg.def.endSlot) {
     if (inIsland) {
       DSP_DIAG_SEG(SEGMENT, rangeStart,
                    "compositeReplaySchedule UNIT kind=ISLAND-tail islandBefore=%d startSlot=%d endSlot=%d segParent=[%d-%d]",
                    islandIdx, rangeStart, seg.def.endSlot, seg.def.startSlot, seg.def.endSlot);
       schedule.units.emplace_back(REPLAY_UNIT_TRITON_ISLAND, rangeStart, seg.def.endSlot, islandIdx++);
     } else {
       DSP_DIAG_SEG(SEGMENT, rangeStart,
                    "compositeReplaySchedule UNIT kind=GAP-tail startSlot=%d endSlot=%d segParent=[%d-%d]",
                    rangeStart, seg.def.endSlot, seg.def.startSlot, seg.def.endSlot);
       schedule.units.emplace_back(REPLAY_UNIT_GAP, rangeStart, seg.def.endSlot, -1);
     }
   }
 }

 // Pre-allocate replay handles for each island
 schedule.compositeReplayHandles.resize(schedule.units.size());
 return schedule;
}
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// hasCompositeHandles — check if per-island captures are ready for replay
// ═══════════════════════════════════════════════════════════════════════════════
bool NativeDynamicShapePlan::hasCompositeHandles(const GraphSegment& seg) const {
  auto& sched = seg.exec.compositeReplaySchedule;
  for (auto& u : sched.units) {
    if (u.kind == REPLAY_UNIT_TRITON_ISLAND) {
      int idx = u.islandIndex;
      if (idx >= 0 && idx < static_cast<int>(sched.compositeReplayHandles.size()) &&
          sched.compositeReplayHandles[idx] != nullptr &&
          sched.compositeReplayHandles[idx]->isReady()) {
        return true;
      }
    }
  }
  return false;
}

// executeGapFast REMOVED — consolidated into executeSlot() for single code path.
// Gap ops during replay now call executeSlot() directly, which handles:
//   - Identity ops, frozen constants, fused chains
//   - View-capable fast path (reshape/expand_dims/squeeze/strided_slice)
//   - Frozen context execution with proper nullify, sync, and reconcile
// Having two separate paths caused accuracy bugs (missing view-capable handling).

// ═══════════════════════════════════════════════════════════════════════════════
// compositeReplay — single clean replay path for composite (island+gap) segments
// ═══════════════════════════════════════════════════════════════════════════════
//
// Preconditions: shapes frozen, composite handles captured, arg tables populated.
// Executes replay schedule in program order:
//   TRITON_ISLAND → refresh arg tables, pre-zero outputs, graph launch
//   GAP           → executeSlot() per slot (single code path, full correctness)
//

// Diagnostic counter: tracks redundant refreshArgTablesForReplay calls
// (called when argTableStable was already true). Per-thread to avoid contention.
static thread_local int tl_redundantRefreshCount = 0;

Status NativeDynamicShapePlan::compositeReplay(
    GraphSegment& seg, ReplaySchedule& sched,
    NDArray** externalArrays, int numExt, void* stream) {

  // Phase assertion: compositeReplay MUST be called in SHAPES_FROZEN or later.
  // Calling during SLOT_BY_SLOT means shapes aren't stable and graph replay is unsafe.
#ifndef NDEBUG
  if (planPhase_ < PlanPhase::SHAPES_FROZEN) {
    DSP_DIAG(FALLBACK, "PHASE_VIOLATION: compositeReplay called in phase %s, "
                       "requires >= SHAPES_FROZEN. seg[%d-%d] execCount=%d",
             dsp::planPhaseName(planPhase_), seg.def.startSlot, seg.def.endSlot, executeCount_);
    assert(false && "compositeReplay requires planPhase_ >= SHAPES_FROZEN");
  }
#endif

  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Cross-stream sync: ensure Java .assign() writes on default stream are visible
  {
    cudaStream_t defaultStream = nullptr;
    auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    if (defaultStreamPtr != nullptr) defaultStream = *defaultStreamPtr;
    if (defaultStream != nullptr && defaultStream != cudaStr) {
      cudaEvent_t evt = getCrossStreamEvent();
      cudaEventRecord(evt, defaultStream);
      cudaStreamWaitEvent(cudaStr, evt, 0);
      DSP_DIAG(STREAM_SYNC,
               "compositeReplay cross-stream sync: recordedOn=defaultStream=%p waitedOn=dspStream=%p seg=[%d-%d] execCount=%d",
               (void*)defaultStream, (void*)cudaStr,
               seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
    }
  }

  // Set DSP execution stream for async H2D copies
  sd::graph::DspStreamGuard dspStreamGuard(cudaStr);

  // Sync variable external inputs to device.
  // Variable inputs (attention_mask, position_ids, etc.) are modified host-side
  // between replay steps via putScalar()/assign(). These host writes call
  // writePrimary(), making isPrimaryActual()=true and isSpecialActual()=false,
  // so syncToSpecial(false) correctly detects the need for H2D transfer.
  // We use forceSync=true as defense-in-depth for any edge cases where host
  // writes don't properly tick writePrimary().
  if (shapesFrozen_ && !externalInputIsVariable_.empty()) {
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] == nullptr) continue;
      if (ei < static_cast<int>(externalInputIsVariable_.size()) &&
          !externalInputIsVariable_[ei]) continue;
      // Variable inputs (placeholders): force H2D sync to pick up host-side writes.
      // forceSync=true is required because markOrderedRangeDeviceCurrent() calls
      // readSpecial() on inputs after each step, which can leave isSpecialActual()=true
      // even after Java writes new values via writePrimary(). Without forceSync,
      // syncToSpecial() skips the H2D copy and ops use stale device data.
      externalArrays[ei]->dataBuffer()->syncToSpecial(true);
    }
  } else {
    // No variable classification available — sync all with normal conditional check.
    // Do NOT use forceSync here: constants have valid device data (isSpecialActual()=true)
    // and forcing H2D would overwrite device-authoritative data with potentially stale host data.
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr) {
        externalArrays[ei]->syncToDevice();
      }
    }
  }

  // Fingerprint every variable external input at replay entry so step-to-step
  // content drift of placeholders (inputs_embeds, attention_mask, KV caches) is
  // visible. Repeated fingerprints across execCounts imply a "stuck input" bug.
  // Safe here: compositeReplay runs in SHAPES_FROZEN+ (asserted above), not during
  // CUDA graph capture. Gated on EXECUTE diag category; no-op otherwise.
  if (shapesFrozen_ && !externalInputIsVariable_.empty()) {
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] == nullptr) continue;
      if (ei < static_cast<int>(externalInputIsVariable_.size()) &&
          !externalInputIsVariable_[ei]) continue;
      const char* name = (ei < static_cast<int>(externalInputNames_.size()))
                         ? externalInputNames_[ei].c_str() : "?";
      DSP_DIAG_FINGERPRINT("replay-entry", ei, name, externalArrays[ei],
                           seg.exec.executionCount);
    }
  }

  // Defensively re-validate external-input device addresses against the captured
  // arg table. Java-side code paths like SameDiff.associateArrayWithVariable() can
  // rebind a weight to a fresh DataBuffer without notifying the native plan — the
  // old argTable would then contain pointers to freed device memory. Without this
  // check, the fast-replay path below would silently replay a CUDA graph / Triton
  // arg table that reads from the freed buffer, producing wrong outputs rather
  // than a lifecycle error. Cost: one hash of ext-input specialBuffer pointers.
  if (seg.exec.argTableStable && seg.exec.capturedInputAddrKey != 0) {
    LongType currentAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);
    if (currentAddrKey != seg.exec.capturedInputAddrKey) {
      DSP_DIAG(FALLBACK,
               "EXT_INPUT_REBIND_DETECTED: seg[%d-%d] current=%lld captured=%lld "
               "→ invalidating argTableStable (forcing refresh) execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               (long long)currentAddrKey, (long long)seg.exec.capturedInputAddrKey,
               seg.exec.executionCount);
      seg.exec.argTableStable = false;
    }
  }

  // Defensively re-validate INTERNAL output-slot device addresses against the
  // captured hash. Composite Triton graphs bake arg-table pointers for internal
  // slot buffers at capture time. If any internal output slot reallocates
  // between replay steps (view reshape, Java-side GC, DataBuffer rebind), the
  // fast-replay path silently reads/writes the prior step's buffer — the exact
  // fingerprint of "OPTIMAL step N = REF step N-1" stale-output symptoms.
  // The external-input guard above only covers weights/placeholders, not the
  // internal slot buffers produced by earlier ops in the segment.
  if (seg.exec.argTableStable && seg.exec.capturedSlotAddrHash != 0) {
    LongType currentSlotHash = computeSlotAddrHash(
        outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
    if (currentSlotHash != seg.exec.capturedSlotAddrHash) {
      DSP_DIAG(FALLBACK,
               "SLOT_ADDR_DRIFT_DETECTED: seg[%d-%d] current=0x%llx captured=0x%llx "
               "→ invalidating argTableStable (forcing refresh) execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               (long long)currentSlotHash, (long long)seg.exec.capturedSlotAddrHash,
               seg.exec.executionCount);
      seg.exec.argTableStable = false;
    } else {
      DSP_DIAG(EXECUTE,
               "SLOT_ADDR_STABLE: seg[%d-%d] hash=0x%llx execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               (long long)currentSlotHash, seg.exec.executionCount);
    }
  }

  // Refresh arg tables + D2D copy (skip when stable — fast replay path)
  bool useFastReplay = seg.exec.argTableStable &&
                       !Environment::getInstance().tritonVerifyKernels();
#if HAVE_TRITON && defined(SD_CUDA)
  if (!useFastReplay) {
   auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
   if (tritonBackend != nullptr) {
     auto refreshStatus = tritonBackend->refreshArgTablesForReplay(
         seg, externalArrays, numExt, outputSlots_, totalOutputSlots_, stream);
     if (refreshStatus != Status::OK) {
       DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: arg table refresh FAILED seg[%d-%d]",
                seg.def.startSlot, seg.def.endSlot);
       return refreshStatus;
     }
     tritonBackend->copyConsolidatedArgTableToDevice(seg, stream);
   }
 } else {
   DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: fast-replay (argTableStable) — skip refresh seg[%d-%d]",
            seg.def.startSlot, seg.def.endSlot);
 }
#endif
  cudaGetLastError();  // Clear sticky errors

  DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
               "compositeReplay invoking prezeroSegmentOutputs seg=[%d-%d] stream=%p execCount=%d",
               seg.def.startSlot, seg.def.endSlot, (void*)cudaStr, seg.exec.executionCount);
  prezeroSegmentOutputs(seg, stream);

  // Execute units in program order
  DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: seg[%d-%d] %d units execCount=%d",
           seg.def.startSlot, seg.def.endSlot,
           static_cast<int>(sched.units.size()), seg.exec.executionCount);

  for (auto& unit : sched.units) {
    if (unit.kind == REPLAY_UNIT_GAP) {
      DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap [%d-%d]", unit.startSlot, unit.endSlot);
      for (int s = unit.startSlot; s <= unit.endSlot; s++) {
        auto slotStatus = executeSlot(s, externalArrays, numExt, stream);
        if (slotStatus != Status::OK) {
          DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap slot %d FAILED status=%d",
                   s, static_cast<int>(slotStatus));
          return slotStatus;
        }
      }

    } else {  // REPLAY_UNIT_TRITON_ISLAND
      int idx = unit.islandIndex;
      if (idx < 0 || idx >= static_cast<int>(sched.compositeReplayHandles.size()) ||
          !sched.compositeReplayHandles[idx] || !sched.compositeReplayHandles[idx]->isReady()) {
        DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d handle not ready", idx);
        return Status::KERNEL_FAILURE;
      }

      // NOTE: Global batch-zero above already zeroed all needsZeroedOutput buffers
      // for the entire segment, so no per-island pre-zero needed here.

      // Graph launch
      DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d [%d-%d] launching", idx, unit.startSlot, unit.endSlot);
      bool launchOk = sched.compositeReplayHandles[idx]->replay(stream);
      if (!launchOk) {
        DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d launch FAILED", idx);
        return Status::KERNEL_FAILURE;
      }
    }
  }

  // Diagnostic: check final output after composite replay (only when DSP diagnostics enabled)
  if (DSP_DIAG_ENABLED(EXECUTE)) {
    int finalOutputSlot = -1;
    if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
      finalOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
    }
    if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_)
      finalOutputSlot = seg.def.endSlot;
    if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
        outputSlots_[finalOutputSlot] != nullptr) {
      auto* replayOut = outputSlots_[finalOutputSlot];
      if (replayOut->dataType() == FLOAT32 && replayOut->lengthOf() > 0) {
        cudaStreamSynchronize(cudaStr);
        int replayArgmax = dspArgmax(DSP_BUF(replayOut), replayOut->dataType(),
                                     replayOut->lengthOf());
        DSP_DIAG(EXECUTE, "POST_COMPOSITE_REPLAY_ARGMAX seg[%d-%d] slot=%d argmax=%d len=%lld execCount=%d",
                 seg.def.startSlot, seg.def.endSlot, finalOutputSlot, replayArgmax,
                 (long long)replayOut->lengthOf(), seg.exec.executionCount);
      }
    }
  }

  // Update replay tracking
  seg.exec.lastReplayExecCount = seg.exec.executionCount;

  // FORCE_RECAPTURE: invalidate after replay so next step re-captures
  if (Environment::getInstance().tritonForceRecapture()) {
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after replay execCount=%d",
             seg.exec.executionCount);
  }

  return Status::OK;
}

// ── LRU GRAPH EVICTION ──────────────────────────────────────────────────────
// Evicts captured graphs to free GPU memory. Returns number of graphs evicted.
// When dspLruEviction is true, evicts least-recently-replayed graphs first.
// Otherwise evicts smallest (fewest nodes) first (legacy behavior).
int NativeDynamicShapePlan::evictLruGraphs(int segIdx, size_t neededBytes, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  bool lruMode = Environment::getInstance().dspLruEviction();
  int maxEvictions = Environment::getInstance().dspCaptureOomMaxRetries();
  int numEvicted = 0;

  for (int evictAttempt = 0; evictAttempt < maxEvictions; evictAttempt++) {
    // Check if we have enough memory already
    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    if (gpuFree >= neededBytes) {
      DSP_DIAG(MEMORY, "evictLruGraphs: have enough memory after %d evictions (%zuMB free >= %zuMB needed)",
               numEvicted, gpuFree / (1024*1024), neededBytes / (1024*1024));
      break;
    }

    // Find the best candidate to evict
    int evictIdx = -1;
    if (lruMode) {
      // LRU: find segment with smallest lastReplayExecCount (least recently used)
      int lruExecCount = INT_MAX;
      for (size_t si = 0; si < segments_.size(); si++) {
        if (static_cast<int>(si) == segIdx) continue;
        auto& candidate = segments_[si];
        if (!candidate.exec.replayHandle || !candidate.exec.replayHandle->isReady()) continue;
        if (candidate.exec.lastReplayExecCount < lruExecCount) {
          lruExecCount = candidate.exec.lastReplayExecCount;
          evictIdx = static_cast<int>(si);
        }
      }
    } else {
      // Smallest-first: find segment with fewest CUDA graph nodes
      size_t smallestNodes = SIZE_MAX;
      for (size_t si = 0; si < segments_.size(); si++) {
        if (static_cast<int>(si) == segIdx) continue;
        auto& candidate = segments_[si];
        if (!candidate.exec.replayHandle || !candidate.exec.replayHandle->isReady()) continue;
        auto* cudaReplay = dynamic_cast<CudaGraphReplayHandle*>(candidate.exec.replayHandle.get());
        size_t nodeCount = cudaReplay ? cudaReplay->getNumNodes() : 1;
        if (nodeCount == 0) nodeCount = 1;
        if (nodeCount < smallestNodes) {
          smallestNodes = nodeCount;
          evictIdx = static_cast<int>(si);
        }
      }
    }

    if (evictIdx < 0) {
      DSP_DIAG(MEMORY, "evictLruGraphs: no more evictable segments (evicted %d)", numEvicted);
      break;
    }

    // Evict the selected segment
    auto& evictSeg = segments_[evictIdx];
    DSP_DIAG(MEMORY, "evictLruGraphs: evicting seg[%d-%d] (lruExec=%d, mode=%s) for seg idx=%d (attempt %d/%d)",
             evictSeg.def.startSlot, evictSeg.def.endSlot, evictSeg.exec.lastReplayExecCount,
             lruMode ? "LRU" : "smallest", segIdx, evictAttempt + 1, maxEvictions);

    evictSeg.exec.replayHandle->releaseWorkspace(nullptr, evictSeg.def.startSlot);

    // Free pinned host pointers
    evictSeg.exec.replayHandle->freeHostPointers();
    evictSeg.exec.replayHandle->clearExternalAddresses();

    // Destroy replay handle (frees cudaGraphExec + cudaGraph)
    evictSeg.exec.replayHandle.reset();

    // Reset evicted segment for future re-capture
    evictSeg.exec.cachedShapeKey = 0;
    evictSeg.exec.capturedInputAddrKey = 0;
    evictSeg.exec.capturedCreateValueKey = 0;
    evictSeg.exec.compilationFailed = false;
    evictSeg.exec.gapOpsCapturedInGraph = false;
    evictSeg.exec.argTableStable = false;
    evictSeg.exec.compiledByBackend.clear();
    evictSeg.exec.executionCount = 0;
    evictSeg.exec.lastReplayExecCount = 0;

    numEvicted++;

    // Sync to ensure GPU memory is freed
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
    }
    cudaGetLastError();
  }

  // Final pool trim after evictions
  if (numEvicted > 0) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);
    DSP_DIAG(MEMORY, "evictLruGraphs: evicted %d segments, trimmed pool on device %d", numEvicted, deviceId);
  }

  return numEvicted;
}

// ── PROACTIVE PRE-CAPTURE MEMORY CLEANUP ───────────────────────────────────
// Called before workspace allocation when about to capture a graph.
// Frees cached-but-unused GPU memory and evicts LRU graphs if needed.
void NativeDynamicShapePlan::proactivePreCaptureMemoryCleanup(GraphSegment& seg, int segIdx, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  int deviceId = 0;
  cudaGetDevice(&deviceId);

  // 1. Trim CUDA memory pool — cheap, can reclaim hundreds of MB
  DSP_DIAG(MEMORY, "proactive cleanup: trimming pool on device %d for seg[%d-%d]",
           deviceId, seg.def.startSlot, seg.def.endSlot);
  memory::CudaMemoryPool::getInstance().trimPool(deviceId);
  if (cudaStr != nullptr) {
    memory::CudaMemoryPool::getInstance().trimPoolOnStream(deviceId, cudaStr);
  }

  // 2. Check if we have enough memory
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);

  // Estimate needed: capture workspace + cuBLAS workspace (if not allocated) + safety margin
  size_t neededBytes = 0;
  if (sharedCaptureWorkspace_ == nullptr) {
    neededBytes += TRITON_CAPTURE_WORKSPACE_SIZE;  // 128MB default
  }
  if (cublasWorkspaceBuffer_ == nullptr) {
    neededBytes += Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
  }
  neededBytes += Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;

  DSP_DIAG(MEMORY, "proactive cleanup: gpuFree=%zuMB, needed=%zuMB (ws=%zuMB, cublas=%zuMB, safety=%dMB) for seg[%d-%d]",
           gpuFree / (1024*1024), neededBytes / (1024*1024),
           (sharedCaptureWorkspace_ == nullptr ? TRITON_CAPTURE_WORKSPACE_SIZE : 0) / (1024*1024),
           (cublasWorkspaceBuffer_ == nullptr ? (size_t)(Environment::getInstance().dspCublasWorkspaceMb()) : 0),
           Environment::getInstance().dspGraphMetadataSafetyMb(),
           seg.def.startSlot, seg.def.endSlot);

  if (gpuFree >= neededBytes) {
    DSP_DIAG(MEMORY, "proactive cleanup: sufficient memory (%zuMB >= %zuMB), no eviction needed",
             gpuFree / (1024*1024), neededBytes / (1024*1024));
    return;
  }

  // 3. LRU eviction
  DSP_DIAG(MEMORY, "proactive cleanup: insufficient memory (%zuMB < %zuMB), starting LRU eviction",
           gpuFree / (1024*1024), neededBytes / (1024*1024));
  int numEvicted = evictLruGraphs(segIdx, neededBytes, stream);

  // 4. Final trim after evictions
  if (numEvicted > 0) {
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);
  }

  // Log final state
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(MEMORY, "proactive cleanup complete: evicted=%d, gpuFree=%zuMB/%zuMB for seg[%d-%d]",
           numEvicted, gpuFree / (1024*1024), gpuTotal / (1024*1024),
           seg.def.startSlot, seg.def.endSlot);
}

#endif  // SD_CUDA

// isStrictNoFallbackMode_gpu removed — all modes now throw on failure.
// There is no "non-strict" mode. Failures crash loudly, never fall back.

// ─── DSP Verify Helpers ────────────────────────────────────────────────────

// Source type name for diagnostics
static const char* sourceTypeName(int8_t st) {
  switch (static_cast<NativeSourceType>(st)) {
    case SOURCE_CONSTANT: return "CONSTANT";
    case SOURCE_VARIABLE: return "VARIABLE";
    case SOURCE_PLACEHOLDER: return "PLACEHOLDER";
    case SOURCE_OP_OUTPUT: return "OP_OUTPUT";
    default: return "UNKNOWN";
  }
}

#ifdef SD_CUDA
// Templated helpers in DspVerifyUtils.h (dspVerifyCopyValues, dspMaxDiff, dspFormatValues, etc.)
#endif  // SD_CUDA

void NativeDynamicShapePlan::clearGpuBackendFailedCache() {
#if HAVE_TRITON && defined(SD_CUDA)
  TritonGraphBackend::getInstance().clearFailedSegmentCache();
#endif
}

GraphBackend* NativeDynamicShapePlan::getGpuGraphBackend() {
  if (gpuGraphBackendChecked_) return gpuGraphBackend_;
  gpuGraphBackendChecked_ = true;

  // If a specific backend is forced via setGraphExecutionMode(), use only that one.
  // SLOT_BY_SLOT and graph-replay-only modes don't use a GPU compiler backend —
  // they rely on the GraphReplayHandle (CUDA/HIP/L0/Vulkan/Metal) for capture/replay.
  if (graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_CUDA_GRAPHS ||
      graphExecutionMode_ == GraphExecutionMode::GEM_HIP_GRAPHS ||
      graphExecutionMode_ == GraphExecutionMode::GEM_LEVELZERO ||
      graphExecutionMode_ == GraphExecutionMode::GEM_VULKAN ||
      graphExecutionMode_ == GraphExecutionMode::GEM_METAL) {
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }

#if HAVE_TRITON && defined(SD_CUDA)
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON ||
     graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
   auto& triton = TritonGraphBackend::getInstance();
   if (triton.isAvailable()) {
     gpuGraphBackend_ = &triton;
     DSP_DIAG(BACKEND, "using Triton GPU compiler backend");
     return gpuGraphBackend_;
   }
   if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON) {
     DSP_DIAG(BACKEND, "Triton backend requested but not available");
     gpuGraphBackend_ = nullptr;
     return nullptr;
   }
   DSP_DIAG(BACKEND, "Triton unavailable in AUTO mode, trying NVRTC/PTX backends");
 }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON) {
    DSP_DIAG(BACKEND, "Triton backend requested but not compiled (HAVE_TRITON=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
  if (graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    DSP_DIAG(BACKEND, "Triton not compiled (HAVE_TRITON=0); AUTO mode will try NVRTC/PTX/CUDA graphs");
  }
#endif

#ifdef SD_CUDA
  if (graphExecutionMode_ == GraphExecutionMode::GEM_NVRTC_JIT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& nvrtc = NvrtcGraphBackend::getInstance();
    if (nvrtc.isAvailable()) {
      gpuGraphBackend_ = &nvrtc;
      DSP_DIAG(BACKEND, "using NVRTC GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_NVRTC_JIT) {
      DSP_DIAG(BACKEND, "NVRTC backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }

  if (graphExecutionMode_ == GraphExecutionMode::GEM_PTX_JIT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& ptx = PtxGraphBackend::getInstance();
    if (ptx.isAvailable()) {
      gpuGraphBackend_ = &ptx;
      DSP_DIAG(BACKEND, "using PTX template GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_PTX_JIT) {
      DSP_DIAG(BACKEND, "PTX backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#endif

#ifdef SD_TPU
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU ||
     graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
   auto& tpu = TpuGraphBackend::getInstance();
   if (tpu.isAvailable()) {
     gpuGraphBackend_ = &tpu;
     DSP_DIAG(BACKEND, "using TPU HLO compiler backend");
     return gpuGraphBackend_;
   }
   if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU) {
     DSP_DIAG(BACKEND, "TPU backend requested but not available");
     gpuGraphBackend_ = nullptr;
     return nullptr;
   }
 }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU) {
    DSP_DIAG(BACKEND, "TPU backend requested but not compiled (SD_TPU=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#ifdef HAVE_HEXAGON_MLIR
  if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON ||
     graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
   auto& hexagon = HexagonGraphBackend::getInstance();
   if (hexagon.isAvailable()) {
     gpuGraphBackend_ = &hexagon;
     DSP_DIAG(BACKEND, "using Hexagon MLIR NPU compiler backend");
     return gpuGraphBackend_;
   }
   if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON) {
     DSP_DIAG(BACKEND, "Hexagon backend requested but not available");
     gpuGraphBackend_ = nullptr;
     return nullptr;
   }
 }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON) {
    DSP_DIAG(BACKEND, "Hexagon backend requested but not compiled (HAVE_HEXAGON_MLIR=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

  gpuGraphBackend_ = nullptr;
  return nullptr;
}

Status NativeDynamicShapePlan::executeSegmentWithGpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  // Derive segIdx for proactive eviction and OOM retry.
  int segIdx = -1;
  for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
    if (&segments_[si] == &seg) { segIdx = si; break; }
  }

  {
    const char* mode = "unknown";
    if (seg.exec.executionCount == 0) mode = "warmup";
    else if (seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady()) mode = "replay";
    else if (seg.exec.compilationFailed) mode = "slot-by-slot(failed)";
    else if (seg.exec.executionCount >= 1) mode = "capture-candidate";
    DSP_DIAG_SEG(SHAPE, seg.def.startSlot,
                 "executeSegmentWithGpuGraph: ENTER seg[%d-%d] mode=%s execCount=%d capturable=%d",
                 seg.def.startSlot, seg.def.endSlot, mode, seg.exec.executionCount, seg.def.isCapturable ? 1 : 0);
  }

#ifdef SD_CUDA
  // ── Segment lifecycle: SEG_ENTER ──────────────────────────────────────
  if (Environment::getInstance().tritonVerifyKernels()) {
    // Ensure VERIFY diagnostic category is enabled and output level is FULL
    // when tritonVerifyKernels is on (may be set at runtime via Java, after
    // DspDiagnostics constructor)
    if (!DSP_DIAG_ENABLED(VERIFY)) {
      sd::graph::DspDiagnostics::getInstance().enableCategories(sd::graph::DSP_DIAG_VERIFY);
      sd::graph::DspDiagnostics::getInstance().setLevel(sd::graph::DSP_LEVEL_FULL);
    }
    const char* mode = "unknown";
    if (seg.exec.executionCount == 0) mode = "warmup";
    else if (seg.exec.executionCount == 1) mode = "compile";
    else if (seg.exec.replayHandle != nullptr) mode = "replay";
    else if (seg.exec.compilationFailed) mode = "slot-by-slot";
    else mode = "capture";
    DSP_DIAG(VERIFY, "SEG_ENTER seg[%d-%d] execCount=%d mode=%s",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount, mode);
    // Dump external input actuality flags for first N inputs
    int detailLimit = sd::graph::DspDiagnostics::getInstance().diagDetailLimit();
    int dumpCount = std::min(numExt, detailLimit);
    for (int i = 0; i < dumpCount; i++) {
      if (externalArrays[i] != nullptr && externalArrays[i]->dataBuffer() != nullptr) {
        auto* db = externalArrays[i]->dataBuffer();
        DSP_DIAG(VERIFY, "  EXT_INPUT[%d] dtype=%s len=%lld pAct=%d sAct=%d addr=%p",
                 i, DataTypeUtils::asString(externalArrays[i]->dataType()).c_str(),
                 (long long)externalArrays[i]->lengthOf(),
                 db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0,
                 DSP_BUF(externalArrays[i]));
      }
    }
    if (numExt > detailLimit) {
      DSP_DIAG(VERIFY, "  ... and %d more external inputs", numExt - detailLimit);
    }
  }
#endif

  auto* backend = getGpuGraphBackend();
  if (backend == nullptr) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: no GPU backend selected for seg[%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }
  const char* backendName = backend->name();
#if HAVE_TRITON
  auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
#else
  void* tritonBackend = nullptr;
#endif

  // If compilation previously failed validation, never try again
  if (seg.exec.compilationFailed) {
    return Status::KERNEL_FAILURE;
  }

  // Check if this segment can be compiled by the selected GPU backend
  if (!backend->canFuseSegment(slots_, seg.def.startSlot, seg.def.endSlot)) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: backend=%s cannot fuse seg[%d-%d]",
             backendName, seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;  // Caller will fall back to CUDA Graphs
  }

  // First execution: run slot-by-slot warmup BEFORE compilation.
  if (seg.exec.executionCount == 0) {
#ifdef SD_CUDA
    // ── Plan structure dump (one-time, on first segment execution) ─────────
    if (Environment::getInstance().tritonVerifyKernels()) {
      DSP_DIAG(VERIFY, "=== PLAN STRUCTURE ===");
      DSP_DIAG(VERIFY, "Plan: %d steps, %d output slots, %d external inputs, %d segments",
               numSlots_, totalOutputSlots_, numExternalInputs_, (int)segments_.size());
      for (int si = 0; si < (int)segments_.size(); si++) {
        auto& s = segments_[si];
        DSP_DIAG(VERIFY, "Segment %d: slots [%d..%d] (%d ops) %s",
                 si, s.def.startSlot, s.def.endSlot, s.def.endSlot - s.def.startSlot + 1,
                 s.def.isCapturable ? "capturable" : "non-capturable");
      }
      // Per-step wiring
      std::unordered_map<std::string, int> opHistogram;
      for (int s = 0; s < numSlots_; s++) {
        auto& sl = slots_[s];
        opHistogram[sl.ident.opName]++;
        // Build input description
        std::string inputsStr;
        for (int i = 0; i < sl.wiring.numInputs; i++) {
          if (i > 0) inputsStr += ", ";
          int srcIdx = sl.wiring.inputSourceIndices[i];
          if (srcIdx >= 0) {
            inputsStr += "slot#" + std::to_string(srcIdx);
          } else {
            int extIdx = -(srcIdx + 1);
            inputsStr += "ext#" + std::to_string(extIdx);
            if (extIdx < (int)externalInputNames_.size() && !externalInputNames_[extIdx].empty()) {
              inputsStr += ":\"" + externalInputNames_[extIdx] + "\"";
            }
            if (sl.wiring.inputSourceTypes != nullptr) {
              inputsStr += ":";
              inputsStr += sourceTypeName(sl.wiring.inputSourceTypes[i]);
            }
          }
        }
        // Build output description
        std::string outputsStr;
        for (int i = 0; i < sl.wiring.numOutputs; i++) {
          if (i > 0) outputsStr += ",";
          outputsStr += std::to_string(sl.wiring.outputSlotIndices[i]);
        }
        DSP_DIAG(VERIFY, "STEP %4d: %-20s inputs:[%s] -> outputs:[%s]%s%s%s",
                 s, sl.ident.opName.c_str(), inputsStr.c_str(), outputsStr.c_str(),
                 sl.flags.isIdentityOp ? " [IDENTITY]" : "",
                 sl.frozenConstantSlot() ? " [FROZEN]" : "",
                 sl.fusedChain.isFusedChainTail ? " [FUSED_TAIL]" : "");
      }
      // Op histogram
      std::string histStr;
      std::vector<std::pair<std::string, int>> sorted(opHistogram.begin(), opHistogram.end());
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return b.second < a.second; });
      for (auto& p : sorted) {
        if (!histStr.empty()) histStr += ", ";
        histStr += p.first + "=" + std::to_string(p.second);
      }
      DSP_DIAG(VERIFY, "Op histogram: %s", histStr.c_str());
      DSP_DIAG(VERIFY, "=== END PLAN STRUCTURE ===");
    }
#endif

    // Pre-warmup promotion: view-capable slots should enter FROZEN state BEFORE
    // warmup execution when shapes are already frozen. This ensures they take the
    // view installation path (which shares the input's DataBuffer) instead of
    // allocating new output buffers. Without this, gap slots materialize separate
    // buffers that later get H2D-copied during capture, corrupting downstream values.
    if (shapesFrozen_) {
      int promoted = 0;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        auto& sl = slots_[s];
        if (!sl.flags.isViewCapableOp || sl.state_ >= NativeSlot::SlotState::FROZEN) continue;
        sl.state_ = NativeSlot::SlotState::FROZEN;
        promoted++;
      }
      if (promoted > 0) {
        DSP_DIAG(EXECUTE, "pre-warmup view promotion: %d view-capable slots promoted to FROZEN for seg[%d-%d]",
                 promoted, seg.def.startSlot, seg.def.endSlot);
      }
    }

    auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    // NOTE: executeSegmentSlotBySlot already increments seg.exec.executionCount on OK
    // (NativeDynamicShapePlan_segments.cpp line 930). Do NOT increment again here
    // — double-increment causes executionCount to skip the capture window [0,2],
    // preventing CUDA graph capture entirely and causing OOM from leaked per-step
    // allocations.

    // When shapes are frozen and executionCount is 1, the next call would
    // trigger compilation (needsCompile = executionCount==1). If compilation
    // already succeeded during unfrozen execution, skip recompilation by
    // bumping executionCount to 2. But DON'T skip if compilation hasn't
    // happened yet — let it trigger on the next call so cross-segment
    // shapes (now backfilled) are available for the Triton IR builder.
    if (shapesFrozen_ && warmupStatus == Status::OK && seg.exec.executionCount == 1
        && !Environment::getInstance().dspFreezeRecompile()) {
      // Only skip recompilation if segment already has compiled kernels.
      // seg.def.shapeKey != 0 means compilation ran previously and cached the key.
      if (seg.def.shapeKey != 0) {
        seg.exec.executionCount = 2;
        seg.exec.cachedShapeKey = seg.def.shapeKey;
        DSP_DIAG(COMPILE, "Post-freeze warmup: skipping recompile for seg[%d-%d] "
                          "(already compiled, shapeKey=%lld, bumped executionCount to 2)",
                 seg.def.startSlot, seg.def.endSlot, seg.def.shapeKey);
      } else {
        // Segment was never compiled — let executionCount stay at 1 so the
        // next call triggers compilation with backfilled cross-segment shapes.
        // DO NOT set seg.def.shapeKey here — it must stay 0 so the next call's
        // shapeKey check correctly identifies this as "never compiled".
        DSP_DIAG(COMPILE, "Post-freeze warmup: NOT skipping compile for seg[%d-%d] "
                          "(never compiled, executionCount stays at 1)",
                 seg.def.startSlot, seg.def.endSlot);
      }
    }
    return warmupStatus;
  }

  // Compute shape key for cache lookup.
  // When shapes are frozen and the key was already computed, reuse it — the shapes
  // cannot change so the hash is stable. Saves iterating all cross-segment inputs.
  // EXCEPTION: segments with value-dependent ops must ALWAYS recompute the shape key
  // because input VALUES (hashed by computeSegmentShapeKey for small inputs ≤32 elements)
  // can change even when shapes are frozen. Without this guard, the cached key would
  // miss value changes in reshape targets, broadcast dims, etc., causing CUDA graph
  // replay with stale output shapes.
  //
  // REPLAY OPTIMIZATION: During stable replay (executionCount >= 3 with a valid replay
  // handle), skip shape key computation entirely — even for hasValueDepOps segments.
  // The shape key was validated at capture time. Value-dependent inputs that can force
  // a rebuild are tracked separately via createValueKey/address stability checks.
  // If a value change truly requires graph invalidation, the createValueKey mechanism
  // catches it. Skipping shape key here eliminates N syncToHost calls per step
  // (one per small INT/INT64 cross-segment input array).
  // ── Shape key: detect if segment needs recompilation ──
  // Frozen + cached key: reuse. Otherwise: compute once and cache.
  const bool hasInternalValueShapeInputs = segmentHasInternalValueShapeInputs(seg, slots_);
  LongType segShapeKey;
  if (shapesFrozen_ && seg.exec.cachedShapeKey != 0) {
    segShapeKey = seg.exec.cachedShapeKey;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    if (shapesFrozen_) {
      seg.exec.cachedShapeKey = segShapeKey;
    }
  }

  // Diagnostic: scan all outputSlots_ entries for freed DataBuffers.
  // Java may have closed DSP output arrays between steps (e.g., prefill KV outputs via
  // setCloseable(true)+close()), deleting the C++ NDArray and leaving dangling pointers.
  //
  // Always run this scan: during warmup/transitions it handles invalidation gracefully;
  // after warmup with frozen shapes, stale buffers indicate a bug (hard error via REQUIRE_TRUE).
  //
  // REPLAY OPTIMIZATION: Skip during stable replay (executionCount >= 4). In frozen
  // replay, arrays persist and are never closed by Java. The scan iterates all slots
  // in the segment range + all external inputs (~1333). For 278 captured segments,
  // this is significant host-side iteration. After the first few replays validate
  // no stale entries exist, skip the scan.
  bool isStableReplay = shapesFrozen_ && seg.exec.cachedShapeKey != 0 &&
                        seg.exec.executionCount >= 3;

  // View-producer slots that wrap a placeholder DataBuffer are legitimately
  // stale whenever SameDiff replaces the placeholder between calls (e.g.
  // EMULATED_REPLAY supplies a fresh external input every step). Refresh
  // those wrappers in place on EVERY frozen replay — the gate below only
  // controls the expensive stale-buffer scan, but view-wrapper refresh must
  // always run or the slot's DataBuffer will dangle into slot-by-slot exec,
  // where writeOutputSlot's frozen-phase guard rejects the replacement as a
  // lifecycle violation.
  if (shapesFrozen_ && slotIsViewProducer_ != nullptr) {
    int viewRefreshResult =
        refreshStaleViewWrappersInSegment(seg, externalArrays, numExt);
    if (viewRefreshResult > 0) {
      // Fresh wrappers expose new device addresses — force argTable refresh on
      // the next replay. Graph remains valid; no recapture needed.
      seg.exec.argTableStable = false;
      DSP_DIAG(MEMORY,
               "executeSegmentWithGpuGraph: refreshed %d stale view wrappers in seg[%d-%d]",
               viewRefreshResult, seg.def.startSlot, seg.def.endSlot);
    }
  }

  if (seg.exec.executionCount < 4 || !isStableReplay) {
    int invalidCount = 0;
    for (int si = seg.def.startSlot; si <= seg.def.endSlot && si < totalOutputSlots_; si++) {
      NDArray* cached = outputSlots_[si];
      if (cached != nullptr && !cached->isEmpty()) {
        auto* db = cached->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG_SLOT(MEMORY, si, "STALE outputSlots_[%d] detected "
                                    "(arr=%p, db=%p, dbValid=%d, frozenConst=%d). Invalidating.",
                        si, (void*)cached, (void*)db, db ? (db->isValid() ? 1 : 0) : -1,
                        slots_[si].frozenConstantSlot() ? 1 : 0);
          outputSlots_[si] = nullptr;
          if (si < numSlots_ && slots_[si].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
            slots_[si].state_ = NativeSlot::SlotState::FROZEN;
          }
          invalidCount++;
        }
      }
    }
    for (int ei = 0; ei < numExt; ei++) {
      NDArray* ext = externalArrays[ei];
      if (ext != nullptr && !ext->isEmpty()) {
        auto* db = ext->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG(MEMORY, "STALE externalInput[%d] detected "
                           "(arr=%p, db=%p, dbValid=%d)",
                   ei, (void*)ext, (void*)db, db ? (db->isValid() ? 1 : 0) : -1);
          invalidCount++;
        }
      }
    }
    if (invalidCount > 0) {
      DSP_DIAG(MEMORY, "executeSegmentWithGpuGraph: found %d stale entries in slot/external arrays",
               invalidCount);
      if (shapesFrozen_ && seg.exec.executionCount > 1) {
        // After warmup with frozen shapes, stale buffers mean a bug in array lifecycle management
        REQUIRE_TRUE(false, 0, "Stale buffer detected after warmup (executionCount=%d, frozen=%d, "
                               "invalidCount=%d) in seg[%d-%d]. This indicates a bug in DSP array persistence.",
                     seg.exec.executionCount, (int)shapesFrozen_, invalidCount,
                     seg.def.startSlot, seg.def.endSlot);
      }
      // During warmup/transitions, invalidate and re-execute
#ifdef SD_CUDA
      platformCleanupSegmentForRebuild(seg);
      seg.exec.argTableStable = false;
      batchD2DCount_ = 0;
      seg.exec.cachedShapeKey = 0;
#endif
      seg.exec.compilationFailed = false;
      DSP_DIAG(FALLBACK, "invalidated graph for seg[%d-%d] "
                         "due to %d stale entries - executing slot-by-slot this step",
               seg.def.startSlot, seg.def.endSlot, invalidCount);
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  // Pre-execution: ensure all output slots in the segment have live arrays.
  // The Triton kernel's arg mapping references outputSlots_ for both inputs
  // (from prior ops) and outputs (to write results). Slot-by-slot warmup may
  // have released intermediate arrays via releaseAtStep_, leaving entries null.
  // First restore from outputSlots_, then allocate any remaining nulls
  // using cached shape info from warmup.
  //
  //  This MUST happen BEFORE compilation. The compiler resolves
  // arg mappings from outputSlots_ — if intermediate slots are null (released
  // after warmup), the compiler omits them from the arg table, producing
  // sub-kernels with missing inputs that read stale/garbage data on first
  // execution. By populating all slots before compilation, the compiler sees
  // all arrays and builds correct arg mappings.
  //
  // IMPORTANT: Java may close() output arrays between execution steps (e.g.,
  // prefill KV outputs via setCloseable(true)+close()). This frees the underlying
  // DataBuffer while outputSlots_ still holds the NDArray*. Validate the
  // DataBuffer before reusing — invalidate entries pointing to freed buffers.
  //
  //  If any output slot within the segment is allocated at a NEW address
  // (different from capture time), the cached CUDA graph becomes invalid. Triton
  // arg tables are refreshed with new addresses, but native ops (cuBLAS matmul)
  // have addresses baked into the graph. This address inconsistency causes the
  // graph to read stale data from old addresses while Triton writes to new ones.
  // Track any new allocations and invalidate the graph if needed.
  //
  // OPTIMIZATION: Skip when shapes are frozen and we've already done this
  // restoration at least once (executionCount > 2). In steady-state decode,
  // outputSlots_ are stable — no arrays are released or freed between steps.
  // EXCEPTION: segments entering capture for the first time (no replay handle)
  // MUST always get pre-exec restoration — cleanup may have nulled cross-segment
  // input slots that the capture path still needs wired directly.
  int preExecAllocCount = 0;
  if (!(shapesFrozen_ && seg.exec.executionCount > 2 && seg.exec.replayHandle != nullptr)) {
    for (int stepIdx = seg.def.startSlot; stepIdx <= seg.def.endSlot; stepIdx++) {
      NativeSlot& slot = slots_[stepIdx];
      // Phase 2: outputSlots_ == outputSlots_ (unified). No restore needed.
      // Validate input DataBuffers — Java close() may have freed them.
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          auto* db = outputSlots_[srcIdx]->dataBuffer();
          if (db == nullptr || !db->isValid()) {
            outputSlots_[srcIdx] = nullptr;
            if (srcIdx < numSlots_ && slots_[srcIdx].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
              slots_[srcIdx].state_ = NativeSlot::SlotState::FROZEN;
            }
          }
        }
      }
      // Validate or allocate output slot entries
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int slotIdx = slot.wiring.outputSlotIndices[i];
        if (slotIdx < 0 || slotIdx >= totalOutputSlots_) continue;
        // DIAGNOSTIC: trace configured slot pre-exec validation (ND4J_DSP_TRACE_SLOT)
        {
          int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
          if (ts >= 0 && slotIdx == ts && shapesFrozen_) {
            auto* arr = outputSlots_[slotIdx];
            auto* db = arr != nullptr ? arr->dataBuffer() : nullptr;
            DSP_DIAG_SLOT(MEMORY, stepIdx,
                          "PRE_EXEC_VALIDATE: slot=%d arr=%p db=%p valid=%d exec=%d",
                          slotIdx, (void*)arr, (void*)db,
                          db != nullptr && db->isValid() ? 1 : 0,
                          seg.exec.executionCount);
          }
        }
        // Validate existing entry
        if (outputSlots_[slotIdx] != nullptr) {
          auto* db = outputSlots_[slotIdx]->dataBuffer();
          if (db == nullptr || !db->isValid()) {
            {
              int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
              if (ts >= 0 && slotIdx == ts) {
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                              "PRE_EXEC_NULL: slot=%d db=%p was nullOrInvalid exec=%d",
                              slotIdx, (void*)db, seg.exec.executionCount);
              }
            }
            outputSlots_[slotIdx] = nullptr;
            if (stepIdx < numSlots_ && slots_[stepIdx].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
              slots_[stepIdx].state_ = NativeSlot::SlotState::FROZEN;
            }
          }
        }
        if (outputSlots_[slotIdx] == nullptr) {
          // After warmup with frozen shapes, null output slots indicate a persistence bug.
          // Frozen constant slots are exempt (they never allocate output arrays).
          // Warn but continue — the allocation path below will recover.
          if (shapesFrozen_ && seg.exec.executionCount > 1 && !slot.frozenConstantSlot()) {
            DSP_DIAG_SLOT(VERIFY, slotIdx,
                          "BUG: Null output slot %d (%s) after warmup with frozen shapes — persistence bug. execCount=%d",
                          slotIdx, slot.ident.opName.c_str(), seg.exec.executionCount);
          }
          // Phase assertion: allocating a new NDArray during REPLAYING phase is a bug.
          // Output slots should already be populated from warmup/capture. New allocations
          // during replay mean the slot was freed or not persisted correctly.
#ifndef NDEBUG
          if (seg.exec.currentPhase == ExecutionPhase::REPLAYING && !slot.frozenConstantSlot()) {
            DSP_DIAG(FALLBACK, "PHASE_VIOLATION: new NDArray allocation for slot %d (%s) during "
                               "REPLAYING phase — output should already exist from warmup. "
                               "seg[%d-%d] execCount=%d planPhase=%s",
                     slotIdx, slot.ident.opName.c_str(),
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                     dsp::planPhaseName(planPhase_));
            assert(false && "New NDArray allocation during REPLAYING phase");
          }
#endif
          // Allocate from cached shape info (populated during warmup)
          const LongType* shapeInfo = nullptr;
          if (i < static_cast<int>(slot.shapeCache.cachedOutputShapes.size()) && slot.shapeCache.cachedOutputShapes[i]) {
            shapeInfo = slot.shapeCache.cachedOutputShapes[i];
          }
          // For identity/view-like ops that don't cache output shapes,
          // derive the shape from the first input source's existing array
          if (!shapeInfo && slot.wiring.numInputs > 0) {
            int srcIdx = slot.wiring.inputSourceIndices[0];
            NDArray* srcArr = nullptr;
            if (srcIdx < 0) {
              int extIdx = -(srcIdx + 1);
              if (extIdx < numExt) srcArr = externalArrays[extIdx];
            } else if (srcIdx < totalOutputSlots_) {
              srcArr = outputSlots_[srcIdx];
              // Phase 2: outputSlots_ == outputSlots_ (unified), no separate restore
            }
            if (srcArr) shapeInfo = srcArr->shapeInfo();
          }
          if (shapeInfo) {
            auto dt = ArrayOptions::dataType(shapeInfo);
            // For cast ops, the output type must match the declared target type,
            // not the input type. When cachedOutputShapes is empty and the
            // using the input source's shape, the dtype would be wrong
            // (e.g., INT64 input for a cast-to-FLOAT op).
            if (slot.ident.op && slot.ident.op->getOpDescriptor() &&
                slot.ident.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_CAST) &&
                slot.args.numIArgs > 0 && slot.args.iArgs) {
              auto castDt = static_cast<DataType>(slot.args.iArgs[0]);
              if (castDt != dt) {
                DSP_DIAG(EXECUTE, "PRE_EXEC_ALLOC: cast dtype override slot=%d from %s to %s",
                         slotIdx, DataTypeUtils::asString(dt).c_str(),
                         DataTypeUtils::asString(castDt).c_str());
                dt = castDt;
              }
            }
            auto order = shape::order(shapeInfo);
            LongType rank = shape::rank(shapeInfo);
            std::vector<LongType> shapeVec(rank);
            for (int d = 0; d < rank; d++) shapeVec[d] = shapeInfo[d + 1];
            auto* arr = new NDArray(order, shapeVec, dt);
            outputSlots_[slotIdx] = arr;
            // Phase 2: outputSlots_ == outputSlots_ (unified), no separate assignment needed
            preExecAllocCount++;
            {
              int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
              if (ts >= 0 && slotIdx == ts) {
                auto* newDb = arr != nullptr ? arr->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                              "PRE_EXEC_ALLOC: slot=%d arr=%p db=%p exec=%d",
                              slotIdx, (void*)arr, (void*)newDb, seg.exec.executionCount);
              }
            }
            if (Environment::getInstance().tritonVerifyKernels()) {
              DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=ALLOC dtype=%s len=%lld addr=%p",
                       slotIdx, DataTypeUtils::asString(dt).c_str(),
                       (long long)arr->lengthOf(), DSP_BUF(arr));
            }
          }
        }
      }
    }
  } // end if (!(shapesFrozen_ && executionCount > 2))

  // Compile once per stable shape; skip cache probe on steady-state replay.
  // This keeps the hot path focused on dispatch instead of repeated compile checks.
  // NOTE: Pre-exec output slot allocation above ensures all slots are populated
  // before the compiler resolves arg mappings. Without this ordering, intermediate
  // slots released after warmup are null and get omitted from the arg table,
  // causing sub-kernels to read stale data on their first execution.
  bool needsCompile = (seg.exec.executionCount == 1) || (seg.def.shapeKey != segShapeKey);

  // ── Phase guard: compilation must not happen during REPLAYING ────────────
  // If compilation triggers during steady-state replay, it means something
  // invalidated the compiled state without properly demoting the plan phase.
  // This is always a bug — the plan should have been reset to an earlier phase
  // before arriving here. Log a loud error and assert in debug builds.
  if (needsCompile && planPhase_ >= PlanPhase::REPLAYING) {
    DSP_DIAG(COMPILE,
             "ERROR: compilation triggered during REPLAYING phase for seg[%d-%d] "
             "(executionCount=%d, shapeKey cached=%lld current=%lld, planPhase=%d). "
             "Compilation must only happen during warmup/capture phases. "
             "This indicates a phase management bug — plan should have been demoted "
             "before reaching this code path.",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
             (long long)seg.def.shapeKey, (long long)segShapeKey,
             static_cast<int>(planPhase_));
    // In debug builds, fail fast so the root cause is caught during development
#ifndef NDEBUG
    REQUIRE_TRUE(false, 0,
                 "DSP phase contract violation: compilation triggered during REPLAYING phase "
                 "for seg[%d-%d] (executionCount=%d). Fix the phase management bug.",
                 seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
#endif
    // In release builds, demote the plan phase and continue so production doesn't crash
    demotePlanPhase(PlanPhase::POINTERS_STABLE,
                    "compilation triggered during REPLAYING phase");
  }

  if (needsCompile) {
    // When recompiling due to shape change (not the first compile), outputSlots_
    // has stale shapes from the previous execution. The compiler reads these shapes
    // to derive kernel parameters (e.g., seqQ/seqK for FUSED_ATTENTION). Run a
    // slot-by-slot pass first to populate outputSlots_ with current shapes before
    // compiling. This is like a mini-warmup for the new shape configuration.
    bool isRecompileDueToShapeChange = (seg.exec.executionCount > 1) && (seg.def.shapeKey != segShapeKey);
    if (isRecompileDueToShapeChange) {
      // Mid-execution compile: benchmarks care about this. Record it via the
      // plan-level violation counter so callers can assert "0 mid-exec compiles"
      // across a measured window. recordMidExecutionCompile() emits a loud
      // [COMPILE_VIOLATION] log regardless of DSP diagnostics level.
      char reasonBuf[128];
      std::snprintf(reasonBuf, sizeof(reasonBuf),
                    "shape-change recompile (shapeKey %lld->%lld, executionCount=%d)",
                    (long long)seg.def.shapeKey, (long long)segShapeKey,
                    seg.exec.executionCount);
      recordMidExecutionCompile(seg.def.startSlot, seg.def.endSlot, reasonBuf);
      // Temporarily unseal so phaseCompile-gated paths can recompile this
      // segment. The seal is restored below via a direct compileSegment call
      // (the plan-level compilationDone_ flag itself stays set so that other
      // segments remain sealed; only this segment's compile is allowed).
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape change detected for seg[%d-%d] "
                        "(shapeKey %lld->%lld, executionCount=%d). Running slot-by-slot warmup to "
                        "refresh outputSlots_ before recompilation.",
               seg.def.startSlot, seg.def.endSlot, seg.def.shapeKey, segShapeKey, seg.exec.executionCount);
      // Invalidate cached graph — addresses and shapes changed
#ifdef SD_CUDA
      platformCleanupSegmentForRebuild(seg);
      seg.exec.argTableStable = false;
      batchD2DCount_ = 0;
#endif
      auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
      if (warmupStatus != Status::OK) {
        DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape-change warmup FAILED for seg[%d-%d] status=%d",
                 seg.def.startSlot, seg.def.endSlot, static_cast<int>(warmupStatus));
        return warmupStatus;
      }
      // Recompute shape key after warmup — outputSlots_ now has correct shapes
      segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape-change warmup OK for seg[%d-%d], "
                        "recomputed shapeKey=%lld", seg.def.startSlot, seg.def.endSlot, segShapeKey);
    }

    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: backend=%s compile failed for seg[%d-%d]",
               backendName, seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
  }

  // On first compilation, validate coverage
  if (seg.exec.executionCount == 1) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    int compiledCount = 0;
    int failedCount = 0;
    for (const auto& entry : audit) {
      if (entry.wasCompiled) {
        compiledCount++;
      } else {
        failedCount++;
        DSP_DIAG_SLOT(COMPILE, entry.slotIndex, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                      backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (compiledCount == 0 && failedCount > 0) {
      // All ops FAILED compilation — hard error.
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] has zero compiled ops "
                        "(failed=%d). Compilation failures are errors, not fallbacks.",
               backendName, seg.def.startSlot, seg.def.endSlot, failedCount);
      seg.exec.compilationFailed = true;
      return Status::KERNEL_FAILURE;
    }
    if (compiledCount == 0 && failedCount == 0) {
      // All sections stay native-ordered for this segment.
      // The compiled segment has 0 sub-kernels; executeSegment will run
      // everything via the ordered range executor.
      // DO NOT set compilationFailed — allow these segments to be captured as
      // CUDA graphs. During Triton graph capture, native ordered ranges are
      // recorded into the graph via the ordered range executor, enabling
      // single-launch replay instead of per-op kernel dispatch overhead.
      DSP_DIAG(COMPILE, "%s: segment [%d-%d] has only native ordered sections (no Triton kernels needed). "
                        "Segment remains eligible for CUDA graph capture.",
               backendName, seg.def.startSlot, seg.def.endSlot);
    }
    if (failedCount > 0) {
      // Partial compilation failure — hard error. Fix the kernel.
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] partial compile FAILED "
                        "(compiled=%d failed=%d). Compilation failures are errors, not fallbacks.",
               backendName, seg.def.startSlot, seg.def.endSlot, compiledCount, failedCount);
      seg.exec.compilationFailed = true;
      return Status::KERNEL_FAILURE;
    }
  }

  // Execute via selected GPU backend
  seg.def.shapeKey = segShapeKey;

#ifdef SD_CUDA
  cudaStream_t cudaStr = (stream != nullptr)
                         ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // If any output slots were re-allocated at new addresses, the cached CUDA graph
  // is invalid — native ops (cuBLAS) have the old addresses baked in while Triton
  // arg tables were refreshed with new addresses. Invalidate and re-capture.
  if (preExecAllocCount > 0 && seg.exec.replayHandle != nullptr) {
    DSP_DIAG(EXECUTE, "GRAPH INVALIDATED: %d output slots re-allocated at new addresses "
                      "(cache entries freed by Java). seg[%d-%d] will re-capture.",
             preExecAllocCount, seg.def.startSlot, seg.def.endSlot);
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    // Reset execution count to trigger warmup→capture flow
    seg.exec.executionCount = 0;
    seg.exec.compilationFailed = false;
  }

  // CUDA graph capture is BLOCKED when tritonSkipKernels=true. With all Triton
  // kernels skipped, the captured graph contains only native ops executed via
  // orderedRangeExecutor_. During capture, syncToSpecial() records H2D memcpy
  // nodes that bake the capture-step's external input data (attention_mask,
  // position_ids, KV scatter indices) into the graph. On replay, these H2D
  // nodes overwrite freshly-synced device data with stale capture-step values,
  // producing identical outputs (stuck token) for all subsequent decode steps.
  bool allowTritonCudaGraphReplay = Environment::getInstance().tritonGraphCapture() &&
                                    shapesFrozen_ &&
                                    !Environment::getInstance().tritonSkipKernels();

  // BLATANT DIAGNOSTIC: Log the capture decision factors
  int captureMinExec = Environment::getInstance().tritonCaptureMinExec();
  bool forceRecaptureEnabled = Environment::getInstance().tritonForceRecapture();
  bool hasReplayHandle = (seg.exec.replayHandle != nullptr);
  bool replayHandleNull = (seg.exec.replayHandle == nullptr);
  bool notCaptureFailed = !seg.exec.compilationFailed;
  bool execCountInWindow = (seg.exec.executionCount >= captureMinExec) &&
                           (forceRecaptureEnabled || seg.exec.executionCount <= (captureMinExec + 2));
  bool hasCudaStream = (cudaStr != nullptr);
  bool requiresOrderedGapCapture = false;

  DSP_DIAG(EXECUTE, "=== CAPTURE DECISION CHECK seg[%d-%d] ===", seg.def.startSlot, seg.def.endSlot);
  DSP_DIAG(EXECUTE, "  tritonGraphCapture()=%d, shapesFrozen_=%d, tritonSkipKernels=%d => allowTritonCudaGraphReplay=%d",
           Environment::getInstance().tritonGraphCapture() ? 1 : 0,
           shapesFrozen_ ? 1 : 0,
           Environment::getInstance().tritonSkipKernels() ? 1 : 0,
           allowTritonCudaGraphReplay ? 1 : 0);
  DSP_DIAG(EXECUTE, "  seg.exec.executionCount=%d, captureMinExec=%d, window=[%d,%d], inWindow=%d",
           seg.exec.executionCount, captureMinExec, captureMinExec, captureMinExec + 2,
           execCountInWindow ? 1 : 0);
  DSP_DIAG(EXECUTE, "  hasReplayHandle=%d, replayHandleNull=%d",
           hasReplayHandle ? 1 : 0, replayHandleNull ? 1 : 0);
  DSP_DIAG(EXECUTE, "  compilationFailed=%d, cudaStr!=nullptr=%d",
           seg.exec.compilationFailed ? 1 : 0, hasCudaStream ? 1 : 0);

  bool shouldCaptureTritonGraph = false;

  int tritonGapSlotCount = 0;
#if HAVE_TRITON && defined(SD_CUDA)
  // ── Unified composite replay: ALWAYS build a composite schedule ──
 // Every Triton-compiled segment uses composite replay. Segments with no gaps
 // get a single TRITON_ISLAND unit (functionally identical to monolithic replay).
 // Segments with gaps get interleaved TRITON_ISLAND + GAP units. This eliminates
 // the monolithic replay path and the broken view recipe system entirely.
 if (tritonBackend != nullptr) {
   if (seg.exec.compositeReplaySchedule.units.empty()) {
     seg.exec.compositeReplaySchedule = buildCompositeReplaySchedule(seg, slots_, tritonBackend);
     DSP_DIAG(SHAPE, "COMPOSITE_SCHEDULE_BUILT: seg[%d-%d] units=%d",
              seg.def.startSlot, seg.def.endSlot,
              static_cast<int>(seg.exec.compositeReplaySchedule.units.size()));
   }
   auto gapSlots = tritonBackend->getGapSlots(seg, slots_);
   tritonGapSlotCount = static_cast<int>(gapSlots.size());
   if (!gapSlots.empty()) {
     requiresOrderedGapCapture = true;
   }
 }
#else
  seg.exec.compositeReplaySchedule = ReplaySchedule();
#endif

  bool captureWindowSatisfied = execCountInWindow || requiresOrderedGapCapture;
  shouldCaptureTritonGraph = allowTritonCudaGraphReplay &&
                             !hasReplayHandle &&
                             replayHandleNull &&
                             notCaptureFailed &&
                             captureWindowSatisfied &&
                             hasCudaStream;

  if (requiresOrderedGapCapture) {
    DSP_DIAG(EXECUTE,
             "COMPOSITE_GAP_CAPTURE: seg[%d-%d] has %d gap slots. "
             "Gap ops will be EXCLUDED from CUDA graph; "
             "composite replay will execute gaps fresh before Triton-only graph replay.",
             seg.def.startSlot, seg.def.endSlot, tritonGapSlotCount);
  }

  DSP_DIAG(EXECUTE, "  => shouldCaptureTritonGraph=%d", shouldCaptureTritonGraph ? 1 : 0);
  if (!shouldCaptureTritonGraph) {
    if (!allowTritonCudaGraphReplay)
      DSP_DIAG(EXECUTE, "  BLOCKED: allowTritonCudaGraphReplay=false (tritonGraphCapture=%d OR shapesFrozen_=%d OR tritonSkipKernels=%d)",
               Environment::getInstance().tritonGraphCapture() ? 1 : 0, shapesFrozen_ ? 1 : 0,
               Environment::getInstance().tritonSkipKernels() ? 1 : 0);
    if (!replayHandleNull)
      DSP_DIAG(EXECUTE, "  BLOCKED: replayHandle already exists (capture already done or in progress)");
    if (seg.exec.compilationFailed)
      DSP_DIAG(EXECUTE, "  BLOCKED: compilationFailed=true (previous capture failed, warmup path only)");
    if (!captureWindowSatisfied)
      DSP_DIAG(EXECUTE, "  BLOCKED: executionCount=%d outside capture window [%d,%d]",
               seg.exec.executionCount, captureMinExec, captureMinExec + 2);
    if (!hasCudaStream)
      DSP_DIAG(EXECUTE, "  BLOCKED: cudaStr=nullptr (no CUDA stream available)");
  } else {
    DSP_DIAG(EXECUTE, "  >>> CAPTURE WILL BE ATTEMPTED <<<");
  }
  DSP_DIAG(EXECUTE, "=== END CAPTURE DECISION CHECK ===");

  // NOTE: shouldCaptureTritonGraph is ONLY checked when we don't have a captured graph.
  // Once captured, we use useFastReplay based on argTableStable, not executionCount.
  // The executionCount window check prevents repeated capture attempts after success.

  // OPTIMIZATION: When argTableStable, addresses and create-op values haven't changed
  // since last refresh — skip the expensive hash/comparison loops over all external inputs.
  LongType segInputAddrKey;
  bool extAddrsStable;
  LongType createValueKey;
  bool canSkipReplayInvariantRecompute =
      seg.exec.argTableStable && allowTritonCudaGraphReplay &&
      !hasInternalValueShapeInputs;
  if (canSkipReplayInvariantRecompute) {
    // Fast path: arg table is stable, all addresses are known-good
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "seg[%d-%d] argTableStable=true → FAST PATH (skip addr/createValue recompute)",
                 seg.def.startSlot, seg.def.endSlot);
    segInputAddrKey = seg.exec.capturedInputAddrKey;
    extAddrsStable = true;
    createValueKey = seg.exec.capturedCreateValueKey;
  } else {
    segInputAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);
    extAddrsStable = (seg.exec.replayHandle && !seg.exec.replayHandle->getCapturedExternalAddresses().empty())
                     ? externalAddrsMatch(seg, externalArrays, numExt)
                     : (seg.exec.capturedInputAddrKey != 0 && seg.exec.capturedInputAddrKey == segInputAddrKey);
    createValueKey = computeCreateOpValueKey(seg, externalArrays, numExt);
    DSP_DIAG(EXECUTE, "ADDR_CHECK_SLOW: seg[%d-%d] extAddrsStable=%d addrKey=%lld (cached=%lld)",
             seg.def.startSlot, seg.def.endSlot, extAddrsStable ? 1 : 0,
             (long long)segInputAddrKey, (long long)seg.exec.capturedInputAddrKey);
  }
  bool createValuesStable = (createValueKey == 0) ||  // no create ops
                            (seg.exec.capturedCreateValueKey == createValueKey);
  if (hasInternalValueShapeInputs) {
    const bool shapeKeyStable = (seg.exec.cachedShapeKey == 0) ||
                                (seg.exec.cachedShapeKey == segShapeKey);
    seg.exec.argTableStable = seg.exec.argTableStable &&
                              extAddrsStable &&
                              createValuesStable &&
                              shapeKeyStable;
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "INTERNAL_VALUE_SHAPE_TRACKING: seg[%d-%d] argStable=%d shapeStable=%d "
                 "createStable=%d extAddrsStable=%d",
                 seg.def.startSlot, seg.def.endSlot,
                 seg.exec.argTableStable ? 1 : 0,
                 shapeKeyStable ? 1 : 0,
                 createValuesStable ? 1 : 0,
                 extAddrsStable ? 1 : 0);
  }
  if (!createValuesStable && seg.exec.replayHandle) {
    DSP_DIAG(EXECUTE, "CREATE_VALUE_KEY mismatch: captured=%lld current=%lld → invalidating graph seg[%d-%d]",
             (long long)seg.exec.capturedCreateValueKey, (long long)createValueKey, seg.def.startSlot, seg.def.endSlot);
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.executionCount = 0;
    seg.exec.compilationFailed = false;
    extAddrsStable = false;  // Force re-capture path
  }

  // Triton graph replay conditions:
  // 1. Shape key matches (frozen shapes)
  // 2. Create op input values stable (ConstantOfShape shapes unchanged)
  // 3. Input addresses are unchanged since capture
  //  Only enter the Triton replay path for segments actually compiled by Triton.
  // Segments captured by the raw CUDA graph path (NativeDynamicShapePlan_cudagraph.cu)
  // have replayHandles but NO Triton arg tables. The Triton replay path's D2D copy +
  // arg table refresh is incompatible with raw CUDA graphs — it can corrupt cross-segment
  // data, causing downstream segments to read zeros instead of valid output → NaN.
  // compiledByBackend is set to backendName ONLY after a successful Triton execution.
  // Raw CUDA captures leave it empty → excluded from this path → fall through to
  // executeSegmentWithGraph() in cudagraph.cu which handles replay correctly.
  bool isTritonCompiled = (!seg.exec.compiledByBackend.empty() && seg.exec.compiledByBackend == backendName);

  // Invalidate stale graphs that have gap ops baked in. Gap ops must NOT be
  // captured into CUDA graphs — their baked addresses go stale on replay.
  // New captures exclude gap ops; this catches legacy pre-fix graphs.
  if (allowTritonCudaGraphReplay &&
      seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() &&
      isTritonCompiled &&
      seg.exec.gapOpsCapturedInGraph) {
    DSP_DIAG(EXECUTE,
             "STALE_GAP_GRAPH_INVALIDATE: invalidating seg[%d-%d] replay handle "
             "because gap ops were baked into the graph (stale addresses on replay).",
             seg.def.startSlot, seg.def.endSlot);
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.gapOpsCapturedInGraph = false;
    seg.exec.executionCount = captureMinExec;
    hasReplayHandle = false;
    replayHandleNull = true;
    isTritonCompiled = false;
    extAddrsStable = false;
  }

  if (allowTritonCudaGraphReplay && seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() && !isTritonCompiled) {
    DSP_DIAG(EXECUTE, "TRITON_REPLAY_SKIP: seg[%d-%d] has replayHandle but compiledBy='%s' (not %s) "
                      "→ falling through to raw CUDA graph replay path",
             seg.def.startSlot, seg.def.endSlot,
             seg.exec.compiledByBackend.empty() ? "(empty)" : seg.exec.compiledByBackend.c_str(),
             backendName);
  }

  // ── CLEAN REPLAY PATH ──────────────────────────────────────────────────────
  // If composite handles are captured and conditions are met, use compositeReplay()
  // which handles everything: ext sync, arg table refresh, gap execution, graph launch.
  // hasCompositeHandles() already proves Triton compilation — only Triton creates
  // composite handles. Raw CUDA graph segments use replayHandle, never compositeReplayHandles.
  // Previously gated on isTritonCompiled too, but compiledByBackend could be unset
  // when the segment was captured in the same call (chicken-and-egg: flag set at function
  // bottom after capture, but replay check is at the top of the next call).
  bool hasComposite = hasCompositeHandles(seg);
  bool compositeReplayReady = hasComposite;
  DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY_READY_CHECK: seg[%d-%d] compositeReplayReady=%d "
                    "isTritonCompiled=%d hasCompositeHandles=%d compiledBy='%s' execCount=%d",
           seg.def.startSlot, seg.def.endSlot, compositeReplayReady ? 1 : 0,
           isTritonCompiled ? 1 : 0, hasComposite ? 1 : 0,
           seg.exec.compiledByBackend.empty() ? "(empty)" : seg.exec.compiledByBackend.c_str(),
           seg.exec.executionCount);

  if (allowTritonCudaGraphReplay &&
      compositeReplayReady &&
      seg.exec.cachedShapeKey == segShapeKey &&
      createValuesStable &&
      extAddrsStable) {

    DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY_ENTER: seg[%d-%d] → compositeReplay()",
             seg.def.startSlot, seg.def.endSlot);

    auto replayStatus = compositeReplay(seg, seg.exec.compositeReplaySchedule,
                                        externalArrays, numExt, stream);
    if (replayStatus == Status::OK) {
      seg.exec.executionCount++;
      totalGraphReplays_++;
      if (seg.exec.compiledByBackend.empty()) seg.exec.compiledByBackend = backendName;
    }
    return replayStatus;
  }
  // Fall through to capture or slot-by-slot if replay conditions not met

#endif

#if HAVE_TRITON && defined(SD_CUDA)
  struct TritonOrderedRangeGuard {
   bool active = false;
   ~TritonOrderedRangeGuard() {
     if (active) TritonGraphBackend::clearOrderedRangeExecutor();
   }
 } tritonOrderedRangeGuard;

 if (tritonBackend != nullptr) {
   TritonGraphBackend::setOrderedRangeExecutor(
       [this, &seg, externalArrays, numExt, stream](int startSlot, int endSlot) -> Status {
         if (startSlot > endSlot) return Status::OK;

         GraphSegment gapSeg;
         gapSeg.def.startSlot = startSlot;
         gapSeg.def.endSlot = endSlot;
         gapSeg.exec.executionCount = seg.exec.executionCount;
         gapSeg.exec.compilationFailed = seg.exec.compilationFailed;

         // Check if the Triton stream is currently being captured (CUDA graph recording).
         bool streamIsCapturing = false;
#ifdef SD_CUDA
         if (stream != nullptr) {
           cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
           cudaStreamIsCapturing(*static_cast<cudaStream_t*>(stream), &capStat);
           streamIsCapturing = (capStat != cudaStreamCaptureStatusNone);
         }

         cudaStream_t tritonStr = *static_cast<cudaStream_t*>(stream);
         auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
         cudaStream_t gapStr = lcStream ? *lcStream : nullptr;
         bool streamsMatch = (tritonStr == gapStr);

         // One-time diagnostic: log whether streams match
         static bool streamDiagDone = false;
         if (!streamDiagDone) {
           DSP_DIAG(BACKEND, "stream diag: tritonStr=%p gapStr=%p match=%d capturing=%d",
                    (void*)tritonStr, (void*)gapStr, streamsMatch ? 1 : 0,
                    streamIsCapturing ? 1 : 0);
           streamDiagDone = true;
         }

         if (streamIsCapturing) {
           // ── CAPTURE PATH: SKIP gap ops entirely ──
           //
           // During CUDA graph capture, Triton kernels are recorded into the graph.
           // Gap ops (matmul, attention, etc.) must NOT execute because:
           //
           //  1. Executing on the capturing stream bakes stale addresses into the
           //     graph — on replay, gap ops read/write wrong buffers, producing
           //     garbage that accumulates across 30 transformer layers.
           //
           //  2. Executing on a separate stream also fails because native ops
           //     internally use the legacy stream (stream 0) for D2H copies,
           //     allocations, and syncs — all of which are illegal during capture
           //     (error 900/224: "operation would make the legacy stream depend
           //     on a capturing blocking stream").
           //
           // Solution: SKIP gap ops during capture. Warmup already executed them
           // and populated outputSlots_ at the correct addresses. The Triton arg
           // table snapshot will reference these warmup addresses. On replay, the
           // composite replay schedule executes gaps FRESH before graph replay.
           //
           // This is correct because:
           //  - Shapes are frozen (gap output shapes don't change)
           //  - Output buffer addresses are stable (same outputSlots_ from warmup)
           //  - Triton kernels reference buffer addresses via arg tables, which
           //    are refreshed from outputSlots_ before each replay
           //  - The captured graph contains ONLY Triton kernels

           DSP_DIAG(EXECUTE, "GAP_SKIP_DURING_CAPTURE: gap[%d-%d] SKIPPED (warmup outputs "
                    "already at stable addresses) for seg[%d-%d]",
                    startSlot, endSlot, seg.def.startSlot, seg.def.endSlot);

           // gapOpsCapturedInGraph stays false — gaps are NOT in the graph
           return Status::OK;
         }

         // ── NON-CAPTURE PATH: normal gap execution with stream sync ──
         // Triton kernels and gap ops run on different streams. Synchronize
         // to ensure gap ops see completed Triton outputs and vice versa.
         if (!streamsMatch && stream != nullptr) {
           cudaStreamSynchronize(tritonStr);
         }
#endif
         bool savedGraphActive = tl_graphExecutionActive;
         tl_graphExecutionActive = false;
         auto gapStatus = executeSegmentSlotBySlot(gapSeg, externalArrays, numExt, stream);
#ifdef SD_CUDA
         if (!streamsMatch && gapStr != nullptr) {
           cudaStreamSynchronize(gapStr);
         }
#endif
         tl_graphExecutionActive = savedGraphActive;
         return gapStatus;
       });
   tritonOrderedRangeGuard.active = true;
 }
#endif

  Status status = Status::KERNEL_FAILURE;
  bool usedTritonGraphCapture = false;

#ifdef SD_CUDA
  // Recompute shouldCaptureTritonGraph here (same logic as CAPTURE DECISION CHECK above).
  // This is the actual capture point - the diagnostic above just logs the decision.
  bool hasReplayHandleNow = (seg.exec.replayHandle != nullptr);
  bool replayHandleNullNow = (seg.exec.replayHandle == nullptr);
  bool execCountInWindowNow = (seg.exec.executionCount >= captureMinExec) &&
                              (forceRecaptureEnabled || seg.exec.executionCount <= (captureMinExec + 2));
  bool captureWindowSatisfiedNow = execCountInWindowNow || requiresOrderedGapCapture;
  bool shouldCaptureTritonGraphNow = allowTritonCudaGraphReplay &&
                                     !hasReplayHandleNow &&
                                     replayHandleNullNow &&
                                     !seg.exec.compilationFailed &&
                                     captureWindowSatisfiedNow &&
                                     hasCudaStream;
  // OOM retry deferred check: if a previous capture attempt failed with OOM and
  // we haven't reached the retry-after execution count, skip capture for this
  // execution and keep the warmup path active.
  if (seg.exec.captureOomRetries > 0 &&
      seg.exec.executionCount < seg.exec.captureRetryAfterExec) {
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "OOM RETRY DEFERRED: seg[%d-%d] retries=%d execCount=%d retryAfter=%d — warmup path",
                 seg.def.startSlot, seg.def.endSlot, seg.exec.captureOomRetries,
                 seg.exec.executionCount, seg.exec.captureRetryAfterExec);
    shouldCaptureTritonGraphNow = false;
  }

  // Proactive memory cleanup before capture: trim pool, evict LRU graphs if needed.
  if (shouldCaptureTritonGraphNow && Environment::getInstance().dspProactiveEvictBeforeCapture()) {
    proactivePreCaptureMemoryCleanup(seg, segIdx, stream);
  }

  if (shouldCaptureTritonGraphNow) {
    DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                 "GRAPH CAPTURE BEGIN: seg[%d-%d] size=%d execCount=%d shapesFrozen=%d",
                 seg.def.startSlot, seg.def.endSlot, seg.def.endSlot - seg.def.startSlot + 1,
                 seg.exec.executionCount, shapesFrozen_ ? 1 : 0);
    seg.exec.gapOpsCapturedInGraph = false;

    // Set up capture workspace BEFORE beginCapture — cudaMalloc must be outside capture.
    // Native ordered range ops (matmul, attention, concat) need temporary buffers during execution.
    // With tl_graphExecutionActive=true, CudaMemoryPool allocates from this workspace
    // instead of calling cudaMallocAsync (which fails during capture).
    // TRITON_CAPTURE_WORKSPACE_SIZE is now at file scope (above).

    // Create the replayHandle BEFORE capture — it must exist to store workspace, host ptrs, etc.
    {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      seg.exec.replayHandle = GraphReplayFactory::create(deviceId);
    }

    if (seg.exec.replayHandle->getWorkspacePtr() == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);

      // Shared workspace: allocate once, reuse across all segments.
      // Segments execute sequentially (cudaGraphLaunch + cudaStreamSynchronize),
      // and workspace offset resets each capture, so sharing is safe.
      if (sharedCaptureWorkspace_ == nullptr) {
        // First segment — allocate the shared workspace
        cudaError_t err = cudaMalloc(&sharedCaptureWorkspace_, TRITON_CAPTURE_WORKSPACE_SIZE);
        if (err != cudaSuccess) {
          cudaGetLastError();
          sharedCaptureWorkspace_ = nullptr;
        }
        if (sharedCaptureWorkspace_ != nullptr) {
          sharedCaptureWorkspaceBytes_ = TRITON_CAPTURE_WORKSPACE_SIZE;
          sharedCaptureWorkspaceDevice_ = deviceId;
          memory::CudaMemoryPool::getInstance().registerCaptureWorkspace(
              sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                       "allocated SHARED capture workspace: %zuMB on device %d",
                       TRITON_CAPTURE_WORKSPACE_SIZE / (1024*1024), deviceId);
        } else {
          // Shared allocation failed — ABORT capture for this segment.
          platformCleanupSegmentForRebuild(seg);
          return reportOomError(seg, "shared_workspace_allocation",
                                TRITON_CAPTURE_WORKSPACE_SIZE, deviceId);
        }
      } else {
        DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                     "using shared workspace for seg[%d-%d]",
                     seg.def.startSlot, seg.def.endSlot);
      }

      // Point this segment's replay handle at the shared workspace
      seg.exec.replayHandle->useExternalWorkspace(
          sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
    }

    // Guard: if replay handle creation failed, crash immediately.
    // Silent fallthrough to slot-by-slot masks the real bug.
    if (seg.exec.replayHandle == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      char buf[256];
      snprintf(buf, sizeof(buf),
               "NativeDSP: GraphReplayFactory::create returned nullptr for seg[%d-%d] on device %d. "
               "Replay handle creation failed — fix the root cause.",
               seg.def.startSlot, seg.def.endSlot, deviceId);
      DSP_DIAG(COMPILE, "%s", buf);
      THROW_EXCEPTION(buf);
    } else {
      tl_captureWorkspace = seg.exec.replayHandle->getWorkspacePtr();
      tl_captureWorkspaceSize = seg.exec.replayHandle->getWorkspaceBytes();
      tl_captureWorkspaceOffset = 0;
      tl_capturedHostPtrs.clear();
      tl_captureReplicateCache.clear();

      // Allocate pinned host workspace for H2D source copies during capture.
      // During capture, DataBuffer::syncToSpecial and PointersManager need a persistent
      // pinned buffer as H2D memcpy source. Without this, they use _primaryBuffer directly.
      // Temporary arrays (axis/dimension params for gap ops) get freed after the op completes,
      // but the graph's H2D memcpy node bakes the source address — reading freed memory on
      // launch causes SIGSEGV. The pinned workspace persists for the graph's lifetime.
      void* captureHostWs = nullptr;
      {
        auto hostWsErr = cudaMallocHost(&captureHostWs, TRITON_CAPTURE_HOST_WORKSPACE_SIZE);
        if (hostWsErr != cudaSuccess) {
          cudaGetLastError();
          captureHostWs = nullptr;
          // Host workspace allocation failed — H2D copies during capture will use
          // non-pinned _primaryBuffer directly. When temporary arrays (axis/dimension
          // params for gap ops) are freed after the op completes, the graph's H2D
          // memcpy node still references the freed source address, causing SIGSEGV on
          // replay. This is a fatal error, not a degraded-but-correct path.
          int deviceId = 0;
          cudaGetDevice(&deviceId);
          char buf[256];
          snprintf(buf, sizeof(buf),
                   "NativeDSP: cudaMallocHost failed for capture host workspace (%zuMB) "
                   "seg[%d-%d] device %d cudaError=%d (%s). "
                   "Without pinned host workspace, graph replay will SIGSEGV on freed source addresses.",
                   TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024),
                   seg.def.startSlot, seg.def.endSlot, deviceId,
                   static_cast<int>(hostWsErr), cudaGetErrorString(hostWsErr));
          DSP_DIAG(COMPILE, "%s", buf);
          // No TLS or context cleanup needed — capture hasn't started yet
          restoreCublasWorkspaceAfterCapture(stream);
          platformCleanupSegmentForRebuild(seg);
          THROW_EXCEPTION(buf);
        } else {
          DSP_DIAG(MEMORY, "allocated %zuMB pinned host workspace for Triton capture seg[%d-%d]",
                   TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024), seg.def.startSlot, seg.def.endSlot);
        }
      }
      tl_captureHostWorkspace = captureHostWs;
      tl_captureHostWorkspaceSize = (captureHostWs != nullptr) ? TRITON_CAPTURE_HOST_WORKSPACE_SIZE : 0;
      tl_captureHostWorkspaceOffset = 0;
      // Track the host workspace as a captured host pointer for lifetime management.
      // On successful capture, this moves to the replay handle (addCapturedHostPtr).
      // On failure, it's freed immediately (line below at tl_capturedHostPtrs cleanup).
      if (captureHostWs != nullptr) {
        tl_capturedHostPtrs.push_back(captureHostWs);
      }

      // Set capture stream so captureSafeStreamOrDefault() routes ops to the correct stream
      cudaStream_t prevCaptureStream = tl_graphCaptureStream;
      tl_graphCaptureStream = cudaStr;
      auto cleanupCaptureTls = [&](bool freeCapturedHostPtrs) {
        tl_graphExecutionActive = false;
        tl_captureWorkspace = nullptr;
        tl_captureWorkspaceSize = 0;
        tl_captureWorkspaceOffset = 0;
        if (freeCapturedHostPtrs) {
          for (auto* ptr : tl_capturedHostPtrs) {
            if (ptr != nullptr) {
              cudaFreeHost(ptr);
            }
          }
        }
        tl_captureHostWorkspace = nullptr;
        tl_captureHostWorkspaceSize = 0;
        tl_captureHostWorkspaceOffset = 0;
        tl_capturedHostPtrs.clear();
        tl_captureReplicateCache.clear();
        tl_graphCaptureStream = prevCaptureStream;
      };

      // Pre-allocate cuBLAS workspace to prevent internal cudaMalloc during capture.
      // cuBLAS internally allocates workspace on stream 0 for GEMM operations. During
      // graph capture on a named stream, this cross-stream allocation breaks capture,
      // producing invalid graph nodes that SIGSEGV on cudaGraphLaunch.
      const size_t CUBLAS_WORKSPACE_SIZE = Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
      ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
      // NOTE: setCublasWorkspaceForCapture is deferred to AFTER warmup (see below).
      // Calling it here sets cublasSetStream_v2 to the capture stream, which causes
      // cuBLAS matmuls in gap ops during warmup to run on tritonStr instead of gapStr.
      // This stream mismatch creates data races: cast ops on gapStr write matmul
      // inputs, but cuBLAS on tritonStr starts before gapStr completes.

      // Reset cast cache indices (NOT full clear) before warmup.
      // Previous segments' warmup entries may still be referenced by their captured
      // graphs — clearCastCache() would delete the NDArrays, causing cudaFreeAsync
      // on the GPU buffers. Those addresses are baked into the captured graph nodes
      // (assign + cuBLAS GEMM), so freeing them causes "illegal memory access" (700)
      // on replay. resetCastCacheIndices() preserves the buffers while letting this
      // segment's warmup reuse or append entries as needed.
      //
      // Note: Shape mismatches from speculative decode (draft vs target model) are
      // handled by the mid-execution clearCastCache() call inside MmulHelper::mmul
      // (lines ~761, 1058), which safely skips frees during capture.
      MmulHelper::resetCastCacheIndices();

      // ── Batch-zero preparation (OUTSIDE capture) ─────────────────────────
      // Collects the set of output buffers that need zeroing before each replay.
      // The pre-capture loop below consumes batchZeroEntries_ via cudaMemsetAsync.
      if (Environment::getInstance().dspBatchZero()) {
        std::unordered_set<int> gapSlots;
        if (Environment::getInstance().dspBatchZeroGapOnly()) {
#if HAVE_TRITON && defined(SD_CUDA)
          auto* tritonBE = dynamic_cast<TritonGraphBackend*>(backend);
         if (tritonBE != nullptr) {
           gapSlots = tritonBE->getGapSlots(seg, slots_);
         } else
#endif
          {
            for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) gapSlots.insert(s);
          }
        } else {
          for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) gapSlots.insert(s);
        }
        collectBatchZeroTargets(gapSlots);
      }

      // Sync external inputs to device before capture — same rationale as non-capture path.
      // Java may have modified host buffers (putScalar + tagLocation(HOST)) between steps.
      // specialBuffer() in arg table population doesn't check for stale device data.
      //
      // directReference optimization: skip syncing non-variable (weight) external inputs.
      // Weight buffers are device-authoritative (isSpecialActual() == true); their syncToDevice()
      // would be a no-op anyway. Skipping the iteration saves CPU time on large external input sets.
      // Variable inputs (input_ids, position_ids, attention_mask) must still be synced.
      {
        int syncedCapture = 0, skippedCapture = 0;
        bool useVariableFilter = shapesFrozen_ && !externalInputIsVariable_.empty();
        for (int ei = 0; ei < numExt; ei++) {
          if (externalArrays[ei] == nullptr) continue;
          if (useVariableFilter &&
              ei < static_cast<int>(externalInputIsVariable_.size()) &&
              !externalInputIsVariable_[ei]) {
            // Weight directReference: skip sync — weight is already device-authoritative
            skippedCapture++;
            continue;
          }
          if (Environment::getInstance().tritonVerifyKernels()) {
            auto* db = externalArrays[ei]->dataBuffer();
            DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(capture) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p",
                     -(ei + 1), ei,
                     db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                     db ? (db->isSpecialActual() ? 1 : 0) : -1,
                     (long long)externalArrays[ei]->lengthOf(),
                     DSP_BUF(externalArrays[ei]));
          }
          // forceSync=true: markOrderedRangeDeviceCurrent() calls readSpecial() on
          // inputs after each step, poisoning actuality flags. Without forceSync,
          // syncToSpecial() may skip the H2D copy for variable inputs.
          externalArrays[ei]->dataBuffer()->syncToSpecial(true);
          syncedCapture++;
        }
        DSP_DIAG(MEMORY, "pre-capture EXT_SYNC directReference: %d synced, %d weights skipped (frozen=%d, varFilter=%d)",
                 syncedCapture, skippedCapture, (int)shapesFrozen_, (int)useVariableFilter);
      }

      // Cross-stream ordering: Java-side assign() runs on the default stream or
      // a LaunchContext stream BEFORE DSP execution starts. syncToDevice() above
      // is a no-op when isSpecialActual()=true (set by tickDeviceWrite after
      // assign), so it doesn't establish ordering between the assign stream and
      // cudaStr. Without this, capture can bake in stale device data from the
      // previous step (assign kernel hasn't completed on its stream yet).
      // Same pattern as compositeReplay cross-stream sync (line ~896).
      {
        cudaStream_t defaultStream = nullptr;
        auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
        if (defaultStreamPtr != nullptr) defaultStream = *defaultStreamPtr;
        if (defaultStream != nullptr && defaultStream != cudaStr) {
          cudaEvent_t evt = getCrossStreamEvent();
          cudaEventRecord(evt, defaultStream);
          cudaStreamWaitEvent(cudaStr, evt, 0);
        }
      }
      // Clear any sticky CUDA error before capture — stale errors from prior operations
      // (e.g., cudaFuncGetName on driver-API functions) contaminate capture and launch.
      cudaGetLastError();

      // Diagnostic: fingerprint every variable external input (placeholders) AFTER
      // sync, so step-to-step content drift is visible. A placeholder stuck on the
      // previous step's data will repeat its fingerprint. Runs outside CUDA graph
      // capture (we're pre-cudaStreamBeginCapture here).
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        if (ei >= static_cast<int>(externalInputIsVariable_.size()) || !externalInputIsVariable_[ei]) continue;
        const char* name = (ei < static_cast<int>(externalInputNames_.size()))
                           ? externalInputNames_[ei].c_str() : "?";
        DSP_DIAG_FINGERPRINT("capture-ext-sync", ei, name, externalArrays[ei],
                             seg.exec.executionCount);
      }

      // Configurable: push primary CUDA context during capture.
      // Default OFF — the non-Triton path works without it. Pushing and then popping
      // after capture may cause SIGSEGV on replay (null pointer inside libcuda.so).
      // Enable via ND4J_TRITON_GRAPH_CTX_PUSH=1 for debugging.
      int tritonCaptureDevice = 0;
      cudaGetDevice(&tritonCaptureDevice);
      CUcontext primaryCtx = nullptr;
      CUcontext prevCtx = nullptr;
      bool didPushCtx = false;
      if (Environment::getInstance().tritonGraphCtxPush()) {
        CUdevice cuDev;
        cuDeviceGet(&cuDev, tritonCaptureDevice);
        cuDevicePrimaryCtxRetain(&primaryCtx, cuDev);
        cuCtxGetCurrent(&prevCtx);
        if (prevCtx != primaryCtx) {
          cuCtxPushCurrent(primaryCtx);
          didPushCtx = true;
          DSP_DIAG(EXECUTE, "Triton capture pushed primary ctx %p (was %p) for device %d",
                   (void*)primaryCtx, (void*)prevCtx, tritonCaptureDevice);
        }
      }

      // ── PRE-CAPTURE WARMUP EXECUTION ────────────────────────────────────────
      // During CUDA graph capture, GPU operations are NOT executed — they are only
      // recorded into the graph.  The capture step's output buffers retain whatever
      // values they had BEFORE capture started.  Without a warmup, those values are
      // from the PREVIOUS step's execution, producing a stale/wrong token that
      // corrupts the entire decode sequence.
      //
      // Fix: run a non-capture execution BEFORE capture to produce correct output
      // for this step.  The capture then records the same operations (for replay),
      // but the output buffers already have the correct values from the warmup.
      // This matches the non-Triton CUDA graph path (NativeDynamicShapePlan_cudagraph.cu
      // line 488-490) which runs executeSegmentSlotBySlot() before capture.
      {
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton pre-capture warmup for seg[%d-%d] execCount=%d",
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);

        // Set cuBLAS workspace during warmup too, so cuBLAS selects the same GEMM
        // algorithms as during capture. Without this, warmup may use different
        // algorithms than capture, causing shape/result divergence.
        setCublasWorkspaceForWarmup();

        // Disable frozen fast path for warmup — same rationale as capture below.
        std::vector<NativeSlot::SlotState> savedSlotStateWarmup(seg.def.endSlot - seg.def.startSlot + 1);
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          savedSlotStateWarmup[s - seg.def.startSlot] = slots_[s].state_;
          // Demote FROZEN→SHAPE_CACHED but preserve FROZEN_CONSTANT (see non-capture fix)
          if (slots_[s].state_ == NativeSlot::SlotState::FROZEN)
            slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
        }

        // DIAGNOSTIC: use slot-by-slot execution for warmup instead of Triton backend
        // to isolate whether the issue is in the Triton execution path or in the warmup environment
        Status warmupStatus;
        {
          // Use slot-by-slot for warmup — this matches the REF path exactly
          GraphSegment warmupSeg;
          warmupSeg.def.startSlot = seg.def.startSlot;
          warmupSeg.def.endSlot = seg.def.endSlot;
          warmupSeg.exec.executionCount = seg.exec.executionCount;
          warmupSeg.exec.compilationFailed = seg.exec.compilationFailed;
          warmupStatus = executeSegmentSlotBySlot(warmupSeg, externalArrays, numExt, stream);
        }
        // Restore frozen state after warmup
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          slots_[s].state_ = savedSlotStateWarmup[s - seg.def.startSlot];
        }

        if (warmupStatus != Status::OK) {
          DSP_DIAG(EXECUTE, "FATAL: Triton pre-capture warmup FAILED for seg[%d-%d] status=%d. "
                            "BLOCKING EXECUTION.",
                   seg.def.startSlot, seg.def.endSlot, static_cast<int>(warmupStatus));
          seg.exec.compilationFailed = true;
          cleanupCaptureTls(true);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          // Destroy the replay handle created before warmup — it holds workspace
          // memory even though capture never started.
          platformCleanupSegmentForRebuild(seg);
          return warmupStatus;
        }
        // Decrement executionCount — the warmup was an extra execution that should
        // not count toward the capture threshold.
        if (seg.exec.executionCount > 0) seg.exec.executionCount--;

        // Synchronize before capture to ensure warmup results are visible
        cudaStreamSynchronize(cudaStr);
        cudaGetLastError();

        // Reset cast cache indices after warmup so capture starts from index 0.
        // We intentionally preserve the warmup's cast cache entries (NOT full clear).
        // Capture reuses them via assign() — the graph records a cast kernel from the
        // real input to the cached buffer, then cuBLAS reads the cached buffer.
        // clearCastCache() would delete these entries, forcing capture to allocate
        // new ones from the capture workspace. Those workspace sub-allocations
        // cannot be individually freed by cudaFreeAsync (they're interior pointers
        // of the 32MB workspace block), so subsequent clearCastCache() calls
        // corrupt the CUDA memory pool → "illegal memory access" on replay.
        MmulHelper::resetCastCacheIndices();

        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton pre-capture warmup DONE for seg[%d-%d]",
                     seg.def.startSlot, seg.def.endSlot);

        // DIAGNOSTIC: dump warmup's final output argmax for comparison with replay
        {
          int finalOutputSlot = -1;
          if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
            finalOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
          }
          if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_)
            finalOutputSlot = seg.def.endSlot;
          if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
              outputSlots_[finalOutputSlot] != nullptr) {
            auto* warmupOut = outputSlots_[finalOutputSlot];
            if (warmupOut->dataType() == FLOAT32 && warmupOut->lengthOf() > 0) {
              int warmupArgmax = dspArgmax(DSP_BUF(warmupOut), warmupOut->dataType(),
                                           warmupOut->lengthOf());
              std::string warmupVals = dspDumpSlotValues(DSP_BUF(warmupOut), warmupOut->dataType(),
                                                         warmupOut->lengthOf(), 4);
              DSP_DIAG(EXECUTE, "WARMUP ARGMAX: slot=%d argmax=%d len=%lld vals=%s execCount=%d",
                       finalOutputSlot, warmupArgmax, (long long)warmupOut->lengthOf(),
                       warmupVals.c_str(), seg.exec.executionCount);
            }
          }
        }

        // ── RESTORE NULL OUTPUT SLOTS FROM CACHE ─────────────────────────────
        // The warmup execution may clear some outputSlots_ entries (e.g. control
        // flow CF_SWITCH dead outputs, or segment cleanup paths).  The values
        // were captured into outputSlots_ during execution, so restore any
        // Phase 2: outputSlots_ == outputSlots_ (unified).
        // Post-warmup restoration is a no-op — arrays produced during warmup
        // are already in outputSlots_ (which IS outputSlots_).
      }

      // DIAGNOSTIC: warmup-only mode — skip capture, use warmup result directly.
      // Enables bisection: if warmup-only produces correct output but capture+replay
      // does not, the bug is in capture/replay.
      {
        static bool warmupOnly = Environment::getInstance().triton().warmupOnly();
        if (warmupOnly) {
          DSP_DIAG(EXECUTE, "WARMUP_ONLY: skipping capture for seg[%d-%d], using warmup result",
                   seg.def.startSlot, seg.def.endSlot);
          cleanupCaptureTls(true);
          // Don't need the replay handle — fall through to non-capture path next time.
          // Destroy it to free the workspace memory allocated at line 1756.
          seg.exec.compilationFailed = true;
          platformCleanupSegmentForRebuild(seg);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          return Status::OK;
        }
      }

      // NOW set cuBLAS handle to capture stream — AFTER warmup completed.
      // During warmup, gap ops must use their default stream (gapStr) for cuBLAS.
      // Only during actual capture do we switch cuBLAS to tritonStr so GEMM nodes
      // are recorded into the CUDA graph on the correct stream.
      setCublasWorkspaceForCapture(stream);

      // cuBLAS workspace preservation during capture.
      //
      //  Once shapes are frozen (shapesFrozen_ == true), NEVER zero the cuBLAS workspace.
      // During capture, cuBLAS stores plan/descriptor data in the workspace. Captured CUDA graphs
      // inherit these cached plans and omit H2D re-upload nodes. Zeroing the workspace destroys
      // cached plans, causing GEMM kernels to read zeros and hang on replay.
      //
      // The workspace content must be preserved across ALL captures and replays once frozen.
      // cuBLAS plans are stable for fixed shapes, so preservation is safe.
      //
      // Pre-frozen (shapes not yet frozen): zeroing is acceptable as no graphs are captured yet.
      if (shapesFrozen_ && cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
        DSP_DIAG(MEMORY, "pre-capture: cuBLAS workspace PRESERVED (%zuMB) — shapes frozen, plans stable",
                 cublasWorkspaceSize_ / (1024*1024));
        // Do NOT zero — preserve cuBLAS plan data for captured graph replay
      }
      // Note: Pre-frozen zeroing removed entirely — not needed for correctness and
      // adds unnecessary overhead. cuBLAS handles uninitialized workspace correctly.

      // Disable frozen fast path during capture. Same rationale as non-Triton path:
      // capture may re-create views, and the frozen context has stale input/output pointers
      // from the prior non-capture execution. Using the full (non-frozen) path during capture
      // is a one-time cost — all context pointers are properly reconfigured with capture-time
      // arrays, including correct nullify() calls to zero output buffers.
      // Save and restore frozenContextReady after capture so replay uses frozen fast path.
      std::vector<NativeSlot::SlotState> savedSlotStateTriton(seg.def.endSlot - seg.def.startSlot + 1);
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        savedSlotStateTriton[s - seg.def.startSlot] = slots_[s].state_;
        // Demote FROZEN→SHAPE_CACHED but preserve FROZEN_CONSTANT (see non-capture fix)
        if (slots_[s].state_ == NativeSlot::SlotState::FROZEN)
          slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
      }
      auto restoreCaptureSlotState = [&]() {
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          slots_[s].state_ = savedSlotStateTriton[s - seg.def.startSlot];
        }
      };

      std::vector<std::pair<int, NDArray*>> savedExtForCapture;
      std::vector<std::pair<int, NDArray*>> savedSlotsForCapture;

      // Pre-capture promotion: view-capable slots should enter FROZEN state BEFORE
      // capture begins. The view installation path reuses the input's DataBuffer
      // directly, which is correct for BOTH constant and variable inputs:
      // - For constant inputs: the view sees the stable warmup value
      // - For variable inputs: the view shares the input's DataBuffer, so replay
      //   automatically sees updated values when the input is refreshed
      //
      // Without this promotion, view-capable slots go through normal execution
      // during capture: they allocate new output buffers and create H2D capture
      // nodes. Later, those H2D nodes can copy stale host data over the output,
      // corrupting the values downstream consumers depend on.
      //
      // isDataDependent is NOT a disqualifier: a reshape's output shape comes from
      // input values, but the view still shares the input's DataBuffer correctly.
      if (shapesFrozen_) {
        int promoted = 0;
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          auto& sl = slots_[s];
          if (!sl.flags.isViewCapableOp || sl.state_ >= NativeSlot::SlotState::FROZEN) continue;
          sl.state_ = NativeSlot::SlotState::FROZEN;
          promoted++;
        }
        if (promoted > 0) {
          DSP_DIAG(EXECUTE, "pre-capture view promotion: %d view-capable slots promoted to FROZEN for seg[%d-%d]",
                   promoted, seg.def.startSlot, seg.def.endSlot);
        }
      }

      // Pre-capture batch-zero: zero all registered buffers BEFORE beginCapture.
      // These cudaMemsetAsync calls execute normally on the stream (not captured).
      //
      // IMPORTANT: Only for MONOLITHIC capture. For COMPOSITE capture, skip batch-zero
      // here because composite capture re-executes gap ops between islands, and those
      // gap ops need valid intermediate results from the warmup as inputs. Batch-zero
      // would destroy those intermediate values (zeroing gap op input buffers), causing
      // gap ops to read zeros and produce wrong results that propagate through the
      // entire model. Composite replay handles zeroing correctly: pre-replay batch-zero
      // zeros outputs before each replay, and gap ops call nullify() on their own outputs.
      bool willUseCompositeCapture = false;
#if HAVE_TRITON && defined(SD_CUDA)
      {
       auto& schedCheck = seg.exec.compositeReplaySchedule;
       for (auto& u : schedCheck.units) {
         if (u.kind == REPLAY_UNIT_TRITON_ISLAND) { willUseCompositeCapture = true; break; }
       }
     }
#endif
      if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty() &&
          !willUseCompositeCapture) {
        for (auto& entry : batchZeroEntries_) {
          if (entry.ptr != nullptr && entry.bytes > 0) {
            cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr);
          }
        }
        DSP_DIAG(MEMORY, "pre-capture batch-zero: %d buffers zeroed via cudaMemsetAsync (fill engines, before beginCapture)",
                 static_cast<int>(batchZeroEntries_.size()));
      } else if (willUseCompositeCapture) {
        DSP_DIAG(MEMORY, "pre-capture batch-zero SKIPPED for composite capture — gap ops need valid warmup data as inputs");
      }

      // ── Save warmup output slot pointers BEFORE capture ─────────────────
      // Gap ops are skipped during capture (they return OK without executing).
      // outputSlots_[] retains warmup values throughout capture — no save/restore needed.
      // Downstream segments see valid warmup data. Triton sub-kernel arg tables
      // reference warmup addresses, which are stable.

      // POST-ALLOCATION MEMORY GATE: workspace + cuBLAS are allocated. Check that
      // enough headroom remains for CUDA driver graph metadata before starting capture.
      // This is tight and accurate — only graph metadata overhead remains.
      {
        size_t gpuFree = 0, gpuTotal = 0;
        cudaMemGetInfo(&gpuFree, &gpuTotal);
        size_t safetyBytes = Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;
        if (gpuFree < safetyBytes) {
          int deviceId = 0;
          cudaGetDevice(&deviceId);
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                       "POST-ALLOC GATE FAILED: free=%zuMB < safety=%zuMB for seg[%d-%d]",
                       gpuFree / (1024*1024), safetyBytes / (1024*1024),
                       seg.def.startSlot, seg.def.endSlot);
          platformCleanupSegmentForRebuild(seg);
          return reportOomError(seg, "post_alloc_gate", safetyBytes, deviceId);
        }
      }

      // ── COMPOSITE CAPTURE: per-island CUDA graphs for mixed Triton/gap segments ──
      // When a segment has interleaved gap ops (matmul, attention) between Triton
      // islands, we must capture each island separately. A monolithic CUDA graph
      // cannot interleave with gap ops — graph replay executes all captured ops
      // atomically. With per-island capture:
      //   1. Island A is captured → CudaGraphReplayHandle stored in compositeReplayHandles[0]
      //   2. Gap ops between A and B execute natively (fresh each replay)
      //   3. Island B is captured → compositeReplayHandles[1]
      //   ...repeat...
      // Replay then follows the schedule in program order:
      //   replay(island_A) → executeSlots(gap_B) → replay(island_C) → ...
      // This preserves data dependencies: gap_B reads fresh island_A output.
      // Declared before the #if block so it's visible to the monolithic path's guard.
      bool didCompositeCapture = false;
#if HAVE_TRITON && defined(SD_CUDA)
      {
       auto& sched = seg.exec.compositeReplaySchedule;
       bool hasIslandUnits = false;
       for (auto& u : sched.units) {
         if (u.kind == REPLAY_UNIT_TRITON_ISLAND) { hasIslandUnits = true; break; }
       }
       if (hasIslandUnits && !sched.units.empty()) {
         DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE_BEGIN: seg[%d-%d] — per-island capture for %d units",
                  seg.def.startSlot, seg.def.endSlot, static_cast<int>(sched.units.size()));
         // Resize compositeReplayHandles to accommodate all TRITON_ISLAND units.
         // islandIndex in TRITON_ISLAND units is the index into this vector.
         int maxIslandIdx = -1;
         for (auto& u : sched.units) {
           if (u.kind == REPLAY_UNIT_TRITON_ISLAND && u.islandIndex > maxIslandIdx) {
             maxIslandIdx = u.islandIndex;
           }
         }
         if (maxIslandIdx >= 0) {
           sched.compositeReplayHandles.resize(maxIslandIdx + 1);
         }

         bool allIslandsOk = true;
         int deviceId = 0;
         cudaGetDevice(&deviceId);

         for (size_t unitIdx = 0; unitIdx < sched.units.size() && allIslandsOk; unitIdx++) {
           auto& unit = sched.units[unitIdx];

           if (unit.kind == REPLAY_UNIT_GAP) {
             // Execute gap slots normally BEFORE the next island — these run natively,
             // producing fresh outputs at stable addresses that the next island reads.
             DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: gap unit [%d-%d] — executing slots natively",
                      unit.startSlot, unit.endSlot);
             for (int s = unit.startSlot; s <= unit.endSlot; s++) {
               auto gapStatus = executeSlot(s, externalArrays, numExt, stream);
               if (gapStatus != Status::OK) {
                 DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: gap slot %d FAILED status=%d",
                          s, static_cast<int>(gapStatus));
                 allIslandsOk = false;
                 break;
               }
             }
           } else {  // REPLAY_UNIT_TRITON_ISLAND
             int islandIdx = unit.islandIndex;
             DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: island unit [%d-%d] islandIdx=%d — begin capture",
                      unit.startSlot, unit.endSlot, islandIdx);

             // Create a new handle for this island
             auto islandHandle = GraphReplayFactory::create(deviceId);
             if (!islandHandle) {
               DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: GraphReplayFactory::create failed for island %d",
                        islandIdx);
               allIslandsOk = false;
               break;
             }
             // Share the capture workspace (same as monolithic path)
             islandHandle->useExternalWorkspace(sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
             tl_captureWorkspace = islandHandle->getWorkspacePtr();
             tl_captureWorkspaceSize = islandHandle->getWorkspaceBytes();
             tl_captureWorkspaceOffset = 0;

             auto* cudaIslandReplay = static_cast<CudaGraphReplayHandle*>(islandHandle.get());
             auto islandNativeHandle = cudaIslandReplay->getNativeHandle();

             // Set island slot range filter — executeSegment will only capture
             // sub-kernels within [unit.startSlot, unit.endSlot] for this island
             tl_islandSlotMin = unit.startSlot;
             tl_islandSlotMax = unit.endSlot;

             // External inputs are already synced to device (syncToDevice at line 4321).
             // After syncToSpecial(), readSpecial() is called, making isSpecialActual()=true
             // via _readSpecial > _writePrimary. This means the capture-mode guard in
             // DataBuffer::syncToSpecial (line 839: if(isSpecialActual()) return;) already
             // prevents redundant H2D memcpy nodes from being recorded during capture.
             //
             // DO NOT call writeSpecial() here. It poisons isPrimaryActual()=false by
             // bumping _writeSpecial > _writePrimary, which makes subsequent Java getFloat()
             // calls think host data is stale and copy zeros from device over valid host data.
             // This was the root cause of the 20% accuracy bug in cross-plan scenarios.
             DSP_DIAG(MEMORY, "COMPOSITE_CAPTURE: island %d — external inputs already device-actual "
                      "via syncToDevice+readSpecial (NO writeSpecial poisoning)", islandIdx);
             bool islandBeginOk = islandNativeHandle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed);
             if (islandBeginOk) {
               tl_graphExecutionActive = true;

               // NOTE: No in-graph output zeroing here. Internal outputs are NOT marked
               // sAct=1, so ops' nullify() calls naturally record memset nodes during
               // capture. This ensures only outputs that actually need zeroing get zeroed.

               // executeSegment will: skip gaps (streamIsCapturing=true), capture
               // only sub-kernels within [tl_islandSlotMin, tl_islandSlotMax]
               auto captureStatus = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                            outputSlots_, totalOutputSlots_, stream);
               tl_graphExecutionActive = false;
               tl_islandSlotMin = INT_MIN;
               tl_islandSlotMax = INT_MAX;

               if (captureStatus == Status::OK) {
                 bool endOk = islandNativeHandle->endCapture(cudaStr);
                 size_t nodeCount = endOk ? islandNativeHandle->getNumNodes() : 0;
                 DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: island %d endCapture=%d nodes=%zu",
                          islandIdx, endOk ? 1 : 0, nodeCount);

                 if (endOk && nodeCount > 0) {
                   bool instOk = islandNativeHandle->instantiate();
                   if (instOk) {
                     // Validation launch for this island
                     bool launchOk = islandHandle->replay(stream);
                     if (launchOk) {
                       cudaError_t syncErr = cudaStreamSynchronize(cudaStr);
                       if (syncErr == cudaSuccess) {
                         DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: island %d [%d-%d] captured+validated OK",
                                  islandIdx, unit.startSlot, unit.endSlot);
                         sched.compositeReplayHandles[islandIdx] = std::move(islandHandle);
                       } else {
                         DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: island %d validation sync FAILED err=%d",
                                  islandIdx, static_cast<int>(syncErr));
                         cudaGetLastError();
                         allIslandsOk = false;
                       }
                     } else {
                       DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: island %d validation replay FAILED", islandIdx);
                       allIslandsOk = false;
                     }
                   } else {
                     DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: island %d instantiate FAILED", islandIdx);
                     allIslandsOk = false;
                   }
                 } else {
                   // Empty graph or endCapture failed — mark non-capturable
                   if (endOk && nodeCount == 0) {
                     DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: island %d has 0 nodes — segment non-capturable",
                              islandIdx);
                   }
                   allIslandsOk = false;
                 }
               } else {
                 // executeSegment failed during capture — abort
                 tl_islandSlotMin = INT_MIN;
                 tl_islandSlotMax = INT_MAX;
                 if (islandNativeHandle->isCapturing()) {
                   islandNativeHandle->endCapture(cudaStr);
                 }
                 allIslandsOk = false;
               }
             } else {
               // beginCapture failed
               tl_islandSlotMin = INT_MIN;
               tl_islandSlotMax = INT_MAX;
               DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: island %d beginCapture FAILED", islandIdx);
               allIslandsOk = false;
             }
           }
         }  // end for each unit

         if (allIslandsOk) {
           // All islands captured successfully.
           // seg.exec.replayHandle is already created above (the sentinel).
           // Composite replay will use compositeReplayHandles instead.
           status = Status::OK;
           usedTritonGraphCapture = true;
           didCompositeCapture = true;
           // Mark as Triton-compiled immediately so compositeReplay is recognized
           // on the next call. Previously only set at function bottom (line ~4033),
           // but state resets (output slot re-alloc, eviction) between steps could
           // clear compiledByBackend before the next call reached that line.
           if (seg.exec.compiledByBackend.empty()) {
             seg.exec.compiledByBackend = backendName;
           }
           seg.exec.cachedShapeKey = segShapeKey;
           seg.exec.capturedInputAddrKey = segInputAddrKey;
           seg.exec.capturedCreateValueKey = createValueKey;
           seg.exec.capturedSlotAddrHash = computeSlotAddrHash(
               outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
           snapshotExternalAddrs(seg, externalArrays, numExt);
           seg.exec.gapOpsCapturedInGraph = false;
           seg.exec.replayUnitCount = static_cast<int>(sched.units.size());
           DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE_COMPLETE: seg[%d-%d] %d islands captured OK replayUnitCount=%d",
                    seg.def.startSlot, seg.def.endSlot, maxIslandIdx + 1, seg.exec.replayUnitCount);

           // Diagnostic: check final output after composite capture (only when DSP diagnostics enabled)
           if (DSP_DIAG_ENABLED(EXECUTE)) {
             int finalOutputSlot = -1;
             if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
               finalOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
             }
             if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_)
               finalOutputSlot = seg.def.endSlot;
             if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
                 outputSlots_[finalOutputSlot] != nullptr) {
               auto* postCapOut = outputSlots_[finalOutputSlot];
               if (postCapOut->dataType() == FLOAT32 && postCapOut->lengthOf() > 0) {
                 cudaStreamSynchronize(cudaStr);
                 int postCapArgmax = dspArgmax(DSP_BUF(postCapOut), postCapOut->dataType(),
                                               postCapOut->lengthOf());
                 DSP_DIAG(EXECUTE, "POST_COMPOSITE_CAPTURE_ARGMAX seg[%d-%d] slot=%d argmax=%d len=%lld execCount=%d",
                          seg.def.startSlot, seg.def.endSlot, finalOutputSlot, postCapArgmax,
                          (long long)postCapOut->lengthOf(), seg.exec.executionCount);
               }
             }
           }

           // No actuality reset needed — writeSpecial() is no longer called during
           // capture, so external input actuality flags remain in their natural bi-actual
           // state (isPrimaryActual()=true AND isSpecialActual()=true via readSpecial).
           // Java getFloat() will correctly see isPrimaryActual()=true and skip D2H sync.
           DSP_DIAG(MEMORY, "COMPOSITE_CAPTURE: no actuality reset needed — writeSpecial not called");

           // ── Cleanup after successful composite capture ─────────────────
           // Composite capture shares TLS state (tl_captureWorkspace, cuBLAS stream,
           // slot state) with monolithic capture. Must restore even on success.
           // cleanupCaptureTls resets tl_graphCaptureStream to prevCaptureStream.
           //
           // Captured host ptrs: island graphs may contain H2D memcpy nodes whose
           // source addresses point into the pinned host workspace. This happens when
           // native ops within an island use PointersManager or ConstantHelper during
           // capture. Move ownership to the first island handle so the pinned memory
           // persists for the graph's lifetime — same as monolithic path (line 5262).
           if (!tl_capturedHostPtrs.empty() && maxIslandIdx >= 0 &&
               sched.compositeReplayHandles.size() > 0 &&
               sched.compositeReplayHandles[0] != nullptr) {
             for (auto* ptr : tl_capturedHostPtrs) {
               sched.compositeReplayHandles[0]->addCapturedHostPtr(ptr);
             }
             DSP_DIAG(MEMORY, "COMPOSITE_CAPTURE: preserved %zu pinned host ptrs on island[0]",
                      sched.compositeReplayHandles[0]->getCapturedHostPtrs().size());
           }
           cleanupCaptureTls(false);  // false = do NOT free host ptrs, ownership moved to island handle
           if (didPushCtx) {
             CUcontext dummy;
             cuCtxPopCurrent(&dummy);
             CUdevice cuDev;
             cuDeviceGet(&cuDev, tritonCaptureDevice);
             cuDevicePrimaryCtxRelease(cuDev);
           }
           restoreCublasWorkspaceAfterCapture(stream);
           restoreCaptureSlotState();

           // FORCE_RECAPTURE: invalidate graph immediately after composite capture+launch
           // so the NEXT step also re-captures instead of replaying the just-captured graph.
           // Without this, composite captures persist and the next step enters compositeReplay()
           // instead of re-capturing — defeating the purpose of FORCE_RECAPTURE.
           if (Environment::getInstance().tritonForceRecapture()) {
             platformCleanupSegmentForRebuild(seg);
             seg.exec.argTableStable = false;
             batchD2DCount_ = 0;
             seg.exec.capturedInputAddrKey = 0;
             DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after COMPOSITE capture+launch execCount=%d",
                      seg.exec.executionCount);
           }
         } else {
           // Partial failure — free any successfully captured island handles
           for (auto& h : sched.compositeReplayHandles) {
             h.reset();
           }
           // Mark segment as non-capturable to avoid repeated failed attempts
           seg.exec.compilationFailed = true;
           cleanupCaptureTls(false);
           restoreCublasWorkspaceAfterCapture(stream);
           restoreCaptureSlotState();
           platformCleanupSegmentForRebuild(seg);
           DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE_FAILED: seg[%d-%d] — marking non-capturable",
                    seg.def.startSlot, seg.def.endSlot);
           // Fall through to slot-by-slot execution for this step
           status = Status::KERNEL_FAILURE;
         }
       }
     }
     if (!didCompositeCapture) {
       // ── MONOLITHIC CAPTURE (non-composite segments only) ──
#endif  // HAVE_TRITON && defined(SD_CUDA)

      auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
      auto handle = cudaReplay->getNativeHandle();
      bool captureOk = handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed);
      if (captureOk) {
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton graph capture started for seg[%d-%d] execCount=%d",
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
        tl_graphExecutionActive = true;

        // External inputs are already synced to device (syncToDevice before capture).
        // After syncToSpecial(), readSpecial() makes isSpecialActual()=true via
        // _readSpecial > _writePrimary. The capture-mode guard in DataBuffer::syncToSpecial
        // (if(isSpecialActual()) return;) prevents redundant H2D memcpy nodes.
        //
        // DO NOT call writeSpecial() here. It poisons isPrimaryActual()=false, causing
        // Java getFloat() to copy stale device zeros over valid host data across plans.
        // This was the root cause of the 20% VLM accuracy bug.
        //
        // Internal outputs keep their natural actuality state so nullify() records
        // memset nodes during capture for correct replay zeroing.
        DSP_DIAG(MEMORY, "capture: external inputs already device-actual via syncToDevice+readSpecial "
                         "(NO writeSpecial poisoning). Internal outputs NOT marked — nullify() records memset nodes");

        // Query node count mid-capture to verify operations are being recorded
        size_t midCaptureNodes = handle->getNumNodesDuringCapture(cudaStr);
        DSP_DIAG(EXECUTE, "Triton capture mid-check: %zu nodes recorded before executeSegment",
                 midCaptureNodes);

        // Snapshot all buffer addresses at capture entry — compare with replay to detect stale pointers
        {
          std::vector<void*> outAddrs, extAddrs;
          extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
          extractDeviceAddrs(externalArrays, numExt, extAddrs);
          DspDiagnostics::getInstance().clearAddressSnapshots();
          DSP_DIAG_SNAPSHOT_ADDRS("capture-entry", outAddrs.data(), totalOutputSlots_,
                                  extAddrs.data(), numExt);
        }

        auto captureStatus = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                     outputSlots_, totalOutputSlots_, stream);
        tl_graphExecutionActive = false;

        // Snapshot addresses AFTER capture execution to detect pointer changes during capture
        {
          std::vector<void*> outAddrs, extAddrs;
          extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
          extractDeviceAddrs(externalArrays, numExt, extAddrs);
          DSP_DIAG_SNAPSHOT_ADDRS("capture-exit", outAddrs.data(), totalOutputSlots_,
                                  extAddrs.data(), numExt);
          int changed = DSP_DIAG_COMPARE_ADDRS("capture-entry", "capture-exit");
          if (changed > 0) {
            DSP_DIAG(EXECUTE, "WARNING: %d buffer addresses CHANGED during capture execution!", changed);
          }
        }

        // Diagnostic: capture workspace usage
        DSP_DIAG(MEMORY, "capture workspace used: %zu / %zu bytes (%.1f%%)",
                 tl_captureWorkspaceOffset, seg.exec.replayHandle->getWorkspaceBytes(),
                 seg.exec.replayHandle->getWorkspaceBytes() > 0 ? (100.0 * tl_captureWorkspaceOffset / seg.exec.replayHandle->getWorkspaceBytes()) : 0.0);
        // Check for CUDA errors generated during capture — these become invalid graph nodes.
        // Don't use cudaGetLastError (which clears) — peek first for diagnostics.
        {
          cudaError_t capPhaseErr = cudaPeekAtLastError();
          if (capPhaseErr != cudaSuccess) {
            DSP_DIAG(BACKEND, "WARNING - CUDA error during Triton capture phase: %s (%d)",
                     cudaGetErrorString(capPhaseErr), (int)capPhaseErr);
            // Clear it so endCapture can proceed (the graph may still be partially valid)
            cudaGetLastError();
          }
        }

        // Query node count after execution to see how many ops were captured
        size_t postExecNodes = 0;
        {
          cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
          cudaGraph_t capGraph = nullptr;
          unsigned long long capId = 0;
          auto capErr = cudaStreamGetCaptureInfo_v2(cudaStr, &capStat, &capId, &capGraph, nullptr, nullptr);
          if (capErr == cudaSuccess && capGraph != nullptr) {
            cudaGraphGetNodes(capGraph, nullptr, &postExecNodes);
          }
        }
        DSP_DIAG(EXECUTE, "Triton capture post-exec: %zu nodes, captureStatus=%d",
                 postExecNodes, static_cast<int>(captureStatus));

        bool endOk = false;
        if (captureStatus == Status::OK) {
          endOk = handle->endCapture(cudaStr);
        } else {
          DSP_DIAG(EXECUTE, "FATAL: Triton capture execution FAILED status=%d for seg[%d-%d]. "
                            "BLOCKING EXECUTION.",
                   static_cast<int>(captureStatus), seg.def.startSlot, seg.def.endSlot);
          fflush(stdout); fflush(stderr);
          if (handle->isCapturing()) {
            handle->endCapture(cudaStr);
          }
        }

        if (endOk) {
          size_t numGraphNodes = handle->getNumNodes();
          int segSize = seg.def.endSlot - seg.def.startSlot + 1;
          DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                       "GRAPH CAPTURE COMPLETE: seg[%d-%d] %zu nodes captured from %d slots (%.1f nodes/slot)",
                       seg.def.startSlot, seg.def.endSlot, numGraphNodes, segSize,
                       segSize > 0 ? (double)numGraphNodes / segSize : 0.0);
          DSP_DIAG(EXECUTE, "Triton capture endOk: graph has %zu nodes", numGraphNodes);

          // Near-empty graphs have almost no GPU work — replay would skip the vast
          // majority of ops, producing wrong results. A graph with < 5% of the segment's
          // ops as nodes means most ops were gap-skipped during capture and aren't in the
          // graph. Mark as non-capturable so future executions fall back to slot-by-slot.
          double nodeRatio = segSize > 0 ? (double)numGraphNodes / segSize : 0.0;
          if (numGraphNodes == 0 || (segSize > 10 && nodeRatio < 0.05)) {
            DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                         "near-empty Triton graph for seg[%d-%d] (%zu nodes from %d slots, "
                         "ratio=%.2f) — marking as non-capturable",
                         seg.def.startSlot, seg.def.endSlot, numGraphNodes, segSize, nodeRatio);
            seg.exec.compilationFailed = true;
            cleanupCaptureTls(true);
            if (didPushCtx) {
              CUcontext dummy;
              cuCtxPopCurrent(&dummy);
              CUdevice cuDev;
              cuDeviceGet(&cuDev, tritonCaptureDevice);
              cuDevicePrimaryCtxRelease(cuDev);
            }
            restoreCublasWorkspaceAfterCapture(stream);
            restoreCaptureSlotState();
            platformCleanupSegmentForRebuild(seg);
            seg.exec.executionCount++;
            return Status::OK;
          }

          // Sample final output AFTER endCapture (stream no longer capturing, safe)
          if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
            auto* finalOut = outputSlots_[seg.def.endSlot];
            if (finalOut->dataType() == FLOAT32) {
              DSP_DIAG_DUMP_SLOT("capture-post-endCapture", seg.def.endSlot,
                                 DSP_BUF(finalOut), finalOut->lengthOf());
            }
          }
          // Dump top logit from capture execution via DSP_DIAG
          // Use outputSlotIndices[0] to get the ACTUAL final output slot
          // (matches GRAPH_REPLAY logic for apples-to-apples comparison)
          {
            int captureOutputSlot = -1;
            if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
              captureOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
            }
            if (captureOutputSlot < 0 || captureOutputSlot >= totalOutputSlots_) {
              captureOutputSlot = seg.def.endSlot;
            }
            if (captureOutputSlot >= 0 && captureOutputSlot < totalOutputSlots_ &&
                outputSlots_[captureOutputSlot] != nullptr) {
              auto* out = outputSlots_[captureOutputSlot];
              if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
                DSP_DIAG_DUMP_SEG_OUTPUT("CAPTURE_EXEC", captureOutputSlot, DSP_BUF(out),
                                         out->lengthOf(), seg.exec.executionCount, stream);
              }
            }
          }
        }

        if (endOk) {
          auto stats = handle->getStatistics();
          DSP_DIAG(EXECUTE, "Triton graph stats: %d kernels, %d memcpys, %d memsets, "
                            "%d memAllocs, %d memFrees, %d hostCallbacks, %d events, %d empty",
                   stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                   stats.numMemAllocs, stats.numMemFrees,
                   stats.numHostCallbacks, stats.numEvents, stats.numEmpty);
          fflush(stdout); fflush(stderr);
          if (stats.numMemAllocs > 0 || stats.numMemFrees > 0) {
            DSP_DIAG(EXECUTE, "Triton graph has %d MemAlloc + %d MemFree nodes "
                              "(paired alloc/free from cuBLAS internal workspace - CUDA 12+ handles these on replay).",
                     stats.numMemAllocs, stats.numMemFrees);
          }
          if (stats.numHostCallbacks > 0) {
            DSP_DIAG(BACKEND, "WARNING - Graph has %d host callback nodes!",
                     stats.numHostCallbacks);
          }
        }

        // Skip DOT dump by default for Triton graphs — cudaGraphDebugDotPrint with verbose
        // flags may also call cudaGraphKernelNodeGetParams internally, causing the same
        // cudaErrorInvalidDeviceFunction poisoning as getDetailedNodeInfo().
        if (endOk && Environment::getInstance().tritonDumpGraphDot()) {
          cudaGraphDebugDotPrint(handle->getGraph(), "/tmp/triton_graph_debug.dot", 0);
          DSP_DIAG(EXECUTE, "Triton graph dumped to /tmp/triton_graph_debug.dot");
          fflush(stdout); fflush(stderr);
        }

        // Skip getDetailedNodeInfo() for Triton graphs — it calls cudaFuncGetName on each
        // kernel node, which returns cudaErrorInvalidDeviceFunction (error 98) for Triton
        // kernels loaded via cuModuleLoadDataEx (driver API). The 658+ consecutive errors
        // poison the CUDA error state and cause cudaGraphLaunch to SIGSEGV.
        // Use getNumNodes() for basic stats instead (no per-node introspection).
        bool allKernelsValid = true;
        if (endOk) {
#ifdef SD_CUDA
          size_t totalNodes = handle->getNumNodes();
          DSP_DIAG(EXECUTE, "Triton graph has %zu nodes (skipping per-node inspection to avoid error-98 poisoning)",
                   totalNodes);
          fflush(stdout); fflush(stderr);
          // Ensure no sticky errors before instantiation
          cudaGetLastError();
#endif
        }

        bool instantiateOk = endOk && allKernelsValid && handle->instantiate();
        if (instantiateOk) {
          DSP_DIAG(EXECUTE, "Triton graph instantiated OK (graphExec=%p), about to launch...",
                   handle->getGraphExec());
          fflush(stdout); fflush(stderr);
        }

        if (!instantiateOk) {
          int deviceId = 0;
          cudaGetDevice(&deviceId);

          // Check if instantiation failed due to OOM — retry with eviction if possible.
          auto* cudaReplayForOom = seg.exec.replayHandle
                                   ? dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get()) : nullptr;
          bool isOom = cudaReplayForOom && cudaReplayForOom->wasLastInstantiateOom();
          if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
            seg.exec.captureOomRetries++;
            seg.exec.captureRetryAfterExec = seg.exec.executionCount + GraphSegment::retryInterval();
            DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                         "INSTANTIATE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                         seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                         seg.exec.captureRetryAfterExec);

            // Evict LRU graphs to free memory for the next attempt
            evictLruGraphs(segIdx, TRITON_CAPTURE_WORKSPACE_SIZE, stream);

            // Cleanup this failed attempt but do NOT set compilationFailed
            cleanupCaptureTls(true);
            if (didPushCtx) {
              CUcontext dummy;
              cuCtxPopCurrent(&dummy);
              CUdevice cuDev;
              cuDeviceGet(&cuDev, tritonCaptureDevice);
              cuDevicePrimaryCtxRelease(cuDev);
            }
            restoreCublasWorkspaceAfterCapture(stream);
            restoreCaptureSlotState();
            platformCleanupSegmentForRebuild(seg);
            cudaGetLastError();  // Clear sticky error
            // OOM during graph instantiation — throw instead of silently falling back
            // to slot-by-slot. The eviction above freed memory; the next execution
            // attempt (deferred by captureRetryAfterExec) will retry capture.
            // Silently producing output via slot-by-slot masks the OOM and the caller
            // never knows the graph wasn't captured.
            {
              char buf[256];
              snprintf(buf, sizeof(buf),
                       "NativeDSP: graph instantiation OOM for seg[%d-%d] on device %d "
                       "(retry %d/%d, retryAfterExec=%d). Evicted LRU graphs. "
                       "Fix memory pressure — do NOT fall back to slot-by-slot.",
                       seg.def.startSlot, seg.def.endSlot, deviceId,
                       seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                       seg.exec.captureRetryAfterExec);
              DSP_DIAG(COMPILE, "%s", buf);
              THROW_EXCEPTION(buf);
            }
          }

          // Not OOM or retries exhausted — permanent failure
          cleanupCaptureTls(true);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          restoreCaptureSlotState();
          platformCleanupSegmentForRebuild(seg);
          return reportCaptureError(seg, "instantiate", cudaGetLastError(), deviceId);
        }

        // POST-INSTANTIATE MEMORY CHECK removed: the validation launch immediately
        // below will reveal if the graph is usable. No speculative memory gate needed.

        // Graph instantiated — launch to validate the graph is not corrupted.
        // Warmup results are restored from savedWarmupOutputSlots below regardless.
        bool launchOk = false;
        {
          int deviceId = 0;
          cudaGetDevice(&deviceId);
          cudaGetLastError();
          bool replayResult = seg.exec.replayHandle->replay(stream);
          if (!replayResult) {
            cleanupCaptureTls(true);
            if (didPushCtx) {
              CUcontext dummy;
              cuCtxPopCurrent(&dummy);
              CUdevice cuDev;
              cuDeviceGet(&cuDev, tritonCaptureDevice);
              cuDevicePrimaryCtxRelease(cuDev);
            }
            restoreCublasWorkspaceAfterCapture(stream);
            restoreCaptureSlotState();
            platformCleanupSegmentForRebuild(seg);
            return reportReplayError(seg, "validation_launch", cudaGetLastError(), deviceId);
          }
          cudaError_t syncErr = cudaStreamSynchronize(cudaStr);
          if (syncErr != cudaSuccess) {
            cleanupCaptureTls(true);
            if (didPushCtx) {
              CUcontext dummy;
              cuCtxPopCurrent(&dummy);
              CUdevice cuDev;
              cuDeviceGet(&cuDev, tritonCaptureDevice);
              cuDevicePrimaryCtxRelease(cuDev);
            }
            restoreCublasWorkspaceAfterCapture(stream);
            restoreCaptureSlotState();
            platformCleanupSegmentForRebuild(seg);
            return reportReplayError(seg, "validation_sync", syncErr, deviceId);
          }
          DSP_DIAG(EXECUTE, "VALIDATION LAUNCH OK: seg[%d-%d] graph launched and synced successfully",
                   seg.def.startSlot, seg.def.endSlot);
          // LRU tracking: record when this segment was last replayed for eviction ordering
          seg.exec.lastReplayExecCount = executeCount_;
          launchOk = true;
        }

        if (launchOk) {
          if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
            auto* finalOut = outputSlots_[seg.def.endSlot];
            if (finalOut->dataType() == FLOAT32) {
              DSP_DIAG_DUMP_SLOT("capture-post-launch", seg.def.endSlot,
                                 DSP_BUF(finalOut), finalOut->lengthOf());
            }
          }
          // Dump top logit from first replay (graph launch after capture) via DSP_DIAG
          if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
            auto* out = outputSlots_[seg.def.endSlot];
            if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
              DSP_DIAG_DUMP_SEG_OUTPUT("REPLAY_LAUNCH", seg.def.endSlot, DSP_BUF(out),
                                       out->lengthOf(), seg.exec.executionCount, stream);
            }
          }
          // replayHandle already set (created before capture began)
          seg.exec.cachedShapeKey = segShapeKey;
          seg.exec.capturedInputAddrKey = segInputAddrKey;
          seg.exec.capturedCreateValueKey = createValueKey;
          seg.exec.capturedSlotAddrHash = computeSlotAddrHash(
              outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
          snapshotExternalAddrs(seg, externalArrays, numExt);

          // No actuality reset needed — writeSpecial() is no longer called during
          // capture, so external input actuality flags remain in their natural bi-actual
          // state (isPrimaryActual()=true AND isSpecialActual()=true via readSpecial).
          DSP_DIAG(MEMORY, "MONOLITHIC_CAPTURE: no actuality reset needed — writeSpecial not called");

          // Export graph stats and DOT file for diagnostics
          auto stats = handle->getStatistics();
          DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton graph CAPTURED and launched for seg[%d-%d]: "
                                                   "%d kernels, %d memcpy, %d memset, %d memAlloc, %d memFree "
                                                   "(workspace=%zuMB, offset=%zu)",
                       seg.def.startSlot, seg.def.endSlot,
                       stats.numKernels, stats.numMemcpyH2D + stats.numMemcpyD2H + stats.numMemcpyD2D,
                       stats.numMemsets, stats.numMemAllocs, stats.numMemFrees,
                       seg.exec.replayHandle->getWorkspaceBytes() / (1024*1024), tl_captureWorkspaceOffset);
          // Dump H2D memcpy node details to identify the source of baked-in host addresses
          if (stats.numMemcpyH2D > 0 && DSP_DIAG_ENABLED(EXECUTE)) {
            size_t numGraphNodes = 0;
            cudaGraphGetNodes(handle->getGraph(), nullptr, &numGraphNodes);
            if (numGraphNodes > 0) {
              std::vector<cudaGraphNode_t> graphNodes(numGraphNodes);
              cudaGraphGetNodes(handle->getGraph(), graphNodes.data(), &numGraphNodes);
              for (size_t ni = 0; ni < numGraphNodes; ni++) {
                cudaGraphNodeType nodeType;
                cudaGraphNodeGetType(graphNodes[ni], &nodeType);
                if (nodeType == cudaGraphNodeTypeMemcpy) {
                  cudaMemcpy3DParms mcpyParams;
                  memset(&mcpyParams, 0, sizeof(mcpyParams));
                  if (cudaGraphMemcpyNodeGetParams(graphNodes[ni], &mcpyParams) == cudaSuccess) {
                    size_t bytes = mcpyParams.extent.width *
                                   std::max(mcpyParams.extent.height, (size_t)1) *
                                   std::max(mcpyParams.extent.depth, (size_t)1);
                    const char* kindStr = (mcpyParams.kind == cudaMemcpyHostToDevice) ? "H2D" :
                                          (mcpyParams.kind == cudaMemcpyDeviceToDevice) ? "D2D" :
                                          (mcpyParams.kind == cudaMemcpyDeviceToHost) ? "D2H" : "other";
                    DSP_DIAG(EXECUTE, "GRAPH_NODE[%zu] MEMCPY %s: %zu bytes src=%p dst=%p "
                                      "seg[%d-%d]",
                             ni, kindStr, bytes,
                             mcpyParams.srcPtr.ptr, mcpyParams.dstPtr.ptr,
                             seg.def.startSlot, seg.def.endSlot);
                  }
                }
              }
            }
          }
          // Write DOT file for offline analysis.
          // Default: non-verbose (flag 0). Verbose queries kernel node params via
          // cudaFuncGetName, which returns cudaErrorInvalidDeviceFunction for
          // Triton CUfunction handles and may poison driver state.
          // Enable via ND4J_TRITON_GRAPH_DOT_VERBOSE=1 for debugging.
          {
            std::string dotPath = "/tmp/triton_graph_captured.dot";
            unsigned int dotFlags = Environment::getInstance().tritonGraphDotVerbose()
                                    ? cudaGraphDebugDotFlagsVerbose : 0;
            auto dotErr = cudaGraphDebugDotPrint(handle->getGraph(), dotPath.c_str(), dotFlags);
            if (dotErr == cudaSuccess) {
              DSP_DIAG(EXECUTE, "Exported Triton graph DOT to %s (verbose=%d)",
                       dotPath.c_str(), dotFlags != 0);
            }
            cudaGetLastError(); // Clear any error from dot print
          }
          // Write stats to a file the test can read
          {
            FILE* f = fopen("/tmp/triton_graph_stats.txt", "w");
            if (f) {
              fprintf(f, "segment=%d-%d\n", seg.def.startSlot, seg.def.endSlot);
              fprintf(f, "kernels=%d\n", stats.numKernels);
              fprintf(f, "memcpyH2D=%d\n", stats.numMemcpyH2D);
              fprintf(f, "memcpyD2H=%d\n", stats.numMemcpyD2H);
              fprintf(f, "memcpyD2D=%d\n", stats.numMemcpyD2D);
              fprintf(f, "memsets=%d\n", stats.numMemsets);
              fprintf(f, "memAllocs=%d\n", stats.numMemAllocs);
              fprintf(f, "memFrees=%d\n", stats.numMemFrees);
              fprintf(f, "hostCallbacks=%d\n", stats.numHostCallbacks);
              fprintf(f, "events=%d\n", stats.numEvents);
              fprintf(f, "childGraphs=%d\n", stats.numChildGraphs);
              fprintf(f, "totalNodes=%zu\n", handle->getNumNodes());
              fclose(f);
            }
          }
          status = Status::OK;
          usedTritonGraphCapture = true;

          // Phase 2: outputSlots_ == outputSlots_ (unified).
          // No need to sync cache ← output — they are the same pointer.

          // FORCE_RECAPTURE: invalidate graph immediately after capture+launch
          // so the NEXT step also re-captures instead of replaying a stale graph.
          // This ensures every single step is a fresh capture+launch with zero replays.
          if (Environment::getInstance().tritonForceRecapture()) {
            platformCleanupSegmentForRebuild(seg);
            seg.exec.argTableStable = false;
            batchD2DCount_ = 0;
            seg.exec.capturedInputAddrKey = 0;
            DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after capture+launch execCount=%d", seg.exec.executionCount);
          }
        } else {
          int deviceId = 0;
          cudaGetDevice(&deviceId);
          cleanupCaptureTls(true);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          restoreCaptureSlotState();
          platformCleanupSegmentForRebuild(seg);
          return reportCaptureError(seg, "execute_during_capture", cudaGetLastError(), deviceId);
        }
      } else {
        int deviceId = 0;
        cudaGetDevice(&deviceId);

        // Check if beginCapture failed due to OOM — retry with eviction if possible.
        cudaError_t beginErr = cudaGetLastError();
        bool isOom = (beginErr == cudaErrorMemoryAllocation);
        if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
          seg.exec.captureOomRetries++;
          seg.exec.captureRetryAfterExec = seg.exec.executionCount + GraphSegment::retryInterval();
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                       "BEGIN_CAPTURE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                       seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                       seg.exec.captureRetryAfterExec);
          evictLruGraphs(segIdx, TRITON_CAPTURE_WORKSPACE_SIZE, stream);
          cleanupCaptureTls(true);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          restoreCaptureSlotState();
          platformCleanupSegmentForRebuild(seg);
          // OOM during beginCapture — throw instead of silently falling back to slot-by-slot.
          {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "NativeDSP: beginCapture OOM for seg[%d-%d] on device %d "
                     "(retry %d/%d, retryAfterExec=%d). Evicted LRU graphs. "
                     "Fix memory pressure — do NOT fall back to slot-by-slot.",
                     seg.def.startSlot, seg.def.endSlot, deviceId,
                     seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                     seg.exec.captureRetryAfterExec);
            DSP_DIAG(COMPILE, "%s", buf);
            THROW_EXCEPTION(buf);
          }
        }

        cleanupCaptureTls(true);
        if (didPushCtx) {
          CUcontext dummy;
          cuCtxPopCurrent(&dummy);
          CUdevice cuDev;
          cuDeviceGet(&cuDev, tritonCaptureDevice);
          cuDevicePrimaryCtxRelease(cuDev);
        }
        restoreCublasWorkspaceAfterCapture(stream);
        restoreCaptureSlotState();
        platformCleanupSegmentForRebuild(seg);
        return reportCaptureError(seg, "beginCapture", beginErr, deviceId);
      }

      // ── Restore warmup output slots for downstream segment visibility ──
      // This MUST happen regardless of capture success/failure. During capture
      // execution, ops allocate outputs from capture workspace, overwriting
      // outputSlots_[] with workspace addresses. If capture fails (endCapture
      // error, instantiate error, or execution error), outputSlots_[] still
      // has the stale workspace addresses. Downstream segments reading from
      // these get garbage data → NaN propagation.
      //
      // No WARMUP_RESTORE needed: gap ops were skipped during capture, so
      // outputSlots_[] was never overwritten. Warmup data is intact.

      DSP_DIAG(EXECUTE, "CAPTURE_COMPLETE: seg[%d-%d] hasReplay=%d compilationFailed=%d "
                        "numCaptureBuffers=%d",
               seg.def.startSlot, seg.def.endSlot,
               seg.exec.replayHandle != nullptr,
               seg.exec.compilationFailed,
               0);

      // No external/cross-slot rewiring is needed now that replay uses the
      // canonical external and output buffers directly. The restore loops remain
      // harmless no-ops because the saved lists stay empty.
      for (auto& [extIdx, origArr] : savedExtForCapture) {
        externalArrays[extIdx] = origArr;
      }

      // Restore cross-segment output slots to warmup pointers.
      // The producing segment's replay writes fresh data to the warmup array's
      // GPU address (baked during capture). The D2D copy before the consuming
      // segment's replay reads from outputSlots_[] (warmup pointer, with fresh
      // GPU data from the producing segment's replay).
      for (auto& [slotIdx, origArr] : savedSlotsForCapture) {
        if (origArr != nullptr) {
          outputSlots_[slotIdx] = origArr;
        }
      }

      // Restore primary CUDA context if we pushed it
      if (didPushCtx) {
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        CUdevice cuDev;
        cuDeviceGet(&cuDev, tritonCaptureDevice);
        cuDevicePrimaryCtxRelease(cuDev);
      }

      // Restore cuBLAS workspace to default (undo setCublasWorkspaceForCapture)
      restoreCublasWorkspaceAfterCapture(stream);

      // Reset thread-local state after capture attempt
      tl_captureWorkspace = nullptr;
      tl_captureWorkspaceSize = 0;
      tl_captureWorkspaceOffset = 0;
      // Reset host workspace thread-locals (ownership moves to tl_capturedHostPtrs → replay handle)
      tl_captureHostWorkspace = nullptr;
      tl_captureHostWorkspaceSize = 0;
      tl_captureHostWorkspaceOffset = 0;
      tl_graphCaptureStream = prevCaptureStream;
      // Pinned host ptrs: graph's H2D memcpy nodes reference these on replay.
      // On success: move to segment so they persist for graph lifetime.
      // On failure: free immediately (no graph to replay).
      if (usedTritonGraphCapture && seg.exec.replayHandle) {
        for (auto* ptr : tl_capturedHostPtrs) {
          seg.exec.replayHandle->addCapturedHostPtr(ptr);
        }
        DSP_DIAG(MEMORY, "preserved %zu pinned host ptrs for Triton graph replay",
                 seg.exec.replayHandle->getCapturedHostPtrs().size());
      } else {
        // No graph captured — free pinned host ptrs immediately
        for (auto* ptr : tl_capturedHostPtrs) {
          cudaFreeHost(ptr);
        }
      }
      tl_capturedHostPtrs.clear();
      tl_captureReplicateCache.clear();

      // Arrays persist — no pendingClose_ flush needed after capture

      // Restore frozen context state so subsequent executions (including graph replay
      // steps that fall through to direct execution) use the frozen fast path.
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        slots_[s].state_ = savedSlotStateTriton[s - seg.def.startSlot];
      }
    }  // end else (replayHandle != nullptr — workspace allocation succeeded)

  }  // end if (!didCompositeCapture) — monolithic capture only for non-composite segments

  }  // end if (shouldCaptureTritonGraphNow)

#endif

  if (!usedTritonGraphCapture) {
    // Cross-stream ordering for direct execution (same issue as capture path).
    // Java-side assign() on the default stream needs to complete before Triton
    // reads device buffers. syncToDevice() below is a no-op when sAct=true.
#ifdef SD_CUDA
    {
      cudaStream_t defaultStream = nullptr;
      auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
      if (defaultStreamPtr != nullptr) defaultStream = *defaultStreamPtr;
      if (defaultStream != nullptr && defaultStream != cudaStr) {
        cudaEvent_t evt = getCrossStreamEvent();
        cudaEventRecord(evt, defaultStream);
        cudaStreamWaitEvent(cudaStr, evt, 0);
      }
    }
#endif

    // ── Sync external inputs to device BEFORE Triton segment execution ──
    // Triton's arg table population uses specialBuffer() to resolve GPU pointers.
    // specialBuffer() only calls syncToDevice() when the device buffer is nullptr
    // or on the wrong device — it does NOT check if the device data is stale.
    // Java modifies external inputs (attention_mask, position_ids, input_ids) on the
    // host via putScalar() + tagLocation(HOST), making the device data stale.
    // Native ops handle this via prepareSpecialUse() which calls syncToDevice()
    // unconditionally, but Triton bypasses native ops and reads device buffers directly.
    //
    // Use forceSync=true for variable external inputs because:
    // - Triton sub-kernels call readSpecial() on inputs (TritonGraphBackend_kernel.cu:988)
    // - markOrderedRangeDeviceCurrent() calls readSpecial() on gap op inputs
    // - These readSpecial() calls can leave isSpecialActual()=true even after Java
    //   modifies the host buffer with writePrimary(), if the readSpecial happens
    //   to bump _readSpecial higher than _writePrimary in the actuality counter
    //   race between the previous step's C++ execution and this step's Java writes.
    // - forceSync=true bypasses the actuality check, matching compositeReplay behavior.
    if (shapesFrozen_ && !externalInputIsVariable_.empty()) {
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        bool isVariable = ei < static_cast<int>(externalInputIsVariable_.size()) &&
                          externalInputIsVariable_[ei];
        if (Environment::getInstance().tritonVerifyKernels()) {
          auto* db = externalArrays[ei]->dataBuffer();
          DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(direct) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p isVariable=%d",
                   -(ei + 1), ei,
                   db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                   db ? (db->isSpecialActual() ? 1 : 0) : -1,
                   (long long)externalArrays[ei]->lengthOf(),
                   DSP_BUF(externalArrays[ei]),
                   isVariable ? 1 : 0);
        }
        // Variable external inputs MUST use forceSync=true because
        // markOrderedRangeDeviceCurrent() calls readSpecial() on inputs,
        // poisoning actuality flags so that isSpecialActual() returns true
        // even after Java writes new values via writePrimary(). Without
        // forceSync, syncToSpecial() skips the H2D copy and gap ops use
        // stale device data from the previous decode step.
        if (isVariable) {
          externalArrays[ei]->dataBuffer()->syncToSpecial(true);
        } else {
          externalArrays[ei]->syncToDevice();
        }
      }
    } else {
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] != nullptr) {
          if (Environment::getInstance().tritonVerifyKernels()) {
            auto* db = externalArrays[ei]->dataBuffer();
            DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(direct) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p",
                     -(ei + 1), ei,
                     db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                     db ? (db->isSpecialActual() ? 1 : 0) : -1,
                     (long long)externalArrays[ei]->lengthOf(),
                     DSP_BUF(externalArrays[ei]));
          }
          externalArrays[ei]->syncToDevice();
        }
      }
    }

    // NOTE: Do NOT set tl_graphExecutionActive=true here for non-capture Triton execution.
    // That flag suppresses syncToPrimary (D2H transfers), error checking, and
    // PointersManager sync -- behaviors only appropriate during CUDA graph capture.
    // The ordered range executor already handles capture detection independently:
    // it checks cudaStreamIsCapturing() and only sets tl_graphExecutionActive=true
    // when actually capturing. Setting it unconditionally here caused native ordered ops
    // (matmul, gather, etc.) to read stale host data, producing wrong output.

    // Disable frozen fast path for gap ops during Triton segment execution.
    // Same rationale as the capture path (lines 5325-5329): the pre-execution
    // slot restoration at lines 4955-5032 may replace NDArray objects in
    // outputSlots_[], making the frozen context's cached input/output pointers
    // stale. Without clearing frozenContextReady, gap ops write to old arrays
    // while downstream ops read from new arrays, producing wrong output.
    // Save and restore so subsequent executions still benefit from frozen fast path.
    std::vector<NativeSlot::SlotState> savedSlotStateNonCapture(seg.def.endSlot - seg.def.startSlot + 1);
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      savedSlotStateNonCapture[s - seg.def.startSlot] = slots_[s].state_;
      // Demote FROZEN→SHAPE_CACHED so gap ops don't use stale frozen contexts.
      // FROZEN_CONSTANT slots MUST be preserved: their output never changes and
      // prezeroSegmentOutputs relies on frozenConstantSlot() to skip zeroing them.
      // Demoting FROZEN_CONSTANT→SHAPE_CACHED causes prezero to wipe frozen constant
      // outputs with zeros, corrupting all downstream ops (stuck-token root cause).
      if (slots_[s].state_ == NativeSlot::SlotState::FROZEN)
        slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
    }

    // Snapshot addresses for direct execution (baseline for comparison with capture/replay)
    {
      std::vector<void*> outAddrs, extAddrs;
      extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
      extractDeviceAddrs(externalArrays, numExt, extAddrs);
      DSP_DIAG_SNAPSHOT_ADDRS("direct-entry", outAddrs.data(), totalOutputSlots_,
                              extAddrs.data(), numExt);
    }

    DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                 "direct-exec invoking prezeroSegmentOutputs seg=[%d-%d] stream=%p execCount=%d",
                 seg.def.startSlot, seg.def.endSlot, (void*)stream, seg.exec.executionCount);
    prezeroSegmentOutputs(seg, stream);

    try {
      status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                       outputSlots_, totalOutputSlots_, stream);
    } catch (...) {
      // Restore frozenContextReady on exception
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        slots_[s].state_ = savedSlotStateNonCapture[s - seg.def.startSlot];
      }
      throw;  // Re-throw after cleanup
    }

    // Restore frozen context state so subsequent calls use the frozen fast path
    // once context pointers are re-established by the normal path above.
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].state_ = savedSlotStateNonCapture[s - seg.def.startSlot];
    }


    // Dump final output for direct Triton path (baseline comparison)
    if (status == Status::OK && seg.def.endSlot < totalOutputSlots_ &&
        outputSlots_[seg.def.endSlot] != nullptr) {
      auto* finalOut = outputSlots_[seg.def.endSlot];
      if (finalOut->dataType() == FLOAT32) {
        DSP_DIAG_DUMP_SLOT("direct", seg.def.endSlot,
                           DSP_BUF(finalOut), finalOut->lengthOf());
      }
    }
    // Always-on diagnostic: dump top logit for non-capture Triton execution
    if (!usedTritonGraphCapture && status == Status::OK &&
        seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
      auto* out = outputSlots_[seg.def.endSlot];
      if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
        DSP_DIAG_DUMP_SEG_OUTPUT("DIRECT_TRITON", seg.def.endSlot, DSP_BUF(out),
                                 out->lengthOf(), seg.exec.executionCount, stream);
      }
    }
    // Diagnostic: segment exit argmax (only when DSP diagnostics enabled)
    if (DSP_DIAG_ENABLED(EXECUTE) && status == Status::OK) {
      int finalOutputSlot = -1;
      if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
        finalOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
      }
      if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_)
        finalOutputSlot = seg.def.endSlot;
      if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
          outputSlots_[finalOutputSlot] != nullptr) {
        auto* exitOut = outputSlots_[finalOutputSlot];
        if (exitOut->dataType() == FLOAT32 && exitOut->lengthOf() > 0) {
#ifdef SD_CUDA
          auto* cudaStrPtr = (stream != nullptr) ? static_cast<cudaStream_t*>(stream) : nullptr;
          if (cudaStrPtr) cudaStreamSynchronize(*cudaStrPtr);
          int exitArgmax = dspArgmax(DSP_BUF(exitOut), exitOut->dataType(), exitOut->lengthOf());
          DSP_DIAG(EXECUTE, "SEG_EXIT_ARGMAX seg[%d-%d] slot=%d argmax=%d capture=%d execCount=%d",
                   seg.def.startSlot, seg.def.endSlot, finalOutputSlot, exitArgmax,
                   usedTritonGraphCapture ? 1 : 0, seg.exec.executionCount);
#endif
        }
      }


      DSP_DIAG(EXECUTE, "executeSegmentWithGpuGraph: exec%d seg[%d-%d]: backend=%s %s status=%d(%s) "
                        "executionCount=%d compilationFailed=%d usedCapture=%d",
               seg.exec.executionCount, seg.def.startSlot, seg.def.endSlot,
               backendName, status == Status::OK ? "OK" : "FAILED",
               static_cast<int>(status), statusName_gpu(status),
               seg.exec.executionCount,
               seg.exec.compilationFailed ? 1 : 0, usedTritonGraphCapture ? 1 : 0);

      if (status == Status::OK) {
        seg.exec.executionCount++;
        totalGraphReplays_++;
        if (seg.exec.compiledByBackend.empty()) {
          seg.exec.compiledByBackend = backendName;
        }
      }

#ifdef SD_CUDA
      if (Environment::getInstance().tritonVerifyKernels()) {
        DSP_DIAG(VERIFY, "SEG_EXIT seg[%d-%d] status=%s execCount=%d",
                 seg.def.startSlot, seg.def.endSlot, statusName_gpu(status), seg.exec.executionCount);
      }
#endif

      return status;
    }

  }  // end if (!usedTritonGraphCapture)

}  // executeSegmentWithGpuGraph

}  // namespace graph
}  // namespace sd