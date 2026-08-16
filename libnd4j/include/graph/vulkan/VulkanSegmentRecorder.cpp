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

#include <graph/vulkan/VulkanSegmentRecorder.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanReplayHandle.h>
#include <graph/vulkan/VulkanKernelEmitterCatalog.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/RandomGenerator.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <graph/vulkan/VulkanPipelineCache.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/LegacyOpTypeCodes.h>
#include <graph/DspDiagnostics.h>
#include <array/NDArray.h>
#include <array/DataBuffer.h>
#include <array/DataTypeUtils.h>
#include <helpers/shape.h>
#include <ops/op_types.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <system/type_boilerplate.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace sd {
namespace graph {

// ─────────────────────────────────────────────────────────────────────────────
//  Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

VulkanSegmentRecorder::VulkanSegmentRecorder(
    VulkanReplayHandle* handle, GraphCompilationPolicy compilationPolicy)
    : handle_(handle), compilationPolicy_(std::move(compilationPolicy)) {}

VulkanSegmentRecorder::~VulkanSegmentRecorder() {
  if (handle_ == nullptr) return;
  VkDevice dev = handle_->getDevice();
  if (dev == VK_NULL_HANDLE) return;

  // The owning replay fence covers every dispatch that can reference these
  // recorder-local descriptors and buffers.
  if (handle_->waitForReplayIdle() != VK_SUCCESS) return;

  // Destroy the exactly-sized per-dispatch pools. Each destruction also frees
  // the descriptor set allocated from that pool.
  for (VkDescriptorPool pool : descriptorPools_) {
    if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, pool, nullptr);
  }
  descriptorPools_.clear();

  // Random-state buffers are recorder-owned. The replay fence above proves no
  // recorded dispatch can still reference them, so reclaim them immediately.
  VulkanMemoryPool& memoryPool = VulkanMemoryPool::getInstance();
  for (auto& binding : randomStateBindings_) {
    if (binding.operand.specialToken != nullptr) {
      memoryPool.freeImmediate(binding.operand.specialToken);
    }
  }
  randomStateBindings_.clear();

  // NDArray operand VkBuffers belong to VulkanMemoryPool/DataBuffer. Descriptor
  // sets only borrow them, exactly like a CUDA graph borrows stable pointers.
  bindings_.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

VulkanRandomStateWords randomStateWords(RandomGenerator& state) {
  const uint64_t root = static_cast<uint64_t>(state.rootState());
  const uint64_t node = static_cast<uint64_t>(state.nodeState());
  return {static_cast<uint32_t>(root), static_cast<uint32_t>(root >> 32),
          static_cast<uint32_t>(node), static_cast<uint32_t>(node >> 32)};
}

std::string mlirFloatLiteral(double value) {
  std::ostringstream literal;
  literal << std::scientific
          << std::setprecision(std::numeric_limits<double>::max_digits10)
          << value;
  return literal.str();
}

struct DispatchGeometry {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

using VulkanValidateCallback = bool (*)(const NativeSlot&, NDArray**, int,


                                        NDArray**, int, const VulkanDeviceCaps&);
using VulkanEmitCallback = std::string (*)(const NativeSlot&, NDArray**, int,
                                           NDArray**, int, const VulkanDeviceCaps&);
using VulkanGeometryCallback = bool (*)(const NativeSlot&, NDArray**, int,
                                        NDArray**, int, DispatchGeometry&);

struct VulkanOpHandler {
  VulkanValidateCallback validate;
  VulkanEmitCallback emitMlir;
  VulkanGeometryCallback dispatchGeometry;
};

struct VulkanPolicy {
  static constexpr bool matmul = false;
  static constexpr bool rmsNorm = false;
  static constexpr bool rope = false;
  static constexpr bool binary = false;
  static constexpr bool unary = false;
  static constexpr bool softmax = false;
  static constexpr bool layerNorm = false;
  static constexpr bool gather = false;
  static constexpr bool concat = false;
  static constexpr bool transpose = false;
  static constexpr bool transposeDefaultReverse = false;
  static constexpr bool movement = false;
  static constexpr bool split = false;
  static constexpr bool constant = false;
  static constexpr bool reduction = false;
  static constexpr bool structuredCompute = false;
  static constexpr bool multiOutputElementwise = false;
  static constexpr bool contractMovement = false;
  static constexpr bool batchedMatrixList = false;
  static constexpr bool indexedAccumulation = false;
  static constexpr bool indexedTadMovement = false;
  static constexpr bool ternary = false;
};
struct MatmulPolicy : VulkanPolicy { static constexpr bool matmul = true; };
struct RmsNormPolicy : VulkanPolicy { static constexpr bool rmsNorm = true; };
struct RopePolicy : VulkanPolicy { static constexpr bool rope = true; };
struct BinaryPolicy : VulkanPolicy { static constexpr bool binary = true; };
struct UnaryPolicy : VulkanPolicy { static constexpr bool unary = true; };
struct TernaryPolicy : VulkanPolicy { static constexpr bool ternary = true; };
struct SoftmaxPolicy : VulkanPolicy { static constexpr bool softmax = true; };
struct LayerNormPolicy : VulkanPolicy { static constexpr bool layerNorm = true; };
struct GatherPolicy : VulkanPolicy { static constexpr bool gather = true; };
struct ConcatPolicy : VulkanPolicy { static constexpr bool concat = true; };
struct TransposePolicy : VulkanPolicy {
  static constexpr bool transpose = true;
  static constexpr bool transposeDefaultReverse = true;
};
struct PermutePolicy : VulkanPolicy { static constexpr bool transpose = true; };
struct MovementPolicy : VulkanPolicy { static constexpr bool movement = true; };
struct SplitPolicy : VulkanPolicy { static constexpr bool split = true; };
struct ConstantPolicy : VulkanPolicy { static constexpr bool constant = true; };
struct ReductionPolicy : VulkanPolicy { static constexpr bool reduction = true; };
struct StructuredComputePolicy : VulkanPolicy { static constexpr bool structuredCompute = true; };
struct MultiOutputElementwisePolicy : VulkanPolicy {
  static constexpr bool multiOutputElementwise = true;
};
struct ContractMovementPolicy : VulkanPolicy {
  static constexpr bool contractMovement = true;
};
struct BatchedMatrixListPolicy : VulkanPolicy {
  static constexpr bool batchedMatrixList = true;
};
struct IndexedAccumulationPolicy : VulkanPolicy {
  static constexpr bool indexedAccumulation = true;
};
struct IndexedTadMovementPolicy : VulkanPolicy {
  static constexpr bool indexedTadMovement = true;
};

template <typename Policy>
static bool validateVulkanOp(const NativeSlot&, NDArray**, int, NDArray**, int,
                             const VulkanDeviceCaps&);
template <typename Policy>
static std::string emitVulkanOp(const NativeSlot&, NDArray**, int, NDArray**, int,
                                const VulkanDeviceCaps&);
template <typename Policy>
static bool vulkanDispatchGeometry(const NativeSlot&, NDArray**, int,
                                   NDArray**, int, DispatchGeometry&);

static bool validateCatalogOp(const NativeSlot&, NDArray**, int,
                              NDArray**, int, const VulkanDeviceCaps&);
static std::string emitCatalogOp(const NativeSlot&, NDArray**, int,
                                 NDArray**, int, const VulkanDeviceCaps&);
static bool catalogDispatchGeometry(const NativeSlot&, NDArray**, int,
                                    NDArray**, int, DispatchGeometry&);

static const VulkanKernelEmitterInfo* emitterForSlot(
    const NativeSlot& slot) {
  if (slot.legacy.legacyOpType != LEGACY_NOT_SET) {
    const auto family =
        vulkanLegacyFamilyFromTypeCode(slot.legacy.legacyOpType);
    return family.has_value()
               ? findVulkanLegacyKernelEmitter(*family,
                                               slot.legacy.legacyOpNum)
               : nullptr;
  }
  return findVulkanKernelEmitter(slot.ident.opHash);
}

static std::string emitterIdentityAttributes(const NativeSlot& slot) {
  std::ostringstream attributes;
  if (slot.legacy.legacyOpType != LEGACY_NOT_SET) {
    const auto family =
        vulkanLegacyFamilyFromTypeCode(slot.legacy.legacyOpType);
    if (!family.has_value()) return "";
    attributes << "nd4j.legacy_family = "
               << static_cast<int>(*family)
               << " : i32, nd4j.legacy_op_num = "
               << slot.legacy.legacyOpNum << " : i32";
  } else {
    attributes << "nd4j.op_hash = "
               << static_cast<long long>(slot.ident.opHash) << " : i64";
  }
  return attributes.str();
}

static std::string emitterPipelineKey(const NativeSlot& slot) {
  if (slot.legacy.legacyOpType != LEGACY_NOT_SET) {
    const auto family =
        vulkanLegacyFamilyFromTypeCode(slot.legacy.legacyOpType);
    if (!family.has_value()) return "";
    return "legacy:" + std::to_string(static_cast<int>(*family)) + ":" +
           std::to_string(slot.legacy.legacyOpNum);
  }
  return "descriptor:" + std::to_string(slot.ident.opHash);
}

static const VulkanOpHandler* findVulkanHandler(const NativeSlot& slot) {
  if (emitterForSlot(slot) == nullptr) return nullptr;
  static const VulkanOpHandler handler{
      &validateCatalogOp, &emitCatalogOp, &catalogDispatchGeometry};
  return &handler;
}

template <typename T>
struct VulkanFloatType {
  using AccT = typename simdOps::AggregateType<T>::type;

  static bool supported(const VulkanDeviceCaps& caps) {
    if constexpr (std::is_same_v<T, bfloat16>) {
      return false;
    } else if constexpr (sizeof(T) * CHAR_BIT == 16) {
      return caps.fp16 && caps.storage16;
    } else if constexpr (sizeof(T) * CHAR_BIT == 32) {
      return true;
    } else if constexpr (sizeof(T) * CHAR_BIT == 64) {
      return caps.fp64;
    }
    return false;
  }

  static std::string storageType() {
    if constexpr (std::is_same_v<T, bfloat16>) {
      return "bf16";
    }
    return "f" + std::to_string(sizeof(T) * CHAR_BIT);
  }

  static std::string accumulatorType() {
    return "f" + std::to_string(sizeof(AccT) * CHAR_BIT);
  }
};

template <typename T>
static void selectMlirFloatTypes(const VulkanDeviceCaps& caps,
                                 std::string& storageType,
                                 std::string& accumulatorType) {

  if (!VulkanFloatType<T>::supported(caps)) return;
  storageType = VulkanFloatType<T>::storageType();

  accumulatorType = VulkanFloatType<T>::accumulatorType();
}

static bool isFrameworkFloatSelectorType(sd::DataType dataType) {
  return DataTypeUtils::isR(dataType) &&
         DataTypeUtils::sizeOfElement(dataType) * CHAR_BIT >= 16;
}

static bool selectMlirFloatTypes(sd::DataType dataType,
                                 const VulkanDeviceCaps& caps,
                                 std::string& storageType,
                                 std::string& accumulatorType) {
  storageType.clear();
  accumulatorType.clear();
  if (!isFrameworkFloatSelectorType(dataType)) return false;
  BUILD_SINGLE_SELECTOR(dataType, selectMlirFloatTypes,
                        (caps, storageType, accumulatorType), SD_FLOAT_TYPES);
  return !storageType.empty();
}

template <typename I>
static void selectMlirIntegerTypes(const VulkanDeviceCaps& caps,
                                   std::string& storageType,
                                   std::string& accumulatorType,
                                   bool& isUnsigned) {
  if constexpr (std::is_integral_v<I> && !std::is_same_v<I, bool> &&
                (sizeof(I) == 4 || sizeof(I) == 8)) {
    if constexpr (sizeof(I) == 8) {
      if (!caps.int64) return;
    }
    const auto bitWidth = sizeof(I) * CHAR_BIT;
    storageType = "i" + std::to_string(bitWidth);
    accumulatorType = storageType;
    isUnsigned = std::is_unsigned_v<I>;
  }
}

/// Select only storage ABIs whose required feature state is represented by
/// VulkanDeviceCaps. Float16/32/64 use the real feature gates above; signed and
/// unsigned 32-bit integers are core storage types, while 64-bit integer
/// storage is enabled only when shaderInt64 is available. BOOL uses its
/// byte-addressed storage ABI only when storageBuffer8BitAccess is enabled.
static bool selectMlirScalarTypes(sd::DataType dataType,
                                  const VulkanDeviceCaps& caps,
                                  std::string& storageType,
                                  std::string& accumulatorType,
                                  bool& isUnsigned) {
  storageType.clear();
  accumulatorType.clear();
  isUnsigned = false;
  if (isFrameworkFloatSelectorType(dataType)) {
    return selectMlirFloatTypes(
        dataType, caps, storageType, accumulatorType);
  }
  if (DataTypeUtils::isB(dataType)) {
    if (!caps.storage8) return false;
    storageType = "i8";
    accumulatorType = "i32";
    isUnsigned = true;
    return true;
  }
  if (!DataTypeUtils::isZ(dataType)) {
    return false;
  }
  BUILD_SINGLE_SELECTOR(dataType, selectMlirIntegerTypes,
                        (caps, storageType, accumulatorType, isUnsigned),
                        SD_INTEGER_TYPES);
  return !storageType.empty();
}

template <typename T>
static void selectFloatingScalarLimit(bool maximum, double& value,
                                      bool& supported) {
  if constexpr (!std::is_same_v<T, bfloat16>) {
    value = static_cast<double>(maximum ? DataTypeUtils::max<T>()
                                        : DataTypeUtils::min<T>());
    supported = true;
  }
}

template <typename I>
static void selectIntegerScalarLimit(bool maximum, double& value,
                                     bool& supported) {
  if constexpr (std::is_integral_v<I> && sizeof(I) == 4 &&
                !std::is_same_v<I, bool>) {
    value = static_cast<double>(maximum ? DataTypeUtils::max<I>()
                                        : DataTypeUtils::min<I>());
    supported = true;
  }
}

static bool supportedScalarLimit(sd::DataType dataType, bool maximum,
                                 double& value) {
  bool supported = false;
  if (isFrameworkFloatSelectorType(dataType)) {
    BUILD_SINGLE_SELECTOR(dataType, selectFloatingScalarLimit,
                          (maximum, value, supported), SD_FLOAT_TYPES);
  } else if (DataTypeUtils::isZ(dataType) && !DataTypeUtils::isB(dataType)) {
    BUILD_SINGLE_SELECTOR(dataType, selectIntegerScalarLimit,
                          (maximum, value, supported), SD_INTEGER_TYPES);
  }
  return supported;
}

template <typename I>
static void selectMlirIndexType(std::string& type) {
  static_assert(std::is_integral_v<I>, "Vulkan gather indices must be integral");
  type = "i" + std::to_string(sizeof(I) * CHAR_BIT);
}

static bool selectMlirIndexType(sd::DataType dataType, std::string& type) {
  type.clear();
  if (!DataTypeUtils::isZ(dataType) || DataTypeUtils::isB(dataType)) {
    return false;
  }
  BUILD_SINGLE_SELECTOR(dataType, selectMlirIndexType, (type), SD_INTEGER_TYPES);
  return type == "i32";
}

template <typename I>
static void selectVulkanIntegerDtypeBit(uint32_t& bit) {
  if constexpr (std::is_integral_v<I> && !std::is_same_v<I, bool>) {
    if constexpr (sizeof(I) == 4) {
      bit = std::is_signed_v<I> ? VULKAN_DTYPE_SIGNED_INT32
                                : VULKAN_DTYPE_UNSIGNED_INT32;
    } else {
      bit = VULKAN_DTYPE_INDEX;
    }
  }
}

static uint32_t vulkanDtypeBit(sd::DataType dataType) {
  if (DataTypeUtils::isR(dataType)) return VULKAN_DTYPE_FLOAT;
  if (DataTypeUtils::isB(dataType)) return VULKAN_DTYPE_BOOL;
  if (!DataTypeUtils::isZ(dataType)) return VULKAN_DTYPE_NONE;
  uint32_t bit = VULKAN_DTYPE_NONE;
  BUILD_SINGLE_SELECTOR(dataType, selectVulkanIntegerDtypeBit, (bit),
                        SD_INTEGER_TYPES);
  return bit;
}

template <typename I>
static void selectSignedIndexType(bool& supported) {
  supported = std::is_integral_v<I> && std::is_signed_v<I>;
}

static bool isSignedIntegerIndexType(sd::DataType dataType) {
  if (dataType != DataType::INT32) return false;
  bool supported = false;
  BUILD_SINGLE_SELECTOR(dataType, selectSignedIndexType, (supported), SD_INTEGER_TYPES);
  return supported;
}

/// Frozen argument storage must be internally consistent before any metadata is read.
static bool hasValidArgumentStorage(const NativeSlot& slot) {
  const auto& args = slot.args;
  return args.numIArgs >= 0 && (args.numIArgs == 0 || args.iArgs != nullptr) &&
         args.numTArgs >= 0 && (args.numTArgs == 0 || args.tArgs != nullptr) &&
         args.numBArgs >= 0 && (args.numBArgs == 0 || args.bArgs != nullptr) &&
         args.numDArgs >= 0 && (args.numDArgs == 0 || args.dArgs != nullptr) &&
         args.numSArgs >= 0 && (args.numSArgs == 0 || args.sArgs != nullptr);
}

/// DArgs are the framework's output-dtype override channel. Triton forwards
/// them through Context independently of an op's operational arguments. Vulkan
/// accepts the same contract only when every override agrees with the concrete
/// output allocated by framework shape inference.
static bool outputDataTypeArgumentsMatch(const NativeSlot& slot,
                                         NDArray** outputs, int numOut) {
  const auto& args = slot.args;
  if (args.numDArgs == 0) return true;
  if (outputs == nullptr || numOut <= 0 || args.dArgs == nullptr ||
      args.numDArgs != numOut) {
    return false;
  }
  for (int i = 0; i < numOut; ++i) {
    if (outputs[i] == nullptr || args.dArgs[i] != outputs[i]->dataType()) {
      return false;
    }
  }
  return true;
}

struct StaticRangeSpec {
  bool integerArguments = false;
  sd::LongType integerStart = 0;
  sd::LongType integerLimit = 0;
  sd::LongType integerDelta = 1;
  double valueStart = 0.0;
  double valueDelta = 1.0;
  sd::LongType length = 0;
};

static uint64_t signedMagnitude(sd::LongType value) {
  const uint64_t bits = static_cast<uint64_t>(value);
  return value < 0 ? uint64_t{0} - bits : bits;
}

static bool normalizeRollShift(sd::LongType shift, sd::LongType dimension,
                               sd::LongType& normalized) {
  if (dimension <= 0) return false;
  const uint64_t divisor = static_cast<uint64_t>(dimension);
  const uint64_t remainder = signedMagnitude(shift) % divisor;
  const uint64_t positive =
      shift < 0 && remainder != 0 ? divisor - remainder : remainder;
  normalized = static_cast<sd::LongType>(positive);
  return true;
}

/// Decode only replay-safe range forms: all range-defining values must be
/// frozen IArgs or TArgs. Tensor-input range remains data-dependent and is not
/// legal for a descriptor-stable replay segment.
static bool readStaticRangeSpec(const NativeSlot& slot,
                                StaticRangeSpec& spec) {
  if (slot.args.numBArgs != 0 || slot.args.numDArgs > 1 ||
      slot.args.numSArgs != 0) {
    return false;
  }
  const bool integerArguments =
      slot.args.numTArgs == 0 && slot.args.numIArgs >= 1 &&
      slot.args.numIArgs <= 3;
  const bool floatingArguments =
      slot.args.numIArgs == 0 && slot.args.numTArgs >= 1 &&
      slot.args.numTArgs <= 3;
  if (integerArguments == floatingArguments) return false;

  spec = StaticRangeSpec{};
  spec.integerArguments = integerArguments;
  if (integerArguments) {
    const int count = slot.args.numIArgs;
    spec.integerStart = count == 1 ? 0 : slot.args.iArgs[0];
    spec.integerLimit = count == 1 ? slot.args.iArgs[0]
                                           : slot.args.iArgs[1];
    spec.integerDelta = count < 3 ? 1 : slot.args.iArgs[2];
    if (spec.integerDelta == 0) return false;

    uint64_t distance = 0;
    uint64_t stepMagnitude = 0;
    if (spec.integerDelta > 0) {
      if (spec.integerLimit <= spec.integerStart) return false;
      distance = static_cast<uint64_t>(spec.integerLimit) -
                 static_cast<uint64_t>(spec.integerStart);
      stepMagnitude = static_cast<uint64_t>(spec.integerDelta);
    } else {
      if (spec.integerLimit >= spec.integerStart) return false;
      distance = static_cast<uint64_t>(spec.integerStart) -
                 static_cast<uint64_t>(spec.integerLimit);
      stepMagnitude = uint64_t{0} -
                      static_cast<uint64_t>(spec.integerDelta);
    }
    // The framework computes (limit - start) in LongType. Reject the cases in
    // which that source expression would overflow rather than inventing a
    // Vulkan-only result.
    if (distance > static_cast<uint64_t>(
                       std::numeric_limits<sd::LongType>::max()) ||
        stepMagnitude == 0) {
      return false;
    }
    const uint64_t quotient = distance / stepMagnitude;
    if (quotient == 0 ||
        quotient > static_cast<uint64_t>(
                       std::numeric_limits<sd::LongType>::max())) {
      return false;
    }
    spec.length = static_cast<sd::LongType>(quotient);
    const sd::LongType candidate =
        spec.integerStart + spec.length * spec.integerDelta;
    if (signedMagnitude(candidate) < signedMagnitude(spec.integerLimit)) {
      if (spec.length == std::numeric_limits<sd::LongType>::max()) {
        return false;
      }
      ++spec.length;
    }
    spec.valueStart = static_cast<double>(spec.integerStart);
    spec.valueDelta = static_cast<double>(spec.integerDelta);
    return spec.length > 0;
  }

  const int count = slot.args.numTArgs;
  const double start = count == 1 ? 0.0 : slot.args.tArgs[0];
  const double limit = count == 1 ? slot.args.tArgs[0]
                                  : slot.args.tArgs[1];
  const double delta = count < 3 ? 1.0 : slot.args.tArgs[2];
  if (!std::isfinite(start) || !std::isfinite(limit) ||
      !std::isfinite(delta) || delta == 0.0) {
    return false;
  }
  const double quotient = (limit - start) / delta;
  if (!std::isfinite(quotient) || quotient < 1.0 ||
      quotient >= static_cast<double>(
                        std::numeric_limits<sd::LongType>::max())) {
    return false;
  }
  spec.length = static_cast<sd::LongType>(quotient);
  const double candidate = start + static_cast<double>(spec.length) * delta;
  if (!std::isfinite(candidate)) return false;
  if (std::fabs(candidate) < std::fabs(limit)) {
    if (spec.length == std::numeric_limits<sd::LongType>::max()) {
      return false;
    }
    ++spec.length;
  }

  // The three-TArg implementation first materializes start and delta as
  // FLOAT32 scalars. Preserve that framework conversion before the kernel
  // promotes floating arithmetic to AccT.
  spec.valueStart = count == 3 ? static_cast<float>(start) : start;
  spec.valueDelta = count == 3 ? static_cast<float>(delta) : delta;
  return spec.length > 0 && std::isfinite(spec.valueStart) &&
         std::isfinite(spec.valueDelta);
}

/// Emit the exact logical shape, element strides, and base offset carried by an
/// NDArray. Descriptor bindings point at the owning DataBuffer, so preserving the
/// MemRef layout is sufficient for C/F order, sliced arrays, and ordinary views.
static std::string mlirMemrefBody(NDArray* array,
                                  const std::string& elementType) {
  if (array == nullptr || array->rankOf() < 0) return "";
  std::ostringstream type;
  for (int d = 0; d < array->rankOf(); ++d) {
    type << array->sizeAt(d) << "x";
  }
  type << elementType << ", strided<[";
  const sd::LongType* strides = shape::stride(array->shapeInfo());
  for (int d = 0; d < array->rankOf(); ++d) {
    if (d != 0) type << ", ";
    type << strides[d];
  }
  type << "], offset: " << array->offset() << ">";
  return type.str();
}

static bool sameShapeAndType(NDArray* lhs, NDArray* rhs) {
  return lhs != nullptr && rhs != nullptr && lhs->dataType() == rhs->dataType() &&
         lhs->isSameShape(rhs);
}

static bool sameExactView(NDArray* lhs, NDArray* rhs) {
  if (!sameShapeAndType(lhs, rhs) ||
      lhs->getDataBuffer() != rhs->getDataBuffer() ||
      lhs->offset() != rhs->offset()) {
    return false;
  }
  const auto* lhsStrides = shape::stride(lhs->shapeInfo());
  const auto* rhsStrides = shape::stride(rhs->shapeInfo());
  for (int dimension = 0; dimension < lhs->rankOf(); ++dimension) {
    if (lhsStrides[dimension] != rhsStrides[dimension]) return false;
  }
  return true;
}

static bool broadcastShapeMatches(NDArray* lhs,
                                  NDArray* rhs,
                                  NDArray* output) {
  if (lhs == nullptr || rhs == nullptr || output == nullptr) return false;
  const int outputRank = std::max(lhs->rankOf(), rhs->rankOf());
  if (output->rankOf() != outputRank) return false;
  for (int od = outputRank - 1, ld = lhs->rankOf() - 1,
           rd = rhs->rankOf() - 1;
       od >= 0; --od, --ld, --rd) {
    const sd::LongType lhsDim = ld >= 0 ? lhs->sizeAt(ld) : 1;
    const sd::LongType rhsDim = rd >= 0 ? rhs->sizeAt(rd) : 1;
    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1) return false;
    if (output->sizeAt(od) != std::max(lhsDim, rhsDim)) return false;
  }
  return true;
}

static bool hasNoBoolDtypeOrStringArgs(const NativeSlot& slot) {
  return slot.args.numBArgs == 0 && slot.args.numDArgs == 0 &&
         slot.args.numSArgs == 0;
}

static bool argumentContractValuesMatch(
    const VulkanKernelEmitterInfo& emitter, const NativeSlot& slot) {
  const uint8_t traits = emitter.argumentContract.valueTraits;
  if ((traits & VULKAN_ARGUMENT_VALUES_FINITE_TARGS) != 0) {
    for (int i = 0; i < slot.args.numTArgs; ++i) {
      if (!std::isfinite(slot.args.tArgs[i])) return false;
    }
  }
  if ((traits & VULKAN_ARGUMENT_VALUES_ORDERED_TARG_PAIR) != 0 &&
      slot.args.numTArgs == 2 &&
      slot.args.tArgs[0] > slot.args.tArgs[1]) {
    return false;
  }
  if ((traits & VULKAN_ARGUMENT_VALUES_POSITIVE_TARG_ZERO) != 0 &&
      slot.args.numTArgs > 0 &&
      (!std::isfinite(slot.args.tArgs[0]) || slot.args.tArgs[0] <= 0.0)) {
    return false;
  }
  if ((traits & VULKAN_ARGUMENT_VALUES_PROBABILITY_TARG_ZERO) != 0 &&
      slot.args.numTArgs > 0 &&
      (!std::isfinite(slot.args.tArgs[0]) || slot.args.tArgs[0] < 0.0 ||
       slot.args.tArgs[0] >= 1.0)) {
    return false;
  }
  if ((traits &
       VULKAN_ARGUMENT_VALUES_NONZERO_TARG_REQUIRES_FALSE_BARG) != 0 &&
      slot.args.numTArgs > 0 && slot.args.tArgs[0] != 0.0 &&
      slot.args.numBArgs > 0 && slot.args.bArgs[0]) {
    return false;
  }
  if ((traits & VULKAN_ARGUMENT_VALUES_PRESENT_BARG_ZERO_TRUE) != 0 &&
      slot.args.numBArgs > 0 && !slot.args.bArgs[0]) {
    return false;
  }
  return true;
}

static bool argumentContractMatchesSlot(
    const VulkanKernelEmitterInfo& emitter, const NativeSlot& slot,
    int numIn, int numOut, NDArray** outputs) {
  auto matches = [&](int numDArgs) {
    return vulkanArgumentContractMatches(
        emitter, numIn, numOut, slot.args.numTArgs, slot.args.numIArgs,
        slot.args.numBArgs, numDArgs, slot.args.numSArgs);
  };
  if (matches(slot.args.numDArgs)) return true;
  return slot.args.numDArgs > 0 &&
         outputDataTypeArgumentsMatch(slot, outputs, numOut) && matches(0);
}

static bool unaryArgumentsMatch(const VulkanKernelEmitterInfo& emitter,
                                const NativeSlot& slot, int numIn,
                                NDArray** outputs, int numOut) {
  if (emitter.argumentContract.alternativeCount != 0) {
    return argumentContractMatchesSlot(
               emitter, slot, numIn, numOut, outputs) &&
           argumentContractValuesMatch(emitter, slot);
  }
  if (slot.args.numIArgs != 0 || slot.args.numBArgs != 0 ||
      slot.args.numSArgs != 0 ||
      !outputDataTypeArgumentsMatch(slot, outputs, numOut)) {
    return false;
  }
  switch (emitter.argumentSchema) {
    case VulkanArgumentSchema::NONE:
      return numIn == 1 && slot.args.numTArgs == 0;
    case VulkanArgumentSchema::OPTIONAL_SCALAR:
      // The catalogue schema is the executable ABI.  Some configurable native
      // descriptors declare one TArg while their implementation deliberately
      // supplies a semantic default when it is absent (for example relu).
      return numIn == 1 && slot.args.numTArgs <= 1;
    case VulkanArgumentSchema::OPTIONAL_FINITE_SCALAR:
      return numIn == 1 && slot.args.numTArgs <= 1 &&
             (slot.args.numTArgs == 0 ||
              std::isfinite(slot.args.tArgs[0]));
    default:
      return false;
  }
}

static bool unaryProducesFloatingOutput(
    const VulkanKernelEmitterInfo& emitter) {
  return hasVulkanEmitterTrait(
      emitter, VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
}

static bool isActivationBackward(
    const VulkanKernelEmitterInfo& emitter) {
  constexpr uint32_t kTraits =
      sd::ops::OP_TRAIT_ACTIVATION | sd::ops::OP_TRAIT_BACKWARD;
  return (emitter.traits & kTraits) == kTraits;
}

static bool reductionProducesFloatingOutput(
    const VulkanKernelEmitterInfo& emitter) {
  return hasVulkanEmitterTrait(
      emitter, VULKAN_EMITTER_TRAIT_FLOAT_RESULT);
}

static bool normalizeAxis(sd::LongType rawAxis, int rank, int64_t& axis) {
  if (rank <= 0) return false;
  sd::LongType normalized = rawAxis < 0 ? rawAxis + rank : rawAxis;
  if (normalized < 0 || normalized >= rank) return false;
  axis = static_cast<int64_t>(normalized);
  return true;
}

static bool reductionOutputMatches(NDArray* input, NDArray* output,
                                   const std::vector<int64_t>& axes,
                                   bool keepDims);

/// Decode reduction execution metadata from the frozen slot arguments.
static bool reductionForSlot(const NativeSlot& slot, NDArray* input,
                             const VulkanKernelEmitterInfo& emitter,
                             NDArray** outputs, int numOut,
                             std::vector<int64_t>& axes, bool& keepDims,
                             bool& biasCorrected) {
  if (input == nullptr) return false;
  const int rank = input->rankOf();
  const bool indexReduction =
      emitter.argumentSchema == VulkanArgumentSchema::INDEX_REDUCTION;
  const bool acceptsBooleanParameters = hasVulkanEmitterTrait(
      emitter, VULKAN_EMITTER_TRAIT_BOOLEAN_PARAMETERS);
  if (rank < 1 || slot.args.numIArgs > rank ||
      slot.args.numSArgs != 0 ||
      !outputDataTypeArgumentsMatch(slot, outputs, numOut)) {
    return false;
  }
  if (indexReduction) {
    // Runtime-axis tensors are data-dependent. The replay-safe contract is one
    // static axis and an explicit supported index dtype override.
    const int maximumBooleanArguments = acceptsBooleanParameters ? 1 : 0;
    if (slot.args.numIArgs != 1 || slot.args.numDArgs != 1 ||
        slot.args.numTArgs != 0 ||
        slot.args.numBArgs > maximumBooleanArguments) {
      return false;
    }
    int64_t axis = -1;
    if (!normalizeAxis(slot.args.iArgs[0], rank, axis)) return false;
    axes.assign(1, axis);
    keepDims = slot.args.numBArgs == 1 && slot.args.bArgs[0];
    biasCorrected = false;
    return true;
  }
  biasCorrected = false;
  switch (emitter.argumentSchema) {
    case VulkanArgumentSchema::STATISTICAL_REDUCTION: {
      const int maximumBooleanArguments = acceptsBooleanParameters ? 2 : 0;
      if (slot.args.numTArgs > 2 ||
          slot.args.numBArgs > maximumBooleanArguments) {
        return false;
      }
      keepDims = slot.args.numBArgs > 0
                     ? slot.args.bArgs[0]
                     : (slot.args.numTArgs > 0 && slot.args.tArgs[0] != 0.0);
      biasCorrected =
          slot.args.numBArgs > 1
              ? slot.args.bArgs[1]
              : (slot.args.numBArgs == 0 && slot.args.numTArgs > 1 &&
                 slot.args.tArgs[1] != 0.0);
      break;
    }
    case VulkanArgumentSchema::REDUCTION_KEEPDIMS: {
      const int maximumBooleanArguments = acceptsBooleanParameters ? 1 : 0;
      if (slot.args.numTArgs > 1 ||
          slot.args.numBArgs > maximumBooleanArguments) {
        return false;
      }
      keepDims = slot.args.numBArgs == 1
                     ? slot.args.bArgs[0]
                     : (slot.args.numTArgs == 1 && slot.args.tArgs[0] != 0.0);
      break;
    }
    default:
      return false;
  }

  std::set<int64_t> uniqueAxes;
  if (slot.args.numIArgs == 0) {
    for (int d = 0; d < rank; ++d) uniqueAxes.insert(d);
  } else if (hasVulkanEmitterTrait(
                 emitter,
                 VULKAN_EMITTER_TRAIT_TAD_REDUCTION_PERMUTATION) &&
             slot.args.numIArgs == rank) {
    if (numOut != 1 || outputs == nullptr || outputs[0] == nullptr) {
      return false;
    }

    // NativeOpExecutioner supplies CUDA's full-rank TAD permutation, while a
    // declarable reduction using the same typed descriptor supplies the reduced
    // axes directly. A full-rank argument list is the TAD ABI; shorter lists
    // continue through the ordinary axis decoder below.
    std::vector<int64_t> permutation;
    permutation.reserve(static_cast<size_t>(rank));
    for (int i = 0; i < rank; ++i) {
      int64_t axis = -1;
      if (!normalizeAxis(slot.args.iArgs[i], rank, axis) ||
          !uniqueAxes.insert(axis).second) {
        return false;
      }
      permutation.push_back(axis);
    }

    // ShapeUtils::evalDimsForReduceOp follows CUDA's TAD ABI: ordinary
    // dimensions precede a non-empty suffix of reduced dimensions. Recover
    // that suffix from the actual output contract instead of reinterpreting
    // the full permutation as an axis list.
    for (int suffixStart = rank - 1; suffixStart >= 0; --suffixStart) {
      std::vector<int64_t> candidate(
          permutation.begin() + suffixStart, permutation.end());
      std::sort(candidate.begin(), candidate.end());
      if (reductionOutputMatches(input, outputs[0], candidate, keepDims)) {
        axes = std::move(candidate);
        return true;
      }
    }
    return false;
  } else {
    for (int i = 0; i < slot.args.numIArgs; ++i) {
      int64_t axis = -1;
      if (!normalizeAxis(slot.args.iArgs[i], rank, axis) ||
          !uniqueAxes.insert(axis).second) {
        return false;
      }
    }
  }
  axes.assign(uniqueAxes.begin(), uniqueAxes.end());
  return !axes.empty();
}

static bool reductionOutputMatches(NDArray* input, NDArray* output,
                                   const std::vector<int64_t>& axes,
                                   bool keepDims) {
  std::set<int64_t> reduced(axes.begin(), axes.end());
  std::vector<sd::LongType> expected;
  for (int d = 0; d < input->rankOf(); ++d) {
    if (reduced.count(d) != 0) {
      if (keepDims) expected.push_back(1);
    } else {
      expected.push_back(input->sizeAt(d));
    }
  }

  if (expected.empty()) {
    return output->rankOf() == 0 ||
           (output->rankOf() == 1 && output->sizeAt(0) == 1);
  }
  if (output->rankOf() != static_cast<int>(expected.size())) return false;
  for (int d = 0; d < output->rankOf(); ++d) {
    if (output->sizeAt(d) != expected[static_cast<size_t>(d)]) return false;
  }
  return true;
}

/// Decode transpose/permute metadata exactly as the existing Triton path does.
/// Static permute requires one iArg per axis; argument-free transpose reverses axes.
template <typename Policy>
static bool permutationForSlot(const NativeSlot& slot, int rank,
                               std::vector<int64_t>& permutation) {
  permutation.clear();
  if constexpr (Policy::transposeDefaultReverse) {
    if (slot.args.numIArgs == 0) {
      for (int d = rank - 1; d >= 0; --d) permutation.push_back(d);
    }
  }
  if (permutation.empty()) {
    if (slot.args.iArgs == nullptr || slot.args.numIArgs != rank) return false;
    for (int d = 0; d < rank; ++d) {
      permutation.push_back(static_cast<int64_t>(slot.args.iArgs[d]));
    }
  }

  std::vector<bool> seen(static_cast<size_t>(rank), false);
  for (int64_t axis : permutation) {
    if (axis < 0 || axis >= rank || seen[static_cast<size_t>(axis)]) return false;
    seen[static_cast<size_t>(axis)] = true;
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  opIsRecordable — static gate
// ─────────────────────────────────────────────────────────────────────────────
//
// Recordability below is the exact contract for the registered handlers.

template <typename Policy>
static bool opIsRecordableTyped(const NativeSlot& slot,
                                NDArray** inputs, int numIn,
                                NDArray** outputs, int numOut,
                                const VulkanDeviceCaps& caps) {
  const bool zeroInputConstant = Policy::constant && numIn == 0;
  if (!hasValidArgumentStorage(slot) || numIn < 0 || numOut <= 0 ||
      (!zeroInputConstant && (numIn == 0 || inputs == nullptr)) ||
      outputs == nullptr) {
    return false;
  }

  // The MemRef ABI emitted below carries every logical dimension, stride, and
  // view offset. Device-only buffers are valid; allocation/synchronization is
  // owned by the recorder's binding lane and is checked there.
  const auto* contractEmitter =
      emitterForSlot(slot);
  auto isStructuralInput = [&](int index) {
    return contractEmitter != nullptr && index >= 0 &&
           vulkanInputIsStructuralIndex(
               *contractEmitter, static_cast<unsigned>(index));
  };
  auto hasBoundStorage = [](NDArray* array) {
    return array != nullptr && array->dataBuffer() != nullptr &&
           array->dataBuffer()->getLenInBytes() > 0 &&
           array->lengthOf() > 0;
  };
  auto hasSupportedStorage = [&](NDArray* array) {
    if (!hasBoundStorage(array)) return false;
    std::string storageType;
    std::string accumulatorType;
    bool isUnsigned = false;
    return selectMlirScalarTypes(array->dataType(), caps, storageType,
                                 accumulatorType, isUnsigned);
  };
  for (int i = 0; i < numIn; ++i) {
    if (isStructuralInput(i)) {
      if (!hasBoundStorage(inputs[i])) return false;
    } else if (!hasSupportedStorage(inputs[i])) {
      return false;
    }
  }
  for (int i = 0; i < numOut; ++i) {
    if (!hasSupportedStorage(outputs[i])) return false;
  }

  // ── Reusable static data-movement contracts ───────────────────────────────
  // Structural tensors stay in the descriptor ABI, but their values are never
  // read by the recorder or by the generated kernel. Canonical output metadata
  // and frozen IArgs completely determine every index equation below.
  if constexpr (Policy::contractMovement) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr) return false;
    const bool hasArgumentContract =
        emitter->argumentContract.alternativeCount != 0;
    if (hasArgumentContract) {
      if (!vulkanArgumentContractMatches(
              *emitter, numIn, numOut, slot.args.numTArgs,
              slot.args.numIArgs, slot.args.numBArgs, slot.args.numDArgs,
              slot.args.numSArgs) ||
          !argumentContractValuesMatch(*emitter, slot)) {
        return false;
      }
    } else if (slot.args.numTArgs != 0 || slot.args.numDArgs != 0 ||
               slot.args.numSArgs != 0) {
      return false;
    }
    auto sameShapeAndType = [](NDArray* source, NDArray* destination) {
      if (source == nullptr || destination == nullptr ||
          source->dataType() != destination->dataType() ||
          source->rankOf() != destination->rankOf()) {
        return false;
      }
      for (int d = 0; d < source->rankOf(); ++d) {
        if (source->sizeAt(d) != destination->sizeAt(d)) return false;
      }
      return true;
    };
    auto samePayloadType = [&](NDArray* array) {
      return array != nullptr && outputs[0] != nullptr &&
             array->dataType() == outputs[0]->dataType();
    };
    auto integerStructural = [](NDArray* array) {
      return array != nullptr && DataTypeUtils::isZ(array->dataType()) &&
             !DataTypeUtils::isB(array->dataType()) && array->rankOf() <= 1;
    };
    auto denseRawPayload = [](NDArray* array) {
      // Stride-based contiguity instead of ews(): ews() is deprecated, can be
      // stale for views, and reports 0 for DSP pre-allocated buffers whose
      // strides are in fact contiguous.
      if (array == nullptr || array->isView() || array->offset() != 0 ||
          !shape::strideDescendingCAscendingF(array->shapeInfo())) {
        return false;
      }
      const char order = array->ordering();
      return array->rankOf() == 0 || order == 'c' || order == 'f';
    };

    if (emitter->loweringContract == VulkanLoweringContract::LINEAR_COPY) {
      if (slot.args.numBArgs != 0) return false;

      if (usesStructuralShapeCopySchedule(*emitter)) {
        // The underlying op is a raw DataBuffer copy. Replay is legal only for
        // standalone dense payloads. Truncating copies are fully writing; an
        // output larger than its source would be partial.
        return slot.args.numIArgs == 0 && numIn == 2 && numOut == 1 &&
               samePayloadType(inputs[0]) && integerStructural(inputs[1]) &&
               denseRawPayload(inputs[0]) && denseRawPayload(outputs[0]) &&
               outputs[0]->lengthOf() <= inputs[0]->lengthOf();
      }

      if (usesReshapeCopySchedule(*emitter)) {
        if (numIn < 1 || numIn > 2 || numOut != 1 ||
            !samePayloadType(inputs[0])) {
          return false;
        }
        if (numIn == 2) {
          // The resolved output MemRef freezes the replay schedule. The shape
          // tensor remains a structural ABI operand and is never loaded by the
          // device kernel.
          if (slot.args.numIArgs != 0 || !integerStructural(inputs[1]) ||
              inputs[1]->lengthOf() != outputs[0]->rankOf() ||
              (outputs[0]->rankOf() > 0 && outputs[0]->ordering() != 'c')) {
            return false;
          }
        } else {
          int argumentOffset = 0;
          char logicalOrder = 'c';
          if (slot.args.numIArgs > 0 &&
              (slot.args.iArgs[0] == -99 || slot.args.iArgs[0] == -102)) {
            logicalOrder = static_cast<char>(-slot.args.iArgs[0]);
            argumentOffset = 1;
          }
          if (slot.args.numIArgs - argumentOffset != outputs[0]->rankOf() ||
              (outputs[0]->rankOf() > 0 &&
               outputs[0]->ordering() != logicalOrder)) {
            return false;
          }
          bool inferred = false;
          for (int d = 0; d < outputs[0]->rankOf(); ++d) {
            const sd::LongType requested =
                slot.args.iArgs[argumentOffset + d];
            if (requested == -1) {
              if (inferred) return false;
              inferred = true;
            } else if (requested == 0) {
              if (d >= inputs[0]->rankOf() ||
                  outputs[0]->sizeAt(d) != inputs[0]->sizeAt(d)) {
                return false;
              }
            } else if (requested <= 0 || outputs[0]->sizeAt(d) != requested) {
              return false;
            }
          }
        }
        return inputs[0]->lengthOf() == outputs[0]->lengthOf() ||
               inputs[0]->lengthOf() == 1;
      }

      if (emitter->argumentSchema == VulkanArgumentSchema::NONE) {
        if (slot.args.numIArgs != 0 ||
            !usesSameShapeCopySchedule(*emitter)) {
          return false;
        }
        if (usesTrailingPayloadCopySchedule(*emitter)) {
          return numIn >= 1 && numOut == 1 &&
                 sameShapeAndType(inputs[numIn - 1], outputs[0]);
        }
        if (numIn < 1 || numIn != numOut) return false;
        for (int i = 0; i < numIn; ++i) {
          if (!sameShapeAndType(inputs[i], outputs[i])) return false;
        }
        return true;
      }

      if (numIn != 1 || numOut != 1 || !samePayloadType(inputs[0])) {
        return false;
      }
      if (emitter->argumentSchema == VulkanArgumentSchema::AXES_IARGS) {
        const int rank = inputs[0]->rankOf();
        std::vector<bool> removed(static_cast<size_t>(rank), false);
        if (slot.args.numIArgs == 0) {
          for (int d = 0; d < rank; ++d) {
            removed[static_cast<size_t>(d)] = inputs[0]->sizeAt(d) == 1;
          }
        } else {
          for (int i = 0; i < slot.args.numIArgs; ++i) {
            int64_t axis = -1;
            if (!normalizeAxis(slot.args.iArgs[i], rank, axis) ||
                removed[static_cast<size_t>(axis)] ||
                inputs[0]->sizeAt(static_cast<int>(axis)) != 1) {
              return false;
            }
            removed[static_cast<size_t>(axis)] = true;
          }
        }
        int expectedRank = 0;
        for (bool value : removed) expectedRank += value ? 0 : 1;
        if (outputs[0]->rankOf() != expectedRank) return false;
        int outputDimension = 0;
        for (int d = 0; d < rank; ++d) {
          if (!removed[static_cast<size_t>(d)] &&
              outputs[0]->sizeAt(outputDimension++) != inputs[0]->sizeAt(d)) {
            return false;
          }
        }
        return expectedRank == 0 ||
               inputs[0]->ordering() == outputs[0]->ordering();
      }
      if (emitter->argumentSchema == VulkanArgumentSchema::SINGLE_IARG) {
        if (slot.args.numIArgs != 1 ||
            outputs[0]->rankOf() != inputs[0]->rankOf() + 1) {
          return false;
        }
        int64_t axis = -1;
        if (!normalizeAxis(slot.args.iArgs[0], inputs[0]->rankOf() + 1,
                           axis)) {
          return false;
        }
        int inputDimension = 0;
        for (int d = 0; d < outputs[0]->rankOf(); ++d) {
          const sd::LongType expected =
              d == axis ? 1 : inputs[0]->sizeAt(inputDimension++);
          if (outputs[0]->sizeAt(d) != expected) return false;
        }
        return inputs[0]->ordering() == outputs[0]->ordering();
      }
      return false;
    }

    if (usesLinearConcatSchedule(*emitter)) {
        if (numIn < 1 || numOut != 1 || slot.args.numIArgs != 1 ||
            slot.args.numBArgs != 0 || outputs[0]->rankOf() != 1) {
          return false;
        }
        const char order = static_cast<char>(slot.args.iArgs[0]);
        if (order != 'c' && order != 'f') return false;
        uint64_t length = 0;
        for (int i = 0; i < numIn; ++i) {
          if (!samePayloadType(inputs[i]) ||
              static_cast<uint64_t>(inputs[i]->lengthOf()) >
                  std::numeric_limits<uint64_t>::max() - length) {
            return false;
          }
          length += static_cast<uint64_t>(inputs[i]->lengthOf());
        }
        return length == static_cast<uint64_t>(outputs[0]->lengthOf());
      }

    if (usesOutputShapeBroadcastSchedule(*emitter)) {
        if (numIn != 2 || numOut != 1 || slot.args.numIArgs != 0 ||
            slot.args.numBArgs != 0 || !samePayloadType(inputs[0]) ||
            !integerStructural(inputs[1]) ||
            inputs[1]->lengthOf() != outputs[0]->rankOf() ||
            inputs[0]->rankOf() > outputs[0]->rankOf()) {
          return false;
        }
        for (int inputDimension = inputs[0]->rankOf() - 1,
                 outputDimension = outputs[0]->rankOf() - 1;
             inputDimension >= 0; --inputDimension, --outputDimension) {
          const sd::LongType inputSize = inputs[0]->sizeAt(inputDimension);
          const sd::LongType outputSize = outputs[0]->sizeAt(outputDimension);
          if (inputSize != 1 && inputSize != outputSize) return false;
        }
        return true;
      }

    if (usesAxisPartitionSchedule(*emitter)) {
        if (numIn != 2 || numOut < 1 || slot.args.numIArgs != 1 ||
            slot.args.numBArgs != 0 || !integerStructural(inputs[1]) ||
            inputs[1]->lengthOf() != numOut) {
          return false;
        }
        int64_t axis = -1;
        if (!normalizeAxis(slot.args.iArgs[0], inputs[0]->rankOf(), axis)) {
          return false;
        }
        uint64_t partitioned = 0;
        for (int i = 0; i < numOut; ++i) {
          if (!samePayloadType(inputs[0]) || !samePayloadType(outputs[i]) ||
              outputs[i]->rankOf() != inputs[0]->rankOf()) {
            return false;
          }
          for (int d = 0; d < inputs[0]->rankOf(); ++d) {
            if (d != axis &&
                outputs[i]->sizeAt(d) != inputs[0]->sizeAt(d)) {
              return false;
            }
          }
          const sd::LongType size = outputs[i]->sizeAt(static_cast<int>(axis));
          if (size <= 0 ||
              static_cast<uint64_t>(size) >
                  std::numeric_limits<uint64_t>::max() - partitioned) {
            return false;
          }
          partitioned += static_cast<uint64_t>(size);
        }
        return partitioned ==
               static_cast<uint64_t>(inputs[0]->sizeAt(static_cast<int>(axis)));
      }

    return false;
  }

  // ── Batched rank-two matrix-list schedule ─────────────────────────────────
  if constexpr (Policy::batchedMatrixList) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr ||
        !usesBatchedMatrixListSchedule(*emitter) ||
        numIn < 4 || (numIn - 2) % 2 != 0 ||
        numOut != (numIn - 2) / 2 ||
        slot.args.numIArgs != 2 || slot.args.numTArgs != 0 ||
        !hasNoBoolDtypeOrStringArgs(slot)) {
      return false;
    }
    const bool transposeA = slot.args.iArgs[0] != 0;
    const bool transposeB = slot.args.iArgs[1] != 0;
    if ((slot.args.iArgs[0] != 0 && slot.args.iArgs[0] != 1) ||
        (slot.args.iArgs[1] != 0 && slot.args.iArgs[1] != 1) ||
        inputs[0]->rankOf() > 1 || inputs[1]->rankOf() > 1) {
      return false;
    }
    const int batch = numOut;
    if ((inputs[0]->lengthOf() != 1 && inputs[0]->lengthOf() != batch) ||
        (inputs[1]->lengthOf() != 1 && inputs[1]->lengthOf() != batch)) {
      return false;
    }
    NDArray* firstA = inputs[2];
    NDArray* firstB = inputs[2 + batch];
    if (firstA == nullptr || firstB == nullptr ||
        firstA->rankOf() != 2 || firstB->rankOf() != 2 ||
        !DataTypeUtils::isR(firstA->dataType()) ||
        firstB->dataType() != firstA->dataType()) {
      return false;
    }
    const sd::LongType m =
        transposeA ? firstA->sizeAt(1) : firstA->sizeAt(0);
    const sd::LongType k =
        transposeA ? firstA->sizeAt(0) : firstA->sizeAt(1);
    const sd::LongType kb =
        transposeB ? firstB->sizeAt(1) : firstB->sizeAt(0);
    const sd::LongType n =
        transposeB ? firstB->sizeAt(0) : firstB->sizeAt(1);
    if (m <= 0 || n <= 0 || k <= 0 || kb != k) return false;
    for (int b = 0; b < batch; ++b) {
      NDArray* a = inputs[2 + b];
      NDArray* matrixB = inputs[2 + batch + b];
      NDArray* output = outputs[b];
      if (a == nullptr || matrixB == nullptr || output == nullptr ||
          a->rankOf() != 2 || matrixB->rankOf() != 2 ||
          output->rankOf() != 2 ||
          a->dataType() != firstA->dataType() ||
          matrixB->dataType() != firstA->dataType() ||
          output->dataType() != firstA->dataType() ||
          (transposeA ? a->sizeAt(1) : a->sizeAt(0)) != m ||
          (transposeA ? a->sizeAt(0) : a->sizeAt(1)) != k ||
          (transposeB ? matrixB->sizeAt(1) : matrixB->sizeAt(0)) != k ||
          (transposeB ? matrixB->sizeAt(0) : matrixB->sizeAt(1)) != n ||
          output->sizeAt(0) != m || output->sizeAt(1) != n) {
        return false;
      }
    }
    return true;
  }

  // ── Serial indexed accumulation schedule ─────────────────────────────────
  if constexpr (Policy::indexedAccumulation) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr ||
        !usesIndexedAccumulationSchedule(*emitter) ||
        numIn != 3 || numOut != 1 || slot.args.numIArgs != 0 ||
        slot.args.numTArgs != 0 || slot.args.numDArgs != 0 ||
        slot.args.numSArgs != 0 || slot.args.numBArgs > 2 ||
        (slot.args.numBArgs > 1 && slot.args.bArgs[1])) {
      return false;
    }
    NDArray* indices = inputs[0];
    NDArray* updates = inputs[1];
    NDArray* shape = inputs[2];
    NDArray* output = outputs[0];
    if (indices == nullptr || updates == nullptr || shape == nullptr ||
        output == nullptr || indices->rankOf() < 1 || shape->rankOf() != 1 ||
        !DataTypeUtils::isZ(indices->dataType()) ||
        DataTypeUtils::isB(indices->dataType()) ||
        updates->dataType() != output->dataType() ||
        shape->lengthOf() != output->rankOf()) {
      return false;
    }
    const int indexDepth =
        static_cast<int>(indices->sizeAt(indices->rankOf() - 1));
    const int prefixRank = indices->rankOf() - 1;
    const int sliceRank = output->rankOf() - indexDepth;
    if (indexDepth <= 0 || indexDepth > output->rankOf() ||
        updates->rankOf() != prefixRank + sliceRank) {
      return false;
    }
    for (int d = 0; d < prefixRank; ++d) {
      if (updates->sizeAt(d) != indices->sizeAt(d)) return false;
    }
    for (int d = 0; d < sliceRank; ++d) {
      if (updates->sizeAt(prefixRank + d) !=
          output->sizeAt(indexDepth + d)) {
        return false;
      }
    }
    return true;
  }

  // ── Descriptor-driven indexed TAD movement ────────────────────────────────
  if constexpr (Policy::indexedTadMovement) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || !usesIndexedTadMovementSchedule(*emitter) ||
        !vulkanArgumentContractMatches(
            *emitter, numIn, numOut, slot.args.numTArgs,
            slot.args.numIArgs, slot.args.numBArgs, slot.args.numDArgs,
            slot.args.numSArgs) ||
        !hasNoBoolDtypeOrStringArgs(slot) || slot.args.numTArgs != 0) {
      return false;
    }

    auto selectStorage = [&](NDArray* array, std::string& storage,
                             bool& isUnsigned) {
      std::string accumulator;
      return array != nullptr &&
             selectMlirScalarTypes(array->dataType(), caps, storage,
                                   accumulator, isUnsigned);
    };
    auto validateDimensions = [](NDArray* array,
                                 const std::vector<sd::LongType>& dimensions,
                                 sd::LongType& itemCount) {
      if (array == nullptr || dimensions.empty()) return false;
      std::vector<bool> seen(static_cast<size_t>(array->rankOf()), false);
      sd::LongType tadLength = 1;
      for (const auto dimension : dimensions) {
        if (dimension < 0 || dimension >= array->rankOf() ||
            seen[static_cast<size_t>(dimension)]) {
          return false;
        }
        seen[static_cast<size_t>(dimension)] = true;
        if (array->rankOf() > 1) {
          const auto dimensionSize =
              array->sizeAt(static_cast<int>(dimension));
          if (dimensionSize <= 0 ||
              tadLength > std::numeric_limits<sd::LongType>::max() /
                              dimensionSize) {
            return false;
          }
          tadLength *= dimensionSize;
        }
      }
      if (array->rankOf() == 1) {
        itemCount = array->lengthOf();
        return true;
      }
      if (tadLength <= 0 || array->lengthOf() % tadLength != 0) return false;
      itemCount = array->lengthOf() / tadLength;
      return true;
    };

    switch (emitter->recipe) {
      case VulkanKernelRecipe::PULL_INDEXED_TADS: {
        if (numIn != 2 || numOut != 1 || slot.args.numIArgs != 2) {
          return false;
        }
        NDArray* source = inputs[0];
        NDArray* indexes = inputs[1];
        NDArray* destination = outputs[0];
        const auto count = slot.args.iArgs[0];
        const auto dimension = slot.args.iArgs[1];
        std::string sourceStorage;
        std::string indexStorage;
        std::string destinationStorage;
        bool sourceUnsigned = false;
        bool indexUnsigned = false;
        bool destinationUnsigned = false;
        if (source == nullptr || indexes == nullptr || destination == nullptr ||
            source->rankOf() < 1 || source->rankOf() > 2 ||
            destination->rankOf() != source->rankOf() ||
            source->dataType() != destination->dataType() ||
            count <= 0 || indexes->lengthOf() < count ||
            dimension < 0 || dimension >= source->rankOf() ||
            !selectStorage(source, sourceStorage, sourceUnsigned) ||
            !selectStorage(indexes, indexStorage, indexUnsigned) ||
            !selectStorage(destination, destinationStorage,
                           destinationUnsigned) ||
            sourceStorage != destinationStorage ||
            sourceUnsigned != destinationUnsigned ||
            indexStorage != "i64" || indexUnsigned) {
          return false;
        }
        if (source->rankOf() == 1) {
          return dimension == 0 && destination->lengthOf() == count;
        }
        if (dimension == 1) {
          return destination->sizeAt(0) == count &&
                 destination->sizeAt(1) == source->sizeAt(1);
        }
        return dimension == 0 &&
               destination->sizeAt(0) == source->sizeAt(0) &&
               destination->sizeAt(1) == count;
      }
      case VulkanKernelRecipe::DISJOINT_PAIR_SHUFFLE: {
        if (slot.args.numIArgs < 3) return false;
        const auto encodedArrayCount = slot.args.iArgs[0];
        if (encodedArrayCount <= 0 ||
            encodedArrayCount > std::numeric_limits<int>::max()) {
          return false;
        }
        const int arrayCount = static_cast<int>(encodedArrayCount);
        if (numIn != arrayCount + 1 || numOut != arrayCount) return false;

        std::vector<sd::LongType> sharedDimensions;
        int argumentOffset = 1;
        sd::LongType commonItemCount = -1;
        for (int arrayIndex = 0; arrayIndex < arrayCount; ++arrayIndex) {
          if (argumentOffset >= slot.args.numIArgs) return false;
          const auto encodedDimensionCount =
              slot.args.iArgs[argumentOffset++];
          if (encodedDimensionCount <= 0 ||
              encodedDimensionCount > std::numeric_limits<int>::max() ||
              encodedDimensionCount >
                  slot.args.numIArgs - argumentOffset) {
            return false;
          }
          const int dimensionCount =
              static_cast<int>(encodedDimensionCount);
          std::vector<sd::LongType> dimensions;
          dimensions.reserve(static_cast<size_t>(dimensionCount));
          for (int index = 0; index < dimensionCount; ++index) {
            dimensions.push_back(slot.args.iArgs[argumentOffset++]);
          }
          if (arrayIndex == 0) {
            sharedDimensions = dimensions;
          } else if (dimensions != sharedDimensions) {
            return false;
          }

          NDArray* source = inputs[arrayIndex];
          NDArray* destination = outputs[arrayIndex];
          std::string sourceStorage;
          std::string destinationStorage;
          bool sourceUnsigned = false;
          bool destinationUnsigned = false;
          sd::LongType itemCount = 0;
          if (!sameExactView(source, destination) ||
              !selectStorage(source, sourceStorage, sourceUnsigned) ||
              !selectStorage(destination, destinationStorage,
                             destinationUnsigned) ||
              sourceStorage != destinationStorage ||
              sourceUnsigned != destinationUnsigned ||
              !validateDimensions(source, sharedDimensions, itemCount)) {
            return false;
          }
          if (commonItemCount < 0) {
            commonItemCount = itemCount;
          } else if (commonItemCount != itemCount) {
            return false;
          }
        }
        if (argumentOffset != slot.args.numIArgs) return false;

        NDArray* shuffleMap = inputs[arrayCount];
        std::string mapStorage;
        bool mapUnsigned = false;
        return selectStorage(shuffleMap, mapStorage, mapUnsigned) &&
               mapStorage == "i32" && !mapUnsigned &&
               shuffleMap->lengthOf() >= commonItemCount;
      }
      default:
        return false;
    }
  }

  // ── matmul / mmul: exact untransposed A*B subset ──────────────────────────
  if constexpr (Policy::matmul) {
    if (numIn != 2 || numOut != 1 || !hasNoBoolDtypeOrStringArgs(slot)) return false;
    if (slot.args.numIArgs > 3 || slot.args.numTArgs > 2) return false;
    for (int i = 0; i < slot.args.numIArgs; ++i) {
      if (slot.args.iArgs[i] != 0) return false;
    }
    const double alpha = slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 1.0;
    const double beta = slot.args.numTArgs > 1 ? slot.args.tArgs[1] : 0.0;
    if (alpha != 1.0 || beta != 0.0) return false;

    const int rank = inputs[0]->rankOf();
    if ((rank != 2 && rank != 3) || inputs[1]->rankOf() != rank ||
        outputs[0]->rankOf() != rank) {
      return false;
    }
    if (inputs[1]->dataType() != inputs[0]->dataType() ||
        outputs[0]->dataType() != inputs[0]->dataType()) {
      return false;
    }
    const int matrixAxis = rank - 2;
    if ((rank == 3 &&
         (inputs[0]->sizeAt(0) != inputs[1]->sizeAt(0) ||
          outputs[0]->sizeAt(0) != inputs[0]->sizeAt(0))) ||
        inputs[0]->sizeAt(matrixAxis + 1) !=
            inputs[1]->sizeAt(matrixAxis) ||
        outputs[0]->sizeAt(matrixAxis) !=
            inputs[0]->sizeAt(matrixAxis) ||
        outputs[0]->sizeAt(matrixAxis + 1) !=
            inputs[1]->sizeAt(matrixAxis + 1)) {
      return false;
    }
    return true;
  }

  // ── rms_norm: exact rank-2 subset ─────────────────────────────────────────
  if constexpr (Policy::rmsNorm) {
    if ((numIn != 1 && numIn != 2) || numOut != 1 ||
        slot.args.numIArgs != 0 || slot.args.numTArgs > 1 ||
        !hasNoBoolDtypeOrStringArgs(slot)) {
      return false;
    }
    if (inputs[0]->rankOf() != 2 ||
        !sameShapeAndType(inputs[0], outputs[0])) {
      return false;
    }
    if (numIn == 2 &&
        (inputs[1]->rankOf() != 1 ||
         inputs[1]->sizeAt(0) != inputs[0]->sizeAt(1))) {
      return false;
    }
    const double epsilon =
        slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 1.0e-5;
    return std::isfinite(epsilon) && epsilon > 0.0;
  }

  // skip_rms_norm has a different operand/argument contract and is handled by
  // its dedicated fused policy below.

  // ── fused_rope: cached, full-head, adjacent-pair subset ───────────────────
  if constexpr (Policy::rope) {
    if (numIn != 3 || numOut != 1 || slot.args.numIArgs > 3 ||
        slot.args.numTArgs > 2 || !hasNoBoolDtypeOrStringArgs(slot)) {
      return false;
    }
    const sd::LongType ropeType =
        slot.args.numIArgs > 0 ? slot.args.iArgs[0] : 0;
    const sd::LongType positionOffset =
        slot.args.numIArgs > 1 ? slot.args.iArgs[1] : 0;
    const sd::LongType rotaryDims =
        slot.args.numIArgs > 2 ? slot.args.iArgs[2] : 0;
    const double frequencyBase =
        slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 10000.0;
    const double frequencyScale =
        slot.args.numTArgs > 1 ? slot.args.tArgs[1] : 1.0;

    if (ropeType != 1 || positionOffset != 0 ||
        frequencyBase != 10000.0 || frequencyScale != 1.0 ||
        inputs[0]->rankOf() != 4 || outputs[0]->rankOf() != 4 ||
        !sameShapeAndType(inputs[0], outputs[0])) {
      return false;
    }
    const sd::LongType sequence = inputs[0]->sizeAt(1);
    const sd::LongType headSize = inputs[0]->sizeAt(3);
    if (headSize <= 0 || headSize % 2 != 0 ||
        (rotaryDims != 0 && rotaryDims != headSize)) {
      return false;
    }
    for (int i = 1; i <= 2; ++i) {
      if (inputs[i]->dataType() != inputs[0]->dataType() ||
          inputs[i]->rankOf() != 2 ||
          inputs[i]->sizeAt(0) != sequence ||
          inputs[i]->sizeAt(1) != headSize / 2) {
        return false;
      }
    }
    return true;
  }

  // ── Structured compute kernels ────────────────────────────────────────────────────
  // These share a trait-selected schedule; the recipe only chooses emitted
  // arithmetic/indexing. No normalized/GEMM intermediate is materialized on
  // the host or in an extra buffer, and each lowering is one compute pipeline.
  if constexpr (Policy::structuredCompute) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr) return false;
    const bool hasEpsilonParameter = hasVulkanEmitterTrait(
        *emitter, VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER);
    const bool indexedBinary = usesBroadcastBinarySchedule(*emitter);
    if (emitter->argumentContract.alternativeCount == 0 ||
        !vulkanArgumentContractMatches(
            *emitter, numIn, numOut, slot.args.numTArgs,
            slot.args.numIArgs, slot.args.numBArgs, slot.args.numDArgs,
            slot.args.numSArgs) ||
        !argumentContractValuesMatch(*emitter, slot)) {
      return false;
    }

    const double epsilon =
        slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 1.0e-5;
    if (hasEpsilonParameter &&
        (!std::isfinite(epsilon) || epsilon <= 0.0)) {
      return false;
    }

    for (int i = 0; i < numOut; ++i) {
      if (outputs[i] == nullptr ||
          (!indexedBinary &&
           outputs[i]->dataType() != inputs[0]->dataType())) {
        return false;
      }
    }

    switch (emitter->recipe) {
      case VulkanKernelRecipe::WINDOW_PARTITION: {
        if (numIn != 1 || numOut != 1 || inputs[0]->rankOf() != 4 ||
            outputs[0]->rankOf() != 4 ||
            !DataTypeUtils::isR(inputs[0]->dataType()) ||
            outputs[0]->dataType() != inputs[0]->dataType()) {
          return false;
        }
        const sd::LongType window = slot.args.iArgs[0];
        const sd::LongType batch = inputs[0]->sizeAt(0);
        const sd::LongType height = inputs[0]->sizeAt(1);
        const sd::LongType width = inputs[0]->sizeAt(2);
        const sd::LongType channels = inputs[0]->sizeAt(3);
        if (window <= 0 || batch <= 0 || height <= 0 || width <= 0 ||
            channels <= 0 || height % window != 0 || width % window != 0) {
          return false;
        }
        const uint64_t windows =
            static_cast<uint64_t>(batch) *
            static_cast<uint64_t>(height / window) *
            static_cast<uint64_t>(width / window);
        return windows <=
                   static_cast<uint64_t>(std::numeric_limits<sd::LongType>::max()) &&
               outputs[0]->sizeAt(0) == static_cast<sd::LongType>(windows) &&
               outputs[0]->sizeAt(1) == window &&
               outputs[0]->sizeAt(2) == window &&
               outputs[0]->sizeAt(3) == channels;
      }
      case VulkanKernelRecipe::WINDOW_UNPARTITION: {
        if (numIn != 1 || numOut != 1 || inputs[0]->rankOf() != 4 ||
            outputs[0]->rankOf() != 4 ||
            !DataTypeUtils::isR(inputs[0]->dataType()) ||
            outputs[0]->dataType() != inputs[0]->dataType()) {
          return false;
        }
        const sd::LongType window = slot.args.iArgs[0];
        const sd::LongType height = slot.args.iArgs[1];
        const sd::LongType width = slot.args.iArgs[2];
        if (window <= 0 || height <= 0 || width <= 0 ||
            height % window != 0 || width % window != 0 ||
            inputs[0]->sizeAt(1) != window ||
            inputs[0]->sizeAt(2) != window || inputs[0]->sizeAt(3) <= 0) {
          return false;
        }
        const sd::LongType windowsPerBatch =
            (height / window) * (width / window);
        const sd::LongType numWindows = inputs[0]->sizeAt(0);
        if (windowsPerBatch <= 0 || numWindows <= 0 ||
            numWindows % windowsPerBatch != 0) {
          return false;
        }
        const sd::LongType batch = numWindows / windowsPerBatch;
        return outputs[0]->sizeAt(0) == batch &&
               outputs[0]->sizeAt(1) == height &&
               outputs[0]->sizeAt(2) == width &&
               outputs[0]->sizeAt(3) == inputs[0]->sizeAt(3);
      }
      case VulkanKernelRecipe::BIAS_ADD: {
        if (numIn != 2 || numOut != 1 || inputs[0]->rankOf() < 1 ||
            inputs[1]->rankOf() != 1 ||
            !DataTypeUtils::isR(inputs[1]->dataType()) ||
            !DataTypeUtils::isR(outputs[0]->dataType()) ||
            outputs[0]->dataType() != inputs[1]->dataType() ||
            !inputs[0]->isSameShape(outputs[0])) {
          return false;
        }
        const bool inputSupported =
            DataTypeUtils::isR(inputs[0]->dataType()) ||
            inputs[0]->dataType() == DataType::INT32 ||
            inputs[0]->dataType() == DataType::UINT32;
        if (!inputSupported) return false;
        const bool nchw =
            slot.args.numBArgs == 1 && slot.args.bArgs[0];
        if (nchw && inputs[0]->rankOf() < 2) return false;
        const int channelAxis = nchw ? 1 : inputs[0]->rankOf() - 1;
        return inputs[1]->sizeAt(0) == inputs[0]->sizeAt(channelAxis);
      }
      case VulkanKernelRecipe::BATCH_NORM: {
        if (numOut != 1 || inputs[0]->rankOf() < 1 ||
            !sameShapeAndType(inputs[0], outputs[0]) ||
            slot.args.numIArgs > inputs[0]->rankOf() + 2) {
          return false;
        }
        const bool applyScale = slot.args.iArgs[0] != 0;
        const bool applyOffset = slot.args.iArgs[1] != 0;
        if (numIn != 3 + static_cast<int>(applyScale) +
                         static_cast<int>(applyOffset)) {
          return false;
        }
        for (int i = 1; i < numIn; ++i) {
          if (inputs[i]->dataType() != inputs[0]->dataType()) return false;
        }
        std::vector<int64_t> axes;
        std::set<int64_t> uniqueAxes;
        if (slot.args.numIArgs == 2) {
          axes.push_back(inputs[0]->rankOf() - 1);
        } else {
          for (int i = 2; i < slot.args.numIArgs; ++i) {
            int64_t axis = -1;
            if (!normalizeAxis(
                    slot.args.iArgs[i], inputs[0]->rankOf(), axis) ||
                !uniqueAxes.insert(axis).second) {
              return false;
            }
            axes.push_back(axis);
          }
        }
        std::vector<sd::LongType> expected;
        if (axes.size() == 1) {
          expected.push_back(inputs[0]->sizeAt(axes[0]));
        } else {
          expected.assign(static_cast<size_t>(inputs[0]->rankOf()), 1);
          for (int64_t axis : axes) {
            expected[static_cast<size_t>(axis)] = inputs[0]->sizeAt(axis);
          }
        }
        auto shapeMatches = [&](NDArray* array) {
          if (array == nullptr ||
              array->rankOf() != static_cast<int>(expected.size())) {
            return false;
          }
          for (int d = 0; d < array->rankOf(); ++d) {
            if (array->sizeAt(d) != expected[static_cast<size_t>(d)]) {
              return false;
            }
          }
          return true;
        };
        for (int i = 1; i < numIn; ++i) {
          if (!shapeMatches(inputs[i])) return false;
        }
        return true;
      }
      case VulkanKernelRecipe::RMS_NORM_BP: {
        // The optional gamma form also produces dGamma, a cross-row
        // reduction. Baseline Vulkan has no portable floating atomic-add, so
        // record only the exact gamma-free form until that device capability
        // is exposed by the backend.
        if (numIn != 2 || numOut != 1 || inputs[1] == nullptr ||
            !DataTypeUtils::isR(inputs[0]->dataType()) ||
            !DataTypeUtils::isR(inputs[1]->dataType()) ||
            !inputs[0]->isSameShape(inputs[1]) ||
            !sameShapeAndType(inputs[0], outputs[0]) ||
            inputs[0]->rankOf() < 1 ||
            inputs[0]->sizeAt(inputs[0]->rankOf() - 1) <= 0) {
          return false;
        }
        return true;
      }
      case VulkanKernelRecipe::FUSED_LAYER_NORM_BP: {
        // The general rank-N form reduces dGamma across all leading rows and
        // needs floating atomic-add (or an intermediate reduction buffer).
        // Neither is a baseline Vulkan feature represented by our device
        // capabilities.  Rank one is a single row, so the emitted invocation
        // owns both outputs and is race-free.
        if (numIn != 3 || numOut != 2 || inputs[1] == nullptr ||
            inputs[2] == nullptr || outputs[1] == nullptr ||
            inputs[0]->rankOf() != 1 || inputs[1]->rankOf() != 1 ||
            inputs[2]->rankOf() != 1 || outputs[0]->rankOf() != 1 ||
            outputs[1]->rankOf() != 1 ||
            !DataTypeUtils::isR(inputs[0]->dataType()) ||
            !DataTypeUtils::isR(inputs[1]->dataType()) ||
            !DataTypeUtils::isR(inputs[2]->dataType()) ||
            !inputs[0]->isSameShape(inputs[1]) ||
            !inputs[0]->isSameShape(inputs[2]) ||
            !sameShapeAndType(inputs[0], outputs[0]) ||
            !inputs[1]->isSameShape(outputs[1]) ||
            inputs[0]->lengthOf() <= 0) {
          return false;
        }
        return true;
      }
      case VulkanKernelRecipe::PRELU: {
        if (numIn != 2 || numOut != 1 || inputs[0]->rankOf() <= 1 ||
            !DataTypeUtils::isR(inputs[1]->dataType()) ||
            !DataTypeUtils::isR(outputs[0]->dataType()) ||
            !inputs[0]->isSameShape(outputs[0]) ||
            slot.args.numIArgs > inputs[0]->rankOf() - 1) {
          return false;
        }
        std::set<int64_t> sharedAxes;
        for (int i = 0; i < slot.args.numIArgs; ++i) {
          sd::LongType axis = slot.args.iArgs[i];
          if (axis <= 0) axis += inputs[0]->rankOf() - 1;
          if (axis < 1 || axis >= inputs[0]->rankOf()) return false;
          sharedAxes.insert(static_cast<int64_t>(axis));
        }
        uint64_t expectedAlphaLength = 1;
        for (int d = 1; d < inputs[0]->rankOf(); ++d) {
          if (sharedAxes.count(d) != 0) continue;
          const sd::LongType dimension = inputs[0]->sizeAt(d);
          if (dimension <= 0 ||
              expectedAlphaLength >
                  std::numeric_limits<uint64_t>::max() /
                      static_cast<uint64_t>(dimension)) {
            return false;
          }
          expectedAlphaLength *= static_cast<uint64_t>(dimension);
        }
        return expectedAlphaLength ==
               static_cast<uint64_t>(inputs[1]->lengthOf());
      }
      case VulkanKernelRecipe::VISION_EMBEDDING_MERGE: {
        if (numIn != 3 || numOut != 1 || inputs[0]->rankOf() != 3 ||
            inputs[1]->rankOf() != 3 || inputs[2]->rankOf() != 2 ||
            !DataTypeUtils::isR(inputs[0]->dataType()) ||
            !DataTypeUtils::isR(inputs[1]->dataType()) ||
            (inputs[2]->dataType() != DataType::INT32 &&
             inputs[2]->dataType() != DataType::UINT32) ||
            !sameShapeAndType(inputs[0], outputs[0])) {
          return false;
        }
        const sd::LongType batch = inputs[0]->sizeAt(0);
        const sd::LongType sequence = inputs[0]->sizeAt(1);
        const sd::LongType hidden = inputs[0]->sizeAt(2);
        const sd::LongType visionTokens = inputs[1]->sizeAt(1);
        return batch > 0 && sequence > 0 && hidden > 0 && visionTokens >= 0 &&
               inputs[1]->sizeAt(0) == batch &&
               inputs[1]->sizeAt(2) == hidden &&
               inputs[2]->sizeAt(0) == batch &&
               inputs[2]->sizeAt(1) == sequence;
      }
      case VulkanKernelRecipe::APPLY_ALIBI: {
        if (numIn != 1 || numOut != 1 || inputs[0]->rankOf() != 4 ||
            !sameShapeAndType(inputs[0], outputs[0])) {
          return false;
        }
        const sd::LongType batch = inputs[0]->sizeAt(0);
        const sd::LongType heads = inputs[0]->sizeAt(1);
        const sd::LongType sequence = inputs[0]->sizeAt(2);
        const sd::LongType keySequence = inputs[0]->sizeAt(3);
        if (batch <= 0 || heads <= 0 || sequence <= 0 || keySequence <= 0) {
          return false;
        }
        return slot.args.numIArgs == 0 || slot.args.iArgs[0] == heads;
      }
      case VulkanKernelRecipe::ROPE:
      case VulkanKernelRecipe::ROPE_BP: {
        const bool backward =
            hasVulkanOpTrait(*emitter, sd::ops::OP_TRAIT_BACKWARD);
        const int payloadIndex = backward ? 1 : 0;
        NDArray* payload = inputs[payloadIndex];
        if (payload == nullptr ||
            (payload->rankOf() != 3 && payload->rankOf() != 4) ||
            (backward && !sameShapeAndType(inputs[0], payload)) ||
            !sameShapeAndType(payload, outputs[0])) {
          return false;
        }
        const sd::LongType batch = payload->sizeAt(0);
        const sd::LongType sequence = payload->sizeAt(1);
        const sd::LongType headDimension =
            payload->sizeAt(payload->rankOf() - 1);
        const sd::LongType requestedRotaryDimensions =
            slot.args.numIArgs > 2 ? slot.args.iArgs[2] : 0;
        const sd::LongType rotaryDimensions =
            requestedRotaryDimensions > 0 &&
                    requestedRotaryDimensions < headDimension
                ? requestedRotaryDimensions
                : headDimension;
        if (batch <= 0 || sequence <= 0 || headDimension <= 0 ||
            rotaryDimensions <= 0 ||
            (rotaryDimensions > 1 && rotaryDimensions % 2 != 0)) {
          return false;
        }
        return payload->rankOf() != 4 || payload->sizeAt(2) > 0;
      }
      case VulkanKernelRecipe::FLASH_ATTENTION:
      case VulkanKernelRecipe::GROUPED_QUERY_ATTENTION: {
        if (numIn != 3 || numOut != 1 ||
            (inputs[0]->rankOf() != 3 && inputs[0]->rankOf() != 4) ||
            inputs[1]->rankOf() != inputs[0]->rankOf() ||
            inputs[2]->rankOf() != inputs[0]->rankOf() ||
            !sameShapeAndType(inputs[0], outputs[0])) {
          return false;
        }
        const int rank = inputs[0]->rankOf();
        const sd::LongType batch = inputs[0]->sizeAt(0);
        const sd::LongType querySteps = inputs[0]->sizeAt(1);
        const sd::LongType keySteps = inputs[1]->sizeAt(1);
        const sd::LongType headDimension = inputs[0]->sizeAt(rank - 1);
        const sd::LongType queryHeads = rank == 4 ? inputs[0]->sizeAt(2) : 1;
        const sd::LongType keyValueHeads =
            rank == 4 ? inputs[1]->sizeAt(2) : 1;
        if (batch <= 0 || querySteps <= 0 || keySteps <= 0 ||
            headDimension <= 0 || queryHeads <= 0 || keyValueHeads <= 0 ||
            queryHeads % keyValueHeads != 0 ||
            inputs[1]->sizeAt(0) != batch ||
            inputs[2]->sizeAt(0) != batch ||
            inputs[2]->sizeAt(1) != keySteps ||
            inputs[1]->sizeAt(rank - 1) != headDimension ||
            inputs[2]->sizeAt(rank - 1) != headDimension ||
            (rank == 4 &&
             (inputs[2]->sizeAt(2) != keyValueHeads))) {
          return false;
        }
        if (slot.args.numIArgs > 0 &&
            slot.args.iArgs[0] != queryHeads) {
          return false;
        }
        return slot.args.numIArgs <= 1 ||
               slot.args.iArgs[1] == keyValueHeads;
      }
      case VulkanKernelRecipe::FUSED_MROPE: {
        if (numIn != 4 || numOut != 1 ||
            !DataTypeUtils::isR(inputs[0]->dataType()) ||
            inputs[0]->rankOf() != 4 ||
            !sameShapeAndType(inputs[0], outputs[0])) {
          return false;
        }
        const sd::LongType batch = inputs[0]->sizeAt(0);
        const sd::LongType sequence = inputs[0]->sizeAt(1);
        const sd::LongType headDimension = inputs[0]->sizeAt(3);
        const sd::LongType sectionT =
            slot.args.numIArgs > 0 ? slot.args.iArgs[0] : 24;
        const sd::LongType sectionH =
            slot.args.numIArgs > 1 ? slot.args.iArgs[1] : 20;
        const sd::LongType sectionW =
            slot.args.numIArgs > 2 ? slot.args.iArgs[2] : 20;
        const sd::LongType interleaved =
            slot.args.numIArgs > 3 ? slot.args.iArgs[3] : 0;
        const double frequencyBase =
            slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 10000.0;
        if (batch <= 0 || sequence <= 0 || headDimension <= 0 ||
            headDimension % 2 != 0 || sectionT <= 0 || sectionH <= 0 ||
            sectionW <= 0 || sectionT % 2 != 0 || sectionH % 2 != 0 ||
            sectionW % 2 != 0 ||
            sectionT + sectionH + sectionW != headDimension ||
            (interleaved != 0 && interleaved != 1) ||
            !std::isfinite(frequencyBase) || frequencyBase <= 0.0) {
          return false;
        }
        const sd::DataType positionType = inputs[1]->dataType();
        if (!(DataTypeUtils::isR(positionType) ||
              positionType == DataType::INT32 ||
              positionType == DataType::UINT32)) {
          return false;
        }
        for (int i = 1; i < 4; ++i) {
          if (inputs[i]->dataType() != positionType ||
              inputs[i]->rankOf() != 2 ||
              inputs[i]->sizeAt(0) != batch ||
              inputs[i]->sizeAt(1) != sequence) {
            return false;
          }
        }
        return true;
      }
      case VulkanKernelRecipe::SKIP_RMS_NORM: {
        if ((numIn != 3 && numIn != 4) ||
            (numOut != 1 && numOut != 2) ||
            inputs[0]->rankOf() != 2 ||
            !sameShapeAndType(inputs[0], inputs[1]) ||
            !sameShapeAndType(inputs[0], outputs[0])) {
          return false;
        }
        if (numOut == 2 &&
            !sameShapeAndType(inputs[0], outputs[1])) {
          return false;
        }
        const sd::LongType hidden = inputs[0]->sizeAt(1);
        if (inputs[2]->rankOf() != 1 || inputs[2]->sizeAt(0) != hidden) {
          return false;
        }
        return numIn != 4 ||
               (inputs[3]->rankOf() == 1 &&
                inputs[3]->sizeAt(0) == hidden);
      }
      case VulkanKernelRecipe::RMS_NORM_LINEAR: {
        if (numIn != 3 || numOut != 1 || inputs[0]->rankOf() != 2 ||
            inputs[1]->rankOf() != 1 || inputs[2]->rankOf() != 2 ||
            outputs[0]->rankOf() != 2) {
          return false;
        }
        const sd::LongType rows = inputs[0]->sizeAt(0);
        const sd::LongType hidden = inputs[0]->sizeAt(1);
        const sd::LongType projected = inputs[2]->sizeAt(1);
        return inputs[1]->sizeAt(0) == hidden &&
               inputs[2]->sizeAt(0) == hidden &&
               outputs[0]->sizeAt(0) == rows &&
               outputs[0]->sizeAt(1) == projected;
      }
      case VulkanKernelRecipe::FUSED_GEMM_SWIGLU: {
        if (numIn != 3 || numOut != 1 || inputs[0]->rankOf() != 2 ||
            inputs[1]->rankOf() != 2 || inputs[2]->rankOf() != 2 ||
            outputs[0]->rankOf() != 2) {
          return false;
        }
        const sd::LongType rows = inputs[0]->sizeAt(0);
        const sd::LongType hidden = inputs[0]->sizeAt(1);
        const sd::LongType projected = inputs[1]->sizeAt(1);
        return inputs[1]->sizeAt(0) == hidden &&
               inputs[2]->sizeAt(0) == hidden &&
               inputs[2]->sizeAt(1) == projected &&
               outputs[0]->sizeAt(0) == rows &&
               outputs[0]->sizeAt(1) == projected;
      }
      case VulkanKernelRecipe::FUSED_RMS_NORM_SWIGLU: {
        if (numIn != 4 || numOut != 1 || inputs[0]->rankOf() != 3 ||
            inputs[1]->rankOf() != 1 || inputs[2]->rankOf() != 2 ||
            inputs[3]->rankOf() != 2 || outputs[0]->rankOf() != 3) {
          return false;
        }
        const sd::LongType hidden = inputs[0]->sizeAt(2);
        const sd::LongType projected = inputs[2]->sizeAt(1);
        return inputs[1]->sizeAt(0) == hidden &&
               inputs[2]->sizeAt(0) == hidden &&
               inputs[3]->sizeAt(0) == hidden &&
               inputs[3]->sizeAt(1) == projected &&
               outputs[0]->sizeAt(0) == inputs[0]->sizeAt(0) &&
               outputs[0]->sizeAt(1) == inputs[0]->sizeAt(1) &&
               outputs[0]->sizeAt(2) == projected;
      }
      case VulkanKernelRecipe::DOT_PRODUCT_ATTENTION: {
        const bool outputWeights = slot.args.iArgs[1] != 0;
        if ((slot.args.iArgs[0] != 0 && slot.args.iArgs[0] != 1) ||
            (slot.args.iArgs[1] != 0 && slot.args.iArgs[1] != 1) ||
            (numIn != 3 && numIn != 4) ||
            numOut != (outputWeights ? 2 : 1)) {
          return false;
        }
        const int rank = inputs[0]->rankOf();
        if ((rank != 3 && rank != 4) || inputs[1]->rankOf() != rank ||
            inputs[2]->rankOf() != rank || outputs[0]->rankOf() != rank) {
          return false;
        }
        for (int d = 0; d < rank - 2; ++d) {
          if (inputs[0]->sizeAt(d) != inputs[1]->sizeAt(d) ||
              inputs[0]->sizeAt(d) != inputs[2]->sizeAt(d) ||
              outputs[0]->sizeAt(d) != inputs[0]->sizeAt(d)) {
            return false;
          }
        }
        const sd::LongType queryFeatures = inputs[0]->sizeAt(rank - 2);
        const sd::LongType querySteps = inputs[0]->sizeAt(rank - 1);
        const sd::LongType keySteps = inputs[1]->sizeAt(rank - 1);
        const sd::LongType valueFeatures = inputs[2]->sizeAt(rank - 2);
        if (inputs[1]->sizeAt(rank - 2) != queryFeatures ||
            inputs[2]->sizeAt(rank - 1) != keySteps ||
            outputs[0]->sizeAt(rank - 2) != valueFeatures ||
            outputs[0]->sizeAt(rank - 1) != querySteps) {
          return false;
        }
        if (numIn == 4 &&
            (inputs[3]->rankOf() != 2 ||
             inputs[3]->sizeAt(0) != inputs[0]->sizeAt(0) ||
             inputs[3]->sizeAt(1) != keySteps)) {
          return false;
        }
        if (outputWeights) {
          if (outputs[1]->rankOf() != rank) return false;
          for (int d = 0; d < rank - 2; ++d) {
            if (outputs[1]->sizeAt(d) != inputs[0]->sizeAt(d)) return false;
          }
          if (outputs[1]->sizeAt(rank - 2) != keySteps ||
              outputs[1]->sizeAt(rank - 1) != querySteps) {
            return false;
          }
        }
        return true;
      }
      case VulkanKernelRecipe::FUSED_ATTENTION_PROJECTION: {
        if ((numIn != 2 && numIn != 3) || numOut != 1 ||
            slot.args.numIArgs != 0 || slot.args.numTArgs != 0 ||
            slot.args.numBArgs != 0 ||
            (inputs[0]->rankOf() != 3 && inputs[0]->rankOf() != 4) ||
            inputs[1]->rankOf() != 2 || outputs[0]->rankOf() != 3) {
          return false;
        }
        sd::LongType hidden = inputs[0]->sizeAt(2);
        if (inputs[0]->rankOf() == 4) {
          const sd::LongType headDim = inputs[0]->sizeAt(3);
          if (hidden <= 0 || headDim <= 0 ||
              hidden > std::numeric_limits<sd::LongType>::max() / headDim) {
            return false;
          }
          hidden *= headDim;
        }
        const sd::LongType projected = inputs[1]->sizeAt(1);
        if (inputs[1]->sizeAt(0) != hidden ||
            outputs[0]->sizeAt(0) != inputs[0]->sizeAt(0) ||
            outputs[0]->sizeAt(1) != inputs[0]->sizeAt(1) ||
            outputs[0]->sizeAt(2) != projected) {
          return false;
        }
        return numIn != 3 ||
               (inputs[2]->rankOf() == 1 &&
                inputs[2]->sizeAt(0) == projected);
      }
      case VulkanKernelRecipe::FUSED_ELEMENTWISE_CHAIN: {
        if (numOut != 1 || slot.args.numIArgs < 1 ||
            slot.args.numIArgs > 8 || slot.args.numBArgs != 0 ||
            slot.args.numDArgs != 0 || slot.args.numSArgs != 0 ||
            !sameShapeAndType(inputs[0], outputs[0])) {
          return false;
        }
        int binaryInputs = 0;
        bool hasClip = false;
        for (int i = 0; i < slot.args.numIArgs; ++i) {
          const sd::LongType code = slot.args.iArgs[i];
          const bool known = (code >= 0 && code <= 3) ||
                             (code >= 10 && code <= 42) ||
                             (code >= 50 && code <= 59);
          if (!known) return false;
          if ((code >= 0 && code <= 3) || code == 31 ||
              (code >= 50 && code <= 59)) {
            ++binaryInputs;
          }
          hasClip = hasClip || code == 30;
        }
        if (numIn != 1 + binaryInputs ||
            slot.args.numTArgs != (hasClip ? 2 : 0)) {
          return false;
        }
        if (hasClip &&
            (!std::isfinite(slot.args.tArgs[0]) ||
             !std::isfinite(slot.args.tArgs[1]) ||
             slot.args.tArgs[0] > slot.args.tArgs[1])) {
          return false;
        }
        for (int i = 1; i < numIn; ++i) {
          if (!sameShapeAndType(inputs[0], inputs[i])) return false;
        }
        return true;
      }
      case VulkanKernelRecipe::FUSED_BIAS_DROPOUT_RESIDUAL: {
        if (numIn != 3 || numOut != 1 || inputs[0]->rankOf() < 1 ||
            !sameShapeAndType(inputs[0], inputs[2]) ||
            !sameShapeAndType(inputs[0], outputs[0]) ||
            inputs[1]->rankOf() != 1) {
          return false;
        }
        return inputs[1]->sizeAt(0) ==
               inputs[0]->sizeAt(inputs[0]->rankOf() - 1);
      }
      case VulkanKernelRecipe::SWISH_MUL_BP: {
        if (numIn != 3 || numOut != 2 || inputs[1] == nullptr ||
            inputs[2] == nullptr || !DataTypeUtils::isR(inputs[0]->dataType()) ||
            !DataTypeUtils::isR(inputs[1]->dataType()) ||
            !DataTypeUtils::isR(inputs[2]->dataType()) ||
            !inputs[0]->isSameShape(inputs[1]) ||
            !inputs[0]->isSameShape(inputs[2]) ||
            !sameShapeAndType(inputs[0], outputs[0]) ||
            !sameShapeAndType(inputs[0], outputs[1])) {
          return false;
        }
        return true;
      }
      case VulkanKernelRecipe::SWIGLU:
      case VulkanKernelRecipe::GEGLU:
      case VulkanKernelRecipe::REGLU: {
        if (numIn != 1 || numOut != 1 || slot.args.numIArgs != 0 ||
            slot.args.numTArgs != 0 || slot.args.numBArgs != 0 ||
            inputs[0]->rankOf() < 1 ||
            outputs[0]->rankOf() != inputs[0]->rankOf()) {
          return false;
        }
        const int rank = inputs[0]->rankOf();
        const sd::LongType last = inputs[0]->sizeAt(rank - 1);
        if (last <= 0 || last % 2 != 0 ||
            outputs[0]->sizeAt(rank - 1) != last / 2) {
          return false;
        }
        for (int d = 0; d < rank - 1; ++d) {
          if (inputs[0]->sizeAt(d) != outputs[0]->sizeAt(d)) return false;
        }
        return true;
      }
      default:
        return false;
    }
  }

  // ── Argument-free, same-shape elementwise subset ─────────────────────────
  if constexpr (Policy::ternary) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || numIn != 3 || numOut != 1 ||
        slot.args.numIArgs != 0 || !hasNoBoolDtypeOrStringArgs(slot) ||
        !DataTypeUtils::isB(inputs[0]->dataType()) ||
        inputs[1]->dataType() != inputs[2]->dataType() ||
        inputs[1]->dataType() != outputs[0]->dataType() ||
        !inputs[1]->isSameShape(inputs[2]) ||
        !inputs[1]->isSameShape(outputs[0])) {
      return false;
    }
    return true;
  }

  if constexpr (Policy::binary) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || numOut != 1 || slot.args.numIArgs != 0 ||
        !hasNoBoolDtypeOrStringArgs(slot)) {
      return false;
    }
    if (emitter->family == VulkanKernelFamily::LOGICAL &&
        (numIn != 2 || !DataTypeUtils::isB(inputs[0]->dataType()) ||
         !DataTypeUtils::isB(inputs[1]->dataType()) ||
         !DataTypeUtils::isB(outputs[0]->dataType()))) {
      return false;
    }
    if (emitter->family == VulkanKernelFamily::COMPARISON &&
        !DataTypeUtils::isB(outputs[0]->dataType())) {
      return false;
    }
    if (isActivationBackward(*emitter)) {
      if (numIn != 2 || !DataTypeUtils::isR(inputs[0]->dataType()) ||
          !DataTypeUtils::isR(inputs[1]->dataType()) ||
          !DataTypeUtils::isR(outputs[0]->dataType()) ||
          !inputs[0]->isSameShape(inputs[1]) ||
          !inputs[0]->isSameShape(outputs[0])) {
        return false;
      }
      const bool parameterized =
          hasVulkanScalarArgumentSchema(*emitter);
      return slot.args.numTArgs <= (parameterized ? 1 : 0) &&
             (slot.args.numTArgs == 0 ||
              std::isfinite(slot.args.tArgs[0]));
    }
    if (slot.args.numTArgs != 0) return false;
    if (hasVulkanOpTrait(*emitter, sd::ops::OP_TRAIT_IDENTITY) &&
        numIn == 1) {
      return inputs[0]->isSameShape(outputs[0]);
    }
    if (numIn != 2) return false;
    if ((emitter->layoutSupport & VULKAN_LAYOUT_BROADCAST) == 0) {
      return sameShapeAndType(inputs[0], outputs[0]) &&
             inputs[0]->isSameShape(inputs[1]);
    }
    return broadcastShapeMatches(inputs[0], inputs[1], outputs[0]);
  }

  if constexpr (Policy::unary) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || numOut != 1) return false;
    if (emitter->family == VulkanKernelFamily::CAST) {
      if (numIn != 1 || slot.args.numTArgs != 0 ||
          slot.args.numBArgs != 0 || slot.args.numSArgs != 0 ||
          !inputs[0]->isSameShape(outputs[0])) {
        return false;
      }
      sd::DataType target = sd::DataType::UNKNOWN;
      if (slot.args.numDArgs == 1 && slot.args.numIArgs == 0) {
        target = slot.args.dArgs[0];
      } else if (slot.args.numIArgs == 1 && slot.args.numDArgs == 0) {
        target = static_cast<sd::DataType>(slot.args.iArgs[0]);
      } else {
        return false;
      }
      return target == outputs[0]->dataType();
    }
    if (vulkanArgumentContractAcceptsInputCount(*emitter, 3) &&
        numIn == 3) {
      return hasNoBoolDtypeOrStringArgs(slot) &&
             slot.args.numIArgs == 0 && slot.args.numTArgs == 0 &&
             sameShapeAndType(inputs[0], outputs[0]) &&
             inputs[1]->lengthOf() == 1 && inputs[2]->lengthOf() == 1;
    }
    if (numIn != 1 || slot.args.numBArgs != 0 ||
        slot.args.numSArgs != 0 ||
        !outputDataTypeArgumentsMatch(slot, outputs, numOut)) {
      return false;
    }
    if (unaryProducesFloatingOutput(*emitter)) {
      return inputs[0]->isSameShape(outputs[0]) &&
             DataTypeUtils::isR(outputs[0]->dataType());
    }
    return sameShapeAndType(inputs[0], outputs[0]);
  }

  // ── Row-wise softmax over the exact native axis ──────────────────────────
  if constexpr (Policy::softmax) {
    if (numIn != 1 || numOut != 1 || slot.args.numIArgs > 1 ||
        slot.args.numTArgs != 0 || !hasNoBoolDtypeOrStringArgs(slot)) {
      return false;
    }
    int64_t axis = 1;
    if (slot.args.numIArgs == 1 &&
        !normalizeAxis(slot.args.iArgs[0], 2, axis)) {
      return false;
    }
    return axis == 1 && inputs[0]->rankOf() == 2 &&
           sameShapeAndType(inputs[0], outputs[0]);
  }

  // ── Last-dimension rank-2 layer norm with required gain ──────────────────
  if constexpr (Policy::layerNorm) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || (numIn != 2 && numIn != 3) || numOut != 1 ||
        slot.args.numBArgs > 1 || slot.args.numDArgs != 0 ||
        slot.args.numSArgs != 0) {
      return false;
    }
    if (hasVulkanEmitterTrait(
            *emitter, VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER)) {
      if (slot.args.numIArgs != 0 || slot.args.numTArgs > 1) return false;
      const double epsilon =
          slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 1.0e-5;
      if (!std::isfinite(epsilon) || epsilon <= 0.0) return false;
    } else {
      if (slot.args.numIArgs != 1 || slot.args.numTArgs != 0) return false;
      int64_t axis = -1;
      if (!normalizeAxis(slot.args.iArgs[0], 2, axis) || axis != 1) {
        return false;
      }
    }
    if (inputs[0]->rankOf() != 2 ||
        !sameShapeAndType(inputs[0], outputs[0])) {
      return false;
    }
    for (int i = 1; i < numIn; ++i) {
      if (inputs[i]->rankOf() != 1 ||
          inputs[i]->sizeAt(0) != inputs[0]->sizeAt(1)) {
        return false;
      }
    }
    return true;
  }

  // ── Static axis-0 gather ─────────────────────────────────────────────────
  if constexpr (Policy::gather) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || !usesIndexedLookupSchedule(*emitter) ||
        numIn != 2 || numOut != 1 || slot.args.numTArgs != 0 ||
        slot.args.numDArgs != 0 || slot.args.numSArgs != 0) {
      return false;
    }
    const bool modeLookup = usesModeIndexedLookupSchedule(*emitter);
    if (modeLookup) {
      if (slot.args.numIArgs != 1 || slot.args.numBArgs != 0 ||
          (slot.args.iArgs[0] != 0 && slot.args.iArgs[0] != 1)) {
        return false;
      }
    } else if (usesAxisIndexedLookupSchedule(*emitter)) {
      if (slot.args.numIArgs > 1 || slot.args.numBArgs > 1) return false;
    } else {
      return false;
    }
    int64_t axis = 0;
    if (!modeLookup && slot.args.numIArgs == 1 &&
        !normalizeAxis(slot.args.iArgs[0], inputs[0]->rankOf(), axis)) {
      return false;
    }
    if (axis != 0 || inputs[0]->rankOf() < 1 ||
        (inputs[1]->dataType() != DataType::INT32 &&
         inputs[1]->dataType() != DataType::UINT32) ||
        inputs[1]->rankOf() != 1 ||
        outputs[0]->dataType() != inputs[0]->dataType() ||
        outputs[0]->rankOf() != inputs[0]->rankOf() ||
        outputs[0]->sizeAt(0) != inputs[1]->sizeAt(0)) {
      return false;
    }
    for (int d = 1; d < inputs[0]->rankOf(); ++d) {
      if (outputs[0]->sizeAt(d) != inputs[0]->sizeAt(d)) return false;
    }
    return true;
  }

  // ── Static-axis concat ───────────────────────────────────────────────────
  if constexpr (Policy::concat) {
    if (numIn < 2 || numOut != 1 || slot.args.numIArgs != 1 ||
        slot.args.numTArgs != 0 || slot.args.numBArgs > 1 ||
        (slot.args.numBArgs == 1 && slot.args.bArgs[0]) ||
        slot.args.numDArgs != 0 || slot.args.numSArgs != 0) {
      return false;
    }
    const int rank = inputs[0]->rankOf();
    int64_t axis = -1;
    if (rank < 1 ||
        !normalizeAxis(slot.args.iArgs[0], rank, axis) ||
        outputs[0]->dataType() != inputs[0]->dataType() ||
        outputs[0]->rankOf() != rank) {
      return false;
    }

    sd::LongType concatenated = 0;
    for (int i = 0; i < numIn; ++i) {
      if (inputs[i]->dataType() != inputs[0]->dataType() ||
          inputs[i]->rankOf() != rank) {
        return false;

      }
      for (int d = 0; d < rank; ++d) {

        if (d != axis && inputs[i]->sizeAt(d) != inputs[0]->sizeAt(d)) {
          return false;
        }
      }
      concatenated += inputs[i]->sizeAt(axis);
    }
    for (int d = 0; d < rank; ++d) {
      const sd::LongType expected =
          d == axis ? concatenated : inputs[0]->sizeAt(d);
      if (outputs[0]->sizeAt(d) != expected) return false;
    }
    return true;
  }

  // ── transpose / permute ──────────────────────────────────────────────────
  // Only the static integer-argument form is recordable. A tensor-valued
  // permutation is a real runtime operand and is not represented by this emitter.
  if constexpr (Policy::transpose) {
    if (numIn != 1 || numOut != 1 || slot.args.numTArgs != 0 ||
        !hasNoBoolDtypeOrStringArgs(slot)) {
      return false;
    }
    sd::DataType dt = inputs[0]->dataType();
    if (outputs[0]->dataType() != dt) return false;
    int rank = inputs[0]->rankOf();
    if (rank < 2 || rank > 4 || outputs[0]->rankOf() != rank) return false;

    std::vector<int64_t> permutation;
    if (!permutationForSlot<Policy>(slot, rank, permutation)) return false;
    for (int d = 0; d < rank; ++d) {
      if (outputs[0]->sizeAt(d) != inputs[0]->sizeAt(permutation[d])) return false;
    }
    return true;
  }

  // ── Replay-safe constant generation ─────────────────────────────────────
  if constexpr (Policy::constant) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr ||
        emitter->argumentContract.alternativeCount == 0 ||
        !argumentContractMatchesSlot(
            *emitter, slot, numIn, numOut, outputs) ||
        !argumentContractValuesMatch(*emitter, slot)) {
      return false;
    }

    if (emitter->recipe == VulkanKernelRecipe::MIN_MAX_DATATYPE) {
      if (outputs[0]->rankOf() != 0) return false;
      const sd::DataType requested =
          DataTypeUtils::fromInt(static_cast<int>(slot.args.iArgs[0]));
      double value = 0.0;
      return outputs[0]->dataType() == requested &&
             supportedScalarLimit(
                 requested, slot.args.iArgs[1] != 0, value);
    }

    if (emitter->recipe == VulkanKernelRecipe::UNIFORM_RANDOM) {
      return numIn == 0 && DataTypeUtils::isR(outputs[0]->dataType());
    }

    if (emitter->recipe == VulkanKernelRecipe::EYE) {
      sd::DataType expectedType = sd::DataType::FLOAT32;
      if (slot.args.numTArgs == 1) {
        const double code = slot.args.tArgs[0];
        if (!std::isfinite(code) || std::floor(code) != code ||
            code < static_cast<double>(std::numeric_limits<int>::min()) ||
            code > static_cast<double>(std::numeric_limits<int>::max())) {
          return false;
        }
        expectedType = DataTypeUtils::fromInt(static_cast<int>(code));
      }
      if (!DataTypeUtils::isR(expectedType) ||
          outputs[0]->dataType() != expectedType) {
        return false;
      }
      const bool ordered = slot.args.iArgs[0] == -99 ||
                           slot.args.iArgs[0] == -102;
      const int parameterOffset = ordered ? 1 : 0;
      const int dimensionCount = slot.args.numIArgs - parameterOffset;
      if (dimensionCount < 1) return false;
      const char expectedOrder =
          ordered ? static_cast<char>(-slot.args.iArgs[0]) : 'c';
      if (outputs[0]->ordering() != expectedOrder) return false;
      std::vector<sd::LongType> expectedShape;
      if (dimensionCount > 2) {
        expectedShape.reserve(static_cast<size_t>(dimensionCount));
        for (int i = parameterOffset + 2; i < slot.args.numIArgs; ++i) {
          expectedShape.push_back(slot.args.iArgs[i]);
        }
      }
      const sd::LongType rows = slot.args.iArgs[parameterOffset];
      const sd::LongType columns =
          dimensionCount == 1 ? rows : slot.args.iArgs[parameterOffset + 1];
      expectedShape.push_back(rows);
      expectedShape.push_back(columns);
      if (outputs[0]->rankOf() != static_cast<int>(expectedShape.size())) {
        return false;
      }
      for (int d = 0; d < outputs[0]->rankOf(); ++d) {
        if (expectedShape[static_cast<size_t>(d)] <= 0 ||
            outputs[0]->sizeAt(d) != expectedShape[static_cast<size_t>(d)]) {
          return false;
        }
      }
      return true;
    }

    if (emitter->recipe == VulkanKernelRecipe::RANGE) {
      StaticRangeSpec range;
      if (outputs[0]->rankOf() != 1 || !readStaticRangeSpec(slot, range) ||
          outputs[0]->sizeAt(0) != range.length) {
        return false;
      }
      const sd::DataType outputType = outputs[0]->dataType();
      if (slot.args.numDArgs == 1) {
        if (slot.args.dArgs[0] != outputType) return false;
      } else if (range.integerArguments) {
        // This is the exact default chosen by range's shape function. INT64 is
        // deliberately left to the backend lane until its storage capability
        // is represented in VulkanDeviceCaps.
        if (range.integerLimit > std::numeric_limits<int32_t>::max() ||
            outputType != sd::DataType::INT32) {
          return false;
        }
      } else if (!DataTypeUtils::isR(outputType)) {
        return false;
      }

      if (DataTypeUtils::isR(outputType)) {
        // Frozen IArgs are serialized as f64 attributes before the storage
        // dtype conversion. Keep that metadata conversion exact for f16/f32.
        constexpr uint64_t kLargestExactlyRepresentableInteger =
            uint64_t{1} << 53;
        if (range.integerArguments && outputType != sd::DataType::DOUBLE &&
            (signedMagnitude(range.integerStart) >
                 kLargestExactlyRepresentableInteger ||
             signedMagnitude(range.integerDelta) >
                 kLargestExactlyRepresentableInteger)) {
          return false;
        }
        double maximum = 0.0;
        const double last =
            range.valueStart +
            range.valueDelta * static_cast<double>(range.length - 1);
        return supportedScalarLimit(outputType, true, maximum) &&
               std::isfinite(last) &&
               std::fabs(range.valueStart) <= maximum &&
               std::fabs(range.valueDelta) <= maximum &&
               std::fabs(last) <= maximum;
      }

      const long double convertedStart =
          std::trunc(static_cast<long double>(range.valueStart));
      const long double convertedDelta =
          std::trunc(static_cast<long double>(range.valueDelta));
      const long double convertedLast =
          convertedStart +
          convertedDelta * static_cast<long double>(range.length - 1);
      if (outputType == sd::DataType::INT32) {
        const long double lower =
            static_cast<long double>(std::numeric_limits<int32_t>::min());
        const long double upper =
            static_cast<long double>(std::numeric_limits<int32_t>::max());
        return convertedStart >= lower && convertedStart <= upper &&
               convertedDelta >= lower && convertedDelta <= upper &&
               convertedLast >= lower && convertedLast <= upper;
      }
      if (outputType == sd::DataType::UINT32) {
        const long double lowerDelta =
            static_cast<long double>(std::numeric_limits<int32_t>::min());
        const long double upper =
            static_cast<long double>(std::numeric_limits<uint32_t>::max());
        return convertedStart >= 0.0L && convertedStart <= upper &&
               convertedDelta >= lowerDelta && convertedDelta <= upper &&
               convertedLast >= 0.0L && convertedLast <= upper;
      }
      return false;
    }

    if (emitter->recipe == VulkanKernelRecipe::LIN_SPACE) {
      if (outputs[0]->rankOf() != 1) {
        return false;
      }
      const sd::LongType steps = slot.args.iArgs[0];
      const sd::DataType expectedType =
          slot.args.numDArgs == 1 ? slot.args.dArgs[0]
                                  : sd::DataType::FLOAT32;
      if (steps <= 0 || outputs[0]->sizeAt(0) != steps ||
          outputs[0]->dataType() != expectedType ||
          (!DataTypeUtils::isR(expectedType) &&
           expectedType != sd::DataType::INT32 &&
           expectedType != sd::DataType::UINT32)) {
        return false;
      }
      if (!DataTypeUtils::isR(expectedType) && !caps.fp64) {
        return false;
      }
      const double start = slot.args.tArgs[0];
      const double stepOrEnd = slot.args.tArgs[1];
      const bool endSpecified =
          slot.args.numBArgs == 1 && slot.args.bArgs[0];
      const double step =
          steps == 1
              ? 0.0
              : (endSpecified
                     ? (stepOrEnd - start) /
                           (static_cast<double>(steps) - 1.0)
                     : stepOrEnd);
      const double last =
          start + step * (static_cast<double>(steps) - 1.0);
      if (!std::isfinite(start) || !std::isfinite(stepOrEnd) ||
          !std::isfinite(step) || !std::isfinite(last)) {
        return false;
      }
      const double smallest = std::min(start, last);
      const double largest = std::max(start, last);
      if (expectedType == sd::DataType::INT32) {
        return smallest >=
                   static_cast<double>(std::numeric_limits<int32_t>::min()) &&
               largest <=
                   static_cast<double>(std::numeric_limits<int32_t>::max());
      }
      if (expectedType == sd::DataType::UINT32) {
        return smallest >= 0.0 &&
               largest <=
                   static_cast<double>(std::numeric_limits<uint32_t>::max());
      }
      return true;
    }

    if (emitter->recipe == VulkanKernelRecipe::RANK_OF ||
        emitter->recipe == VulkanKernelRecipe::SIZE_OF ||
        emitter->recipe == VulkanKernelRecipe::SIZE_AT) {
      const bool isSizeAt =
          emitter->recipe == VulkanKernelRecipe::SIZE_AT;
      if (outputs[0]->rankOf() != 0) {
        return false;
      }
      sd::LongType value = 0;
      if (isSizeAt) {
        int64_t axis = -1;
        if (!normalizeAxis(slot.args.iArgs[0], inputs[0]->rankOf(), axis)) {
          return false;
        }
        value = inputs[0]->sizeAt(static_cast<int>(axis));
      } else {
        value = emitter->recipe == VulkanKernelRecipe::RANK_OF
                    ? static_cast<sd::LongType>(inputs[0]->rankOf())
                    : inputs[0]->lengthOf();
      }
      if (outputs[0]->dataType() == sd::DataType::INT32) {
        return value <= std::numeric_limits<int32_t>::max();
      }
      if (outputs[0]->dataType() == sd::DataType::UINT32) {
        return value >= 0 &&
               static_cast<uint64_t>(value) <=
                   std::numeric_limits<uint32_t>::max();
      }
      return DataTypeUtils::isR(outputs[0]->dataType());
    }

    if (emitter->recipe == VulkanKernelRecipe::SHAPE_OF) {
      if (inputs[0]->rankOf() < 1 || outputs[0]->rankOf() != 1 ||
          outputs[0]->sizeAt(0) != inputs[0]->rankOf()) {
        return false;
      }
      const sd::DataType requested =
          DataTypeUtils::fromInt(static_cast<int>(slot.args.iArgs[0]));
      if ((requested != sd::DataType::INT32 &&
           requested != sd::DataType::UINT32) ||
          outputs[0]->dataType() != requested) {
        return false;
      }
      const uint64_t largest =
          requested == sd::DataType::UINT32
              ? static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
              : static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
      for (int d = 0; d < inputs[0]->rankOf(); ++d) {
        const sd::LongType dimension = inputs[0]->sizeAt(d);
        if (dimension < 0 || static_cast<uint64_t>(dimension) > largest) {
          return false;
        }
      }
      return true;
    }

    if (emitter->recipe == VulkanKernelRecipe::ONE_HOT) {
      const sd::LongType depth = slot.args.iArgs[1];
      const int outputRank = inputs[0]->rankOf() + 1;
      int64_t axis = -1;
      if (depth <= 0 || depth > std::numeric_limits<int>::max() ||
          !normalizeAxis(slot.args.iArgs[0], outputRank, axis) ||
          outputs[0]->rankOf() != outputRank) {
        return false;
      }
      const sd::DataType expectedOutputType =
          slot.args.numDArgs == 1 ? slot.args.dArgs[0]
                                  : sd::DataType::FLOAT32;
      if (outputs[0]->dataType() != expectedOutputType) return false;
      for (int od = 0, id = 0; od < outputRank; ++od) {
        const sd::LongType expected =
            od == axis ? depth : inputs[0]->sizeAt(id++);
        if (outputs[0]->sizeAt(od) != expected) return false;
      }
      const double on =
          slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 1.0;
      const double off =
          slot.args.numTArgs > 1 ? slot.args.tArgs[1] : 0.0;
      if (!std::isfinite(on) || !std::isfinite(off)) return false;
      if (expectedOutputType == sd::DataType::INT32) {
        const double lower =
            static_cast<double>(std::numeric_limits<int32_t>::min());
        const double upper =
            static_cast<double>(std::numeric_limits<int32_t>::max());
        return on >= lower && on <= upper && off >= lower && off <= upper;
      }
      if (expectedOutputType == sd::DataType::UINT32) {
        const double upper =
            static_cast<double>(std::numeric_limits<uint32_t>::max());
        return on >= 0.0 && on <= upper && off >= 0.0 && off <= upper;
      }
      return DataTypeUtils::isR(expectedOutputType);
    }

    if (emitter->recipe == VulkanKernelRecipe::FILL_AS) {
      const bool integerArgument = slot.args.numIArgs == 1;
      if (inputs[0]->dataType() != outputs[0]->dataType() ||
          inputs[0]->rankOf() != outputs[0]->rankOf()) {
        return false;
      }
      for (int d = 0; d < inputs[0]->rankOf(); ++d) {
        if (inputs[0]->sizeAt(d) != outputs[0]->sizeAt(d)) return false;
      }
      const double value =
          integerArgument ? static_cast<double>(slot.args.iArgs[0])
                          : slot.args.tArgs[0];
      if (!std::isfinite(value)) return false;
      const sd::DataType outputType = outputs[0]->dataType();
      if (DataTypeUtils::isR(outputType)) {
        constexpr uint64_t kLargestExactlyRepresentableInteger =
            uint64_t{1} << 53;
        if (integerArgument && outputType != sd::DataType::DOUBLE &&
            signedMagnitude(slot.args.iArgs[0]) >
                kLargestExactlyRepresentableInteger) {
          return false;
        }
        double maximum = 0.0;
        return supportedScalarLimit(outputType, true, maximum) &&
               std::fabs(value) <= maximum;
      }
      const long double converted =
          std::trunc(static_cast<long double>(value));
      if (outputType == sd::DataType::INT32) {
        return converted >= static_cast<long double>(
                                std::numeric_limits<int32_t>::min()) &&
               converted <= static_cast<long double>(
                                std::numeric_limits<int32_t>::max());
      }
      if (outputType == sd::DataType::UINT32) {
        return converted >= 0.0L &&
               converted <= static_cast<long double>(
                                std::numeric_limits<uint32_t>::max());
      }
      return false;
    }

    const bool structuralShapeGeneration =
        emitter->family == VulkanKernelFamily::CONSTANT_GENERATION &&
        hasVulkanOpTrait(
            *emitter, sd::ops::OP_TRAIT_CONSTANT_GENERATION |
                          sd::ops::OP_TRAIT_FULLY_WRITING |
                          sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) &&
        vulkanInputIsStructuralIndex(*emitter, 0);
    if (structuralShapeGeneration) {
      const sd::DataType structuralType = inputs[0]->dataType();
      if (!DataTypeUtils::isZ(structuralType) ||
          DataTypeUtils::isB(structuralType) ||
          inputs[0]->lengthOf() != outputs[0]->rankOf()) {
        return false;
      }
      const char order = static_cast<char>(slot.args.iArgs[0]);
      return (order == 'c' || order == 'f') &&
             outputs[0]->ordering() == order &&
             DataTypeUtils::fromInt(static_cast<int>(slot.args.iArgs[1])) ==
                 outputs[0]->dataType();
    }

    if (inputs[0]->rankOf() != outputs[0]->rankOf()) return false;
    for (int d = 0; d < inputs[0]->rankOf(); ++d) {
      if (inputs[0]->sizeAt(d) != outputs[0]->sizeAt(d)) return false;
    }
    if (emitter->recipe == VulkanKernelRecipe::ZEROS_AS) {
      return inputs[0]->dataType() == outputs[0]->dataType();
    }
    if (emitter->recipe != VulkanKernelRecipe::ONES_AS) return false;
    return slot.args.numDArgs == 1
               ? slot.args.dArgs[0] == outputs[0]->dataType()
               : inputs[0]->dataType() == outputs[0]->dataType();
  }

  // ── Static equal split / unstack with one multi-output pipeline ──────────
  if constexpr (Policy::split) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr ||
        !usesMultiOutputPartitionSchedule(*emitter) ||
        numIn != 1 || numOut < 1 || slot.args.numTArgs != 0 ||
        slot.args.numBArgs != 0 || slot.args.numDArgs != 0 ||
        slot.args.numSArgs != 0) {
      return false;
    }
    auto outputMatches = [&](NDArray* output,
                             const std::vector<sd::LongType>& shape) {
      if (output == nullptr || output->dataType() != inputs[0]->dataType() ||
          output->rankOf() != static_cast<int>(shape.size())) {
        return false;
      }
      for (int d = 0; d < output->rankOf(); ++d) {
        if (output->sizeAt(d) != shape[static_cast<size_t>(d)]) return false;
      }
      return true;
    };
    const bool preservesAxis =
        outputs[0] != nullptr &&
        outputs[0]->rankOf() == inputs[0]->rankOf();
    const bool removesAxis =
        outputs[0] != nullptr &&
        outputs[0]->rankOf() + 1 == inputs[0]->rankOf();
    if (!preservesAxis && !removesAxis) return false;
    if (preservesAxis) {
      if ((slot.args.numIArgs != 1 && slot.args.numIArgs != 2) ||
          slot.args.iArgs[0] != numOut || numOut <= 0) {
        return false;
      }
      int64_t axis = 0;
      if (slot.args.numIArgs == 2 &&
          !normalizeAxis(slot.args.iArgs[1], inputs[0]->rankOf(), axis)) {
        return false;
      }
      if (inputs[0]->sizeAt(static_cast<int>(axis)) % numOut != 0) {
        return false;
      }
      std::vector<sd::LongType> shape;
      shape.reserve(static_cast<size_t>(inputs[0]->rankOf()));
      for (int d = 0; d < inputs[0]->rankOf(); ++d) {
        shape.push_back(
            d == axis ? inputs[0]->sizeAt(d) / numOut
                      : inputs[0]->sizeAt(d));
      }
      for (int i = 0; i < numOut; ++i) {
        if (!outputMatches(outputs[i], shape)) return false;
      }
      return true;
    }
    if (slot.args.numIArgs != 1 && slot.args.numIArgs != 2) {
      return false;
    }
    int64_t axis = -1;
    if (!normalizeAxis(slot.args.iArgs[0], inputs[0]->rankOf(), axis) ||
        inputs[0]->sizeAt(static_cast<int>(axis)) != numOut ||
        (slot.args.numIArgs == 2 && slot.args.iArgs[1] != numOut)) {
      return false;
    }
    std::vector<sd::LongType> shape;
    shape.reserve(static_cast<size_t>(inputs[0]->rankOf() - 1));
    for (int d = 0; d < inputs[0]->rankOf(); ++d) {
      if (d != axis) shape.push_back(inputs[0]->sizeAt(d));
    }
    for (int i = 0; i < numOut; ++i) {
      if (!outputMatches(outputs[i], shape)) return false;
    }
    return true;
  }

  // ── Replay-safe rank-N data movement ─────────────────────────────────────
  // Frozen integer arguments define static coordinate contracts. Runtime
  // index/control tensors remain descriptor bindings and are loaded by recipes
  // such as gather_nd and roll on every eager execution or replay.
  if constexpr (Policy::movement) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || numOut != 1 ||
        slot.args.numTArgs != 0 || slot.args.numDArgs != 0 ||
        slot.args.numSArgs != 0) {
      return false;
    }

    auto sameShape = [](NDArray* lhs, NDArray* rhs) {
      if (lhs == nullptr || rhs == nullptr || lhs->rankOf() != rhs->rankOf()) {
        return false;
      }
      for (int d = 0; d < lhs->rankOf(); ++d) {
        if (lhs->sizeAt(d) != rhs->sizeAt(d)) return false;
      }
      return true;
    };
    auto samePayload = [&](NDArray* input) {
      return input != nullptr && outputs[0] != nullptr &&
             input->dataType() == outputs[0]->dataType();
    };
    auto checkedDimensionProduct = [](sd::LongType dimension,
                                      sd::LongType multiplier,
                                      sd::LongType& result) {
      if (dimension <= 0 || multiplier <= 0) return false;
      const uint64_t product = static_cast<uint64_t>(dimension) *
                               static_cast<uint64_t>(multiplier);
      if (product > static_cast<uint64_t>(
                        std::numeric_limits<sd::LongType>::max())) {
        return false;
      }
      result = static_cast<sd::LongType>(product);
      return true;
    };

    switch (emitter->recipe) {
      case VulkanKernelRecipe::GATHER_ND: {
        if (numIn != 2 || slot.args.numIArgs != 0 ||
            slot.args.numBArgs > 1 ||
            (slot.args.numBArgs == 1 && slot.args.bArgs[0]) ||
            !samePayload(inputs[0]) || inputs[0]->rankOf() < 1 ||
            inputs[1]->rankOf() < 1 ||
            (inputs[1]->dataType() != DataType::INT32 &&
             inputs[1]->dataType() != DataType::UINT32)) {
          return false;
        }
        const int inputRank = inputs[0]->rankOf();
        const int indicesRank = inputs[1]->rankOf();
        const sd::LongType indexedRank = inputs[1]->sizeAt(indicesRank - 1);
        if (indexedRank <= 0 || indexedRank > inputRank) return false;
        for (sd::LongType d = 0; d < indexedRank; ++d) {
          if (inputs[0]->sizeAt(static_cast<int>(d)) <= 0 ||
              inputs[0]->sizeAt(static_cast<int>(d)) >
                  std::numeric_limits<int32_t>::max()) {
            return false;
          }
        }
        const int expectedRank =
            indicesRank - 1 + inputRank - static_cast<int>(indexedRank);
        if (outputs[0]->rankOf() != expectedRank) return false;
        int outputDimension = 0;
        for (int d = 0; d < indicesRank - 1; ++d, ++outputDimension) {
          if (outputs[0]->sizeAt(outputDimension) != inputs[1]->sizeAt(d)) {
            return false;
          }
        }
        for (int d = static_cast<int>(indexedRank); d < inputRank;
             ++d, ++outputDimension) {
          if (outputs[0]->sizeAt(outputDimension) != inputs[0]->sizeAt(d)) {
            return false;
          }
        }
        return true;
      }
      case VulkanKernelRecipe::TILE: {
        if (numIn != 1 || !samePayload(inputs[0]) ||
            slot.args.numBArgs != 0 ||
            slot.args.numIArgs != inputs[0]->rankOf() ||
            outputs[0]->rankOf() != inputs[0]->rankOf()) {
          return false;
        }
        for (int d = 0; d < inputs[0]->rankOf(); ++d) {
          sd::LongType expected = 0;
          if (!checkedDimensionProduct(inputs[0]->sizeAt(d),
                                       slot.args.iArgs[d], expected) ||
              outputs[0]->sizeAt(d) != expected) {
            return false;
          }
        }
        return true;
      }
      case VulkanKernelRecipe::REPEAT: {
        if (numIn != 1 || !samePayload(inputs[0]) ||
            slot.args.numBArgs != 0 || slot.args.numIArgs < 2 ||
            outputs[0]->rankOf() != inputs[0]->rankOf()) {
          return false;
        }
        const int rank = inputs[0]->rankOf();
        int64_t axis = -1;
        if (!normalizeAxis(slot.args.iArgs[slot.args.numIArgs - 1], rank,
                           axis)) {
          return false;
        }
        const int repeatCount = slot.args.numIArgs - 1;
        if (repeatCount != 1 &&
            repeatCount != inputs[0]->sizeAt(static_cast<int>(axis))) {
          return false;
        }
        uint64_t repeatedSize = 0;
        if (repeatCount == 1) {
          sd::LongType expected = 0;
          if (!checkedDimensionProduct(
                  inputs[0]->sizeAt(static_cast<int>(axis)),
                  slot.args.iArgs[0], expected)) {
            return false;
          }
          repeatedSize = static_cast<uint64_t>(expected);
        } else {
          for (int i = 0; i < repeatCount; ++i) {
            if (slot.args.iArgs[i] < 0 ||
                repeatedSize >
                    static_cast<uint64_t>(
                        std::numeric_limits<sd::LongType>::max()) -
                        static_cast<uint64_t>(slot.args.iArgs[i])) {
              return false;
            }
            repeatedSize += static_cast<uint64_t>(slot.args.iArgs[i]);
          }
          if (repeatedSize == 0) return false;
        }
        for (int d = 0; d < rank; ++d) {
          const sd::LongType expected =
              d == axis ? static_cast<sd::LongType>(repeatedSize)
                        : inputs[0]->sizeAt(d);
          if (outputs[0]->sizeAt(d) != expected) return false;
        }
        return true;
      }
      case VulkanKernelRecipe::REVERSE: {
        if (numIn != 1 || !samePayload(inputs[0]) ||
            slot.args.numBArgs != 0 || !sameShape(inputs[0], outputs[0])) {
          return false;
        }
        std::set<int64_t> axes;
        for (int i = 0; i < slot.args.numIArgs; ++i) {
          int64_t axis = -1;
          if (!normalizeAxis(slot.args.iArgs[i], inputs[0]->rankOf(), axis)) {
            return false;
          }
          axes.insert(axis);
        }
        return true;
      }
      case VulkanKernelRecipe::ROLL: {
        const bool aliasesInput =
            inputs[0] == outputs[0] ||
            (inputs[0] != nullptr && outputs[0] != nullptr &&
             inputs[0]->dataBuffer() == outputs[0]->dataBuffer());
        if (numIn < 1 || numIn > 3 || !samePayload(inputs[0]) ||
            !sameShape(inputs[0], outputs[0]) || aliasesInput ||
            slot.args.numBArgs != 0) {
          return false;
        }
        if (numIn == 1) {
          if (slot.args.numIArgs < 1) return false;
          for (int i = 1; i < slot.args.numIArgs; ++i) {
            int64_t axis = -1;
            if (!normalizeAxis(slot.args.iArgs[i], inputs[0]->rankOf(), axis)) {
              return false;
            }
          }
          return true;
        }
        if (slot.args.numIArgs != 0 || inputs[1] == nullptr) return false;
        if (numIn == 2) return inputs[1]->lengthOf() == 1;
        return inputs[2] != nullptr &&
               inputs[1]->rankOf() == inputs[2]->rankOf() &&
               inputs[1]->lengthOf() == inputs[2]->lengthOf();
      }
      case VulkanKernelRecipe::SLICE: {
        if (numIn != 1 || !samePayload(inputs[0]) ||
            slot.args.numBArgs != 0 || inputs[0]->rankOf() < 1 ||
            slot.args.numIArgs < 2 * inputs[0]->rankOf()) {
          return false;
        }
        const int rank = inputs[0]->rankOf();
        std::vector<sd::LongType> sizes(static_cast<size_t>(rank));
        for (int d = 0; d < rank; ++d) {
          const sd::LongType begin = slot.args.iArgs[d];
          sd::LongType size = slot.args.iArgs[rank + d];
          const sd::LongType dimension = inputs[0]->sizeAt(d);
          if (begin < 0 || begin > dimension) return false;
          if (size == -1) size = dimension - begin;
          if (size <= 0 || size > dimension - begin) return false;
          sizes[static_cast<size_t>(d)] = size;
        }
        if (rank == 1 && sizes[0] == 1 && outputs[0]->rankOf() == 0) {
          return true;
        }
        if (outputs[0]->rankOf() != rank) return false;
        for (int d = 0; d < rank; ++d) {
          if (outputs[0]->sizeAt(d) != sizes[static_cast<size_t>(d)]) {
            return false;
          }
        }
        return true;
      }
      case VulkanKernelRecipe::STRIDED_SLICE: {
        const int rank = inputs[0]->rankOf();
        if (numIn != 1 || !samePayload(inputs[0]) ||
            slot.args.numBArgs != 0 || rank < 1 ||
            slot.args.numIArgs != 5 + 3 * rank ||
            outputs[0]->rankOf() != rank) {
          return false;
        }
        for (int mask = 0; mask < 5; ++mask) {
          if (slot.args.iArgs[mask] != 0) return false;
        }
        for (int d = 0; d < rank; ++d) {
          const sd::LongType begin = slot.args.iArgs[5 + d];
          const sd::LongType end = slot.args.iArgs[5 + rank + d];
          const sd::LongType stride = slot.args.iArgs[5 + 2 * rank + d];
          const sd::LongType dimension = inputs[0]->sizeAt(d);
          if (begin < 0 || begin >= end || end > dimension || stride <= 0) {
            return false;
          }
          const sd::LongType expected =
              (end - begin + stride - 1) / stride;
          if (outputs[0]->sizeAt(d) != expected) return false;
        }
        return true;
      }
      case VulkanKernelRecipe::STACK: {
        if (numIn < 1 || slot.args.numBArgs != 0 ||
            slot.args.numIArgs > 1 || !samePayload(inputs[0])) {
          return false;
        }
        const int inputRank = inputs[0]->rankOf();
        int64_t axis = 0;
        if (slot.args.numIArgs == 1 &&
            !normalizeAxis(slot.args.iArgs[0], inputRank + 1, axis)) {
          return false;
        }
        for (int i = 0; i < numIn; ++i) {
          if (!samePayload(inputs[i])) return false;
          if (sameShape(inputs[0], inputs[i])) continue;
          const bool scalarVectorOne =
              inputs[0]->lengthOf() == 1 && inputs[i]->lengthOf() == 1 &&
              ((inputs[0]->rankOf() == 0 && inputs[i]->rankOf() == 1) ||
               (inputs[0]->rankOf() == 1 && inputs[i]->rankOf() == 0));
          if (!scalarVectorOne) return false;
        }
        if (outputs[0]->rankOf() != inputRank + 1) return false;
        int sourceDimension = 0;
        for (int d = 0; d < outputs[0]->rankOf(); ++d) {
          const sd::LongType expected =
              d == axis ? static_cast<sd::LongType>(numIn)
                        : inputs[0]->sizeAt(sourceDimension++);
          if (outputs[0]->sizeAt(d) != expected) return false;
        }
        return true;
      }
      case VulkanKernelRecipe::TRIU: {
        return numIn == 1 && slot.args.numBArgs == 0 &&
               slot.args.numIArgs <= 1 && inputs[0]->rankOf() >= 2 &&
               samePayload(inputs[0]) && sameShape(inputs[0], outputs[0]);
      }
      case VulkanKernelRecipe::TRIU_BP: {
        const int rank = inputs[0] == nullptr ? -1 : inputs[0]->rankOf();
        return numIn == 2 && slot.args.numBArgs == 0 &&
               slot.args.numIArgs <= 1 && (rank == 0 || rank >= 2) &&
               samePayload(inputs[0]) && samePayload(inputs[1]) &&
               sameShape(inputs[0], inputs[1]) &&
               sameShape(inputs[0], outputs[0]);
      }
      default:
        return false;
    }
  }

  if constexpr (Policy::multiOutputElementwise) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr ||
        !usesMultiOutputNormalizationSchedule(*emitter) ||
        numIn != 3 || numOut != 2 || inputs[0]->lengthOf() != 1 ||
        slot.args.numIArgs != 0 || slot.args.numTArgs > 1 ||
        !hasNoBoolDtypeOrStringArgs(slot) ||
        (slot.args.numTArgs == 1 && !std::isfinite(slot.args.tArgs[0])) ||
        !inputs[1]->isSameShape(inputs[2]) ||
        !inputs[1]->isSameShape(outputs[0]) ||
        !inputs[1]->isSameShape(outputs[1]) ||
        !DataTypeUtils::isR(outputs[0]->dataType()) ||
        !DataTypeUtils::isR(outputs[1]->dataType())) {
      return false;
    }
    return true;
  }

  if constexpr (Policy::reduction) {
    if (numIn != 1 || numOut != 1 || inputs[0]->rankOf() < 1) {
      return false;
    }

    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr) return false;
    const bool indexReduction = hasVulkanEmitterTrait(
        *emitter, VULKAN_EMITTER_TRAIT_INDEX_RESULT);
    if (indexReduction) {
      const auto outputType = outputs[0]->dataType();
      if ((outputType != DataType::INT32 &&
           outputType != DataType::UINT32) ||
          slot.args.numDArgs != 1 || slot.args.dArgs[0] != outputType) {
        return false;
      }
    } else if (reductionProducesFloatingOutput(*emitter)) {
      if (!DataTypeUtils::isR(outputs[0]->dataType())) return false;
    } else if (outputs[0]->dataType() != inputs[0]->dataType()) {
      return false;
    }
    std::vector<int64_t> axes;
    bool keepDims = false;
    bool biasCorrected = false;
    if (hasVulkanEmitterTrait(*emitter, VULKAN_EMITTER_TRAIT_IMPLICIT_LAST_AXIS)) {
      if (slot.args.numIArgs > 1 || slot.args.numTArgs != 0 ||
          slot.args.numBArgs != 0 || slot.args.numSArgs != 0 ||
          !outputDataTypeArgumentsMatch(slot, outputs, numOut)) {
        return false;
      }
      axes.push_back(inputs[0]->rankOf() - 1);
      keepDims = slot.args.numIArgs == 0 || slot.args.iArgs[0] != 0;
    } else if (!reductionForSlot(
                   slot, inputs[0], *emitter, outputs, numOut,
                   axes, keepDims, biasCorrected)) {
      return false;
    }
    if (indexReduction &&
        inputs[0]->sizeAt(static_cast<int>(axes.front())) >
            std::numeric_limits<int32_t>::max()) {
      return false;
    }
    return reductionOutputMatches(inputs[0], outputs[0], axes, keepDims);
  }

  return false;
}

template <typename Policy>
static bool validateVulkanOp(const NativeSlot& slot,
                             NDArray** inputs, int numIn,
                             NDArray** outputs, int numOut,
                             const VulkanDeviceCaps& caps) {
  if constexpr (Policy::constant) {
    if (numIn < 0 || (numIn > 0 &&
                      (inputs == nullptr || inputs[0] == nullptr))) {
      return false;
    }
  } else {
    if (numIn <= 0 || inputs == nullptr || inputs[0] == nullptr) {
      return false;
    }
  }
  return opIsRecordableTyped<Policy>(
      slot, inputs, numIn, outputs, numOut, caps);
}

template <typename Policy>
static bool vulkanDispatchGeometry(const NativeSlot&,
                                   NDArray**, int,
                                   NDArray** outputs, int numOut,
                                   DispatchGeometry& geometry) {
  geometry = {};
  if constexpr (Policy::matmul) {
    if (numOut != 1 || outputs == nullptr || outputs[0] == nullptr) return false;
    const int rank = outputs[0]->rankOf();
    if (rank != 2 && rank != 3) return false;
    const int matrixAxis = rank - 2;
    const sd::LongType x = outputs[0]->sizeAt(matrixAxis + 1);
    const sd::LongType y = outputs[0]->sizeAt(matrixAxis);
    const sd::LongType z = rank == 3 ? outputs[0]->sizeAt(0) : 1;
    constexpr uint64_t kMaxDispatchCount =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    if (x <= 0 || y <= 0 || z <= 0 ||
        static_cast<uint64_t>(x) > kMaxDispatchCount ||
        static_cast<uint64_t>(y) > kMaxDispatchCount ||
        static_cast<uint64_t>(z) > kMaxDispatchCount) {
      return false;
    }
    geometry.x = static_cast<uint32_t>(x);
    geometry.y = static_cast<uint32_t>(y);
    geometry.z = static_cast<uint32_t>(z);
  }
  return true;
}

static bool validateOperandTypeContract(
    const VulkanKernelEmitterInfo& emitter, NDArray** inputs, int numIn,
    const VulkanDeviceCaps& caps) {
  const uint32_t scalarMask =
      emitter.operandTypeContract.scalar32InputMask;
  const uint32_t integer32Mask =
      emitter.operandTypeContract.integer32InputMask;
  const uint32_t integer64Mask =
      emitter.operandTypeContract.integer64InputMask;
  const uint32_t integerIndexMask =
      emitter.operandTypeContract.integerIndexInputMask;
  const uint32_t specialMask =
      scalarMask | integer32Mask | integer64Mask | integerIndexMask;
  const uint32_t overlappingRoles =
      (scalarMask & integer32Mask) | (scalarMask & integer64Mask) |
      (scalarMask & integerIndexMask) | (integer32Mask & integer64Mask) |
      (integer32Mask & integerIndexMask) |
      (integer64Mask & integerIndexMask);
  if (numIn < 0 || numIn > 16 || overlappingRoles != 0) return false;

  bool haveUniformSpecialType = false;
  std::string uniformSpecialType;
  bool uniformSpecialUnsigned = false;
  for (int i = 0; i < numIn; ++i) {
    const uint32_t bit = uint32_t{1} << i;
    if ((specialMask & bit) == 0) continue;
    std::string storage;
    std::string scalarType;
    bool isUnsigned = false;
    if (inputs == nullptr || inputs[i] == nullptr ||
        !selectMlirScalarTypes(inputs[i]->dataType(), caps, storage,
                               scalarType, isUnsigned) ||
        ((integer32Mask & bit) != 0 && scalarType != "i32") ||
        ((integer64Mask & bit) != 0 && scalarType != "i64") ||
        ((integerIndexMask & bit) != 0 && scalarType != "i32" &&
         scalarType != "i64") ||
        ((scalarMask & bit) != 0 && scalarType != "i32" &&
         scalarType != "f32" && scalarType != "f64")) {
      return false;
    }
    if (emitter.operandTypeContract.requireUniformSpecialInputs) {
      if (!haveUniformSpecialType) {
        haveUniformSpecialType = true;
        uniformSpecialType = storage;
        uniformSpecialUnsigned = isUnsigned;
      } else if (storage != uniformSpecialType ||
                 isUnsigned != uniformSpecialUnsigned) {
        return false;
      }
    }
  }
  return true;
}

static bool validateCatalogOp(const NativeSlot& slot,
                              NDArray** inputs, int numIn,
                              NDArray** outputs, int numOut,
                              const VulkanDeviceCaps& caps) {
  const auto* emitter = emitterForSlot(slot);
  const bool zeroInputConstant =
      emitter != nullptr &&
      emitter->family == VulkanKernelFamily::CONSTANT_GENERATION &&
      numIn == 0;
  if (emitter == nullptr || numIn < 0 ||
      (!zeroInputConstant &&
       (numIn == 0 || inputs == nullptr || inputs[0] == nullptr)) ||
      numOut <= 0 || outputs == nullptr || outputs[0] == nullptr) {
    return false;
  }
  for (int i = 0; i < numIn; ++i) {
    if (inputs[i] == nullptr) return false;
    if (vulkanInputIsStructuralIndex(*emitter,
                                     static_cast<unsigned>(i))) {
      // dtypeSupport describes values consumed by the generated kernel.
      // Structural indices remain bound descriptor metadata but are never
      // loaded, so validate their framework type role instead of applying
      // payload/device arithmetic capabilities.
      const sd::DataType structuralType = inputs[i]->dataType();
      if (!DataTypeUtils::isZ(structuralType) ||
          DataTypeUtils::isB(structuralType)) {
        return false;
      }
      continue;
    }
    const uint32_t dtype = vulkanDtypeBit(inputs[i]->dataType());
    if (dtype == VULKAN_DTYPE_NONE ||
        (emitter->dtypeSupport & dtype) == 0) {
      return false;
    }
  }
  for (int i = 0; i < numOut; ++i) {
    if (outputs[i] == nullptr) return false;
    const uint32_t dtype = vulkanDtypeBit(outputs[i]->dataType());
    if (dtype == VULKAN_DTYPE_NONE ||
        (emitter->dtypeSupport & dtype) == 0) {
      return false;
    }
  }
  if (!validateOperandTypeContract(*emitter, inputs, numIn, caps)) {
    return false;
  }
  // Catalogue rank bounds describe the primary input contract. Zero-input
  // constant generators use their output rank instead.
  const int rank = zeroInputConstant ? outputs[0]->rankOf()
                                     : inputs[0]->rankOf();
  if (rank < emitter->minimumRank ||
      (emitter->maximumRank >= 0 && rank > emitter->maximumRank)) {
    return false;
  }
  if (emitter->family == VulkanKernelFamily::ELEMENTWISE_UNARY &&
      !usesContractMovementSchedule(*emitter) &&
      !unaryArgumentsMatch(*emitter, slot, numIn, outputs, numOut)) {
    return false;
  }

  if (usesStructuredComputeSchedule(*emitter)) {
    return validateVulkanOp<StructuredComputePolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }

  switch (emitter->loweringContract) {
    case VulkanLoweringContract::SOFTMAX:
      return validateVulkanOp<SoftmaxPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanLoweringContract::LAYER_NORM:
      return validateVulkanOp<LayerNormPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanLoweringContract::FUSED_LLM:
    case VulkanLoweringContract::DEFAULT:
    case VulkanLoweringContract::LINEAR_COPY:
    case VulkanLoweringContract::INDEXED_TAD_MOVEMENT:
      break;
  }

  if (usesBatchedMatrixListSchedule(*emitter)) {
    return validateVulkanOp<BatchedMatrixListPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesIndexedAccumulationSchedule(*emitter)) {
    return validateVulkanOp<IndexedAccumulationPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesIndexedTadMovementSchedule(*emitter)) {
    return validateVulkanOp<IndexedTadMovementPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesContractMovementSchedule(*emitter)) {
    return validateVulkanOp<ContractMovementPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesRowwiseEpsilonNormalizationSchedule(*emitter)) {
    return validateVulkanOp<RmsNormPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesMultiOutputNormalizationSchedule(*emitter)) {
    return validateVulkanOp<MultiOutputElementwisePolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesCachedRotarySchedule(*emitter)) {
    return validateVulkanOp<RopePolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesRankPermutationSchedule(*emitter)) {
    if (usesDefaultReversePermutationSchedule(*emitter)) {
      return validateVulkanOp<TransposePolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    }
    return validateVulkanOp<PermutePolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }

  switch (emitter->family) {
    case VulkanKernelFamily::MATMUL:
      return validateVulkanOp<MatmulPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::ELEMENTWISE_BINARY:
      return validateVulkanOp<BinaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::ELEMENTWISE_UNARY:
      return validateVulkanOp<UnaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::COMPARISON:
    case VulkanKernelFamily::LOGICAL:
      if (emitter->recipe == VulkanKernelRecipe::BOOLEAN_NOT) {
        return validateVulkanOp<UnaryPolicy>(
            slot, inputs, numIn, outputs, numOut, caps);
      }
      return validateVulkanOp<BinaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::TERNARY:
      return validateVulkanOp<TernaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::NORMALIZATION:
      return false;
    case VulkanKernelFamily::DATA_MOVEMENT:
      if (usesIndexedLookupSchedule(*emitter)) {
        return validateVulkanOp<GatherPolicy>(
            slot, inputs, numIn, outputs, numOut, caps);
      }
      if (usesVariadicAxisConcatSchedule(*emitter)) {
        return validateVulkanOp<ConcatPolicy>(
            slot, inputs, numIn, outputs, numOut, caps);
      }
      if (usesMultiOutputPartitionSchedule(*emitter)) {
        return validateVulkanOp<SplitPolicy>(
            slot, inputs, numIn, outputs, numOut, caps);
      }
      return validateVulkanOp<MovementPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::REDUCTION:
      return validateVulkanOp<ReductionPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::CONSTANT_GENERATION:
      return validateVulkanOp<ConstantPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::CAST:
      return validateVulkanOp<UnaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::UNKNOWN:
    case VulkanKernelFamily::ATTENTION:
      return false;
  }
  return false;
}

static bool catalogDispatchGeometry(const NativeSlot& slot,
                                    NDArray** inputs, int numIn,
                                    NDArray** outputs, int numOut,
                                    DispatchGeometry& geometry) {
  const auto* emitter = emitterForSlot(slot);
  if (emitter == nullptr || outputs == nullptr || numOut < 1 ||
      outputs[0] == nullptr) {
    return false;
  }
  if (emitter->family == VulkanKernelFamily::MATMUL &&
      !usesBatchedMatrixListSchedule(*emitter)) {
    return vulkanDispatchGeometry<MatmulPolicy>(
        slot, inputs, numIn, outputs, numOut, geometry);
  }

  geometry = {};
  constexpr uint64_t kMaxDispatchCount =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  auto checked = [&](sd::LongType value, uint32_t& destination) {
    if (value <= 0 || static_cast<uint64_t>(value) > kMaxDispatchCount) {
      return false;
    }
    destination = static_cast<uint32_t>(value);
    return true;
  };

  if (hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_DISPATCH_OUTER_ROWS)) {
    const int rank = outputs[0]->rankOf();
    if (rank < 1) return false;
    uint64_t rows = 1;
    for (int d = 0; d < rank - 1; ++d) {
      const sd::LongType dimension = outputs[0]->sizeAt(d);
      if (dimension <= 0 ||
          rows > kMaxDispatchCount / static_cast<uint64_t>(dimension)) {
        return false;
      }
      rows *= static_cast<uint64_t>(dimension);
    }
    geometry.x = static_cast<uint32_t>(rows);
    return true;
  }
  if (hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_DISPATCH_SINGLE)) {
    geometry.x = 1;
    return true;
  }

  if (hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_DISPATCH_FIRST_DIM)) {
    return checked(outputs[0]->sizeAt(0), geometry.x);
  }

  if (hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_DISPATCH_ROTARY_GRID)) {
    if (outputs[0]->rankOf() != 4) return false;
    const sd::LongType rotaryPairs = outputs[0]->sizeAt(3) / 2;
    const sd::LongType heads = outputs[0]->sizeAt(2);
    const sd::LongType batch = outputs[0]->sizeAt(0);
    const sd::LongType sequence = outputs[0]->sizeAt(1);
    if (batch <= 0 || sequence <= 0 ||
        static_cast<uint64_t>(batch) >
            kMaxDispatchCount / static_cast<uint64_t>(sequence)) {
      return false;
    }
    const sd::LongType batchSequence = batch * sequence;
    return checked(rotaryPairs, geometry.x) &&
           checked(heads, geometry.y) &&
           checked(batchSequence, geometry.z);
  }

  if (hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_DISPATCH_BATCH_SEQUENCE)) {
    const sd::LongType batch = outputs[0]->sizeAt(0);
    const sd::LongType sequence = outputs[0]->sizeAt(1);
    if (batch <= 0 || sequence <= 0 ||
        static_cast<uint64_t>(batch) >
            kMaxDispatchCount / static_cast<uint64_t>(sequence)) {
      return false;
    }
    return checked(batch * sequence, geometry.x);
  }
  if (hasVulkanEmitterTrait(
          *emitter, VULKAN_EMITTER_TRAIT_DISPATCH_ATTENTION_QUERY)) {
    uint64_t invocations = 1;
    const int rank = outputs[0]->rankOf();
    for (int d = 0; d < rank - 2; ++d) {
      const sd::LongType dim = outputs[0]->sizeAt(d);
      if (dim <= 0 || invocations >
                          kMaxDispatchCount / static_cast<uint64_t>(dim)) {
        return false;
      }
      invocations *= static_cast<uint64_t>(dim);
    }
    const sd::LongType querySteps = outputs[0]->sizeAt(rank - 1);
    if (querySteps <= 0 ||
        invocations > kMaxDispatchCount /
                          static_cast<uint64_t>(querySteps)) {
      return false;
    }
    return checked(
        static_cast<sd::LongType>(
            invocations * static_cast<uint64_t>(querySteps)),
        geometry.x);
  }

  // Variable-input concat intentionally uses one device invocation containing
  // the complete copy schedule. Pairwise same-shape copies and trait-routed axis
  // partitioning write every destination from one real launch.
  if (hasVulkanOpTrait(*emitter, sd::ops::OP_TRAIT_CONCAT)) return true;
  if (usesSameShapeCopySchedule(*emitter) &&
      !usesTrailingPayloadCopySchedule(*emitter) && numOut > 1) {
    uint64_t invocations = 0;
    for (int i = 0; i < numOut; ++i) {
      if (outputs[i] == nullptr || outputs[i]->lengthOf() <= 0) return false;
      const uint64_t length =
          static_cast<uint64_t>(outputs[i]->lengthOf());
      if (length > kMaxDispatchCount ||
          invocations > kMaxDispatchCount - length) {
        return false;
      }
      invocations += length;
    }
    geometry.x = static_cast<uint32_t>(invocations);
    return true;
  }
  if (usesAxisPartitionSchedule(*emitter)) {
    return inputs != nullptr && numIn >= 1 && inputs[0] != nullptr &&
           checked(inputs[0]->lengthOf(), geometry.x);
  }
  return checked(outputs[0]->lengthOf(), geometry.x);
}

}  // anonymous namespace

bool VulkanSegmentRecorder::opIsRecordable(const NativeSlot& slot,
                                            NDArray** inputs, int numIn,
                                            NDArray** outputs, int numOut,
                                            const VulkanDeviceCaps& caps) {
  const VulkanOpHandler* handler = findVulkanHandler(slot);
  if (handler == nullptr) {
    DSP_DIAG(GRAPH_REPLAY,
             "Vulkan recordability reject: descriptor=%s hash=%lld has no handler",
             slot.ident.opName.c_str(),
             static_cast<long long>(slot.ident.opHash));
    return false;
  }

  const bool recordable =
      handler->validate(slot, inputs, numIn, outputs, numOut, caps);
  if (!recordable) {
    const auto* emitter = emitterForSlot(slot);
    const bool hasExplicitArgumentContract =
        emitter != nullptr && emitter->argumentContract.alternativeCount != 0;
    const bool argumentsMatch =
        emitter != nullptr &&
        (!hasExplicitArgumentContract ||
         argumentContractMatchesSlot(
             *emitter, slot, numIn, numOut, outputs));
    const bool argumentValuesMatch =
        emitter != nullptr && argumentContractValuesMatch(*emitter, slot);
    const bool haveInput0 =
        inputs != nullptr && numIn > 0 && inputs[0] != nullptr;
    const bool haveInput1 =
        inputs != nullptr && numIn > 1 && inputs[1] != nullptr;
    const bool haveOutput0 =
        outputs != nullptr && numOut > 0 && outputs[0] != nullptr;
    const bool input0Structural =
        emitter != nullptr && haveInput0 && vulkanInputIsStructuralIndex(*emitter, 0);
    const uint32_t input0Dtype =
        emitter != nullptr && haveInput0 ? vulkanDtypeBit(inputs[0]->dataType())
                                         : VULKAN_DTYPE_NONE;
    const uint32_t output0Dtype =
        emitter != nullptr && haveOutput0 ? vulkanDtypeBit(outputs[0]->dataType())
                                          : VULKAN_DTYPE_NONE;
    const bool input0DtypeSupported =
        emitter != nullptr && haveInput0 &&
        (input0Structural
             ? DataTypeUtils::isZ(inputs[0]->dataType()) &&
                   !DataTypeUtils::isB(inputs[0]->dataType())
             : input0Dtype != VULKAN_DTYPE_NONE &&
                   (emitter->dtypeSupport & input0Dtype) != 0);
    const bool output0DtypeSupported =
        emitter != nullptr && haveOutput0 &&
        output0Dtype != VULKAN_DTYPE_NONE &&
        (emitter->dtypeSupport & output0Dtype) != 0;
    const bool operandTypesMatch =
        emitter != nullptr && inputs != nullptr &&
        validateOperandTypeContract(*emitter, inputs, numIn, caps);
    const bool zeroInputConstant =
        emitter != nullptr &&
        emitter->family == VulkanKernelFamily::CONSTANT_GENERATION && numIn == 0;
    const int primaryRank =
        zeroInputConstant
            ? (haveOutput0 ? outputs[0]->rankOf() : -1)
            : (haveInput0 ? inputs[0]->rankOf() : -1);
    const bool rankMatches =
        emitter != nullptr && primaryRank >= emitter->minimumRank &&
        (emitter->maximumRank < 0 || primaryRank <= emitter->maximumRank);
    const int unaryArguments =
        emitter != nullptr && emitter->family == VulkanKernelFamily::ELEMENTWISE_UNARY
            ? (unaryArgumentsMatch(
                   *emitter, slot, numIn, outputs, numOut)
                   ? 1
                   : 0)
            : -1;
    DSP_DIAG(
        GRAPH_REPLAY,
        "Vulkan recordability reject: descriptor=%s family=%d recipe=%d "
        "traits=0x%x emitterTraits=0x%x structuralMask=0x%x "
        "numIn=%d numOut=%d tArgs=%d iArgs=%d bArgs=%d dArgs=%d sArgs=%d "
        "explicitArgumentContract=%d argumentContractMatch=%d "
        "argumentValues=%d input0Dtype=%d input0Rank=%d "
        "output0Dtype=%d output0Rank=%d",
        slot.ident.opName.c_str(),
        emitter == nullptr ? -1 : static_cast<int>(emitter->family),
        emitter == nullptr ? -1 : static_cast<int>(emitter->recipe),
        emitter == nullptr ? 0u : emitter->traits,
        emitter == nullptr ? 0u : emitter->emitterTraits,
        emitter == nullptr
            ? 0u
            : static_cast<unsigned>(
                  emitter->operandTypeContract.structuralIndexInputMask),
        numIn, numOut, slot.args.numTArgs, slot.args.numIArgs,
        slot.args.numBArgs, slot.args.numDArgs, slot.args.numSArgs,
        hasExplicitArgumentContract ? 1 : 0, argumentsMatch ? 1 : 0,
        argumentValuesMatch ? 1 : 0,
        haveInput0 ? static_cast<int>(inputs[0]->dataType()) : -1,
        haveInput0 ? inputs[0]->rankOf() : -1,
        haveOutput0 ? static_cast<int>(outputs[0]->dataType()) : -1,
        haveOutput0 ? outputs[0]->rankOf() : -1);
    DSP_DIAG(
        GRAPH_REPLAY,
        "Vulkan recordability details: input0DtypeSupported=%d "
        "output0DtypeSupported=%d operandTypes=%d rank=%d unaryArguments=%d "
        "sameShape0Out=%d sameType0Out=%d sameShape01=%d sameType01=%d "
        "tArg0=%.17g tArg1=%.17g bArg0=%d dArg0=%d",
        input0DtypeSupported ? 1 : 0, output0DtypeSupported ? 1 : 0,
        operandTypesMatch ? 1 : 0, rankMatches ? 1 : 0, unaryArguments,
        haveInput0 && haveOutput0 && inputs[0]->isSameShape(outputs[0]) ? 1 : 0,
        haveInput0 && haveOutput0 &&
                inputs[0]->dataType() == outputs[0]->dataType()
            ? 1
            : 0,
        haveInput0 && haveInput1 && inputs[0]->isSameShape(inputs[1]) ? 1 : 0,
        haveInput0 && haveInput1 &&
                inputs[0]->dataType() == inputs[1]->dataType()
            ? 1
            : 0,
        slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 0.0,
        slot.args.numTArgs > 1 ? slot.args.tArgs[1] : 0.0,
        slot.args.numBArgs > 0 ? (slot.args.bArgs[0] ? 1 : 0) : -1,
        slot.args.numDArgs > 0 ? static_cast<int>(slot.args.dArgs[0]) : -1);
  }
  return recordable;
}

// ─────────────────────────────────────────────────────────────────────────────

//  emitMlirModule — textual MLIR per op

// ─────────────────────────────────────────────────────────────────────────────
//
// Generic lowerings carry the concrete descriptor hash in the integer
// attribute nd4j.op_hash; operation names are never semantic lowering input.
//
// The MLIR text is consumed by VulkanPipelineCache::getOrCompile() which runs
// the VulkanOpLoweringPass (populateVulkanLoweringPatterns) before SPIR-V
// conversion.

namespace {

static bool selectMlirFloatingContract(
    NDArray** inputs, int numIn, NDArray** outputs, int numOut,
    const VulkanDeviceCaps& caps,
    std::vector<std::string>& inputStorageTypes,
    std::vector<std::string>& outputStorageTypes,
    std::string& accumulatorType) {
  inputStorageTypes.assign(static_cast<size_t>(numIn), {});
  outputStorageTypes.assign(static_cast<size_t>(numOut), {});
  bool requiresF64Accumulator = false;
  auto select = [&](NDArray* array, std::string& storage) {
    if (array == nullptr) return false;
    std::string localAccumulator;
    if (!selectMlirFloatTypes(
            array->dataType(), caps, storage, localAccumulator)) {
      return false;
    }
    if (localAccumulator == "f64") {
      requiresF64Accumulator = true;
    } else if (localAccumulator != "f32") {
      return false;
    }
    return true;
  };
  for (int i = 0; i < numIn; ++i) {
    if (!select(inputs[i], inputStorageTypes[static_cast<size_t>(i)])) {
      return false;
    }
  }
  for (int i = 0; i < numOut; ++i) {
    if (!select(outputs[i], outputStorageTypes[static_cast<size_t>(i)])) {
      return false;
    }
  }
  accumulatorType = requiresF64Accumulator ? "f64" : "f32";
  return true;
}

static bool selectMlirMixedOperandContract(
    NDArray** inputs, int numIn, NDArray** outputs, int numOut,
    const VulkanDeviceCaps& caps, const VulkanKernelEmitterInfo& emitter,
    std::vector<std::string>& inputStorageTypes,
    std::vector<std::string>& outputStorageTypes,
    std::string& accumulatorType, uint16_t& unsignedInputMask) {
  const uint32_t scalarMask =
      emitter.operandTypeContract.scalar32InputMask;
  const uint32_t integer32Mask =
      emitter.operandTypeContract.integer32InputMask;
  const uint32_t integer64Mask =
      emitter.operandTypeContract.integer64InputMask;
  const uint32_t integerIndexMask =
      emitter.operandTypeContract.integerIndexInputMask;
  const uint32_t specialMask =
      scalarMask | integer32Mask | integer64Mask | integerIndexMask;
  const uint32_t overlappingRoles =
      (scalarMask & integer32Mask) | (scalarMask & integer64Mask) |
      (scalarMask & integerIndexMask) | (integer32Mask & integer64Mask) |
      (integer32Mask & integerIndexMask) |
      (integer64Mask & integerIndexMask);
  // Role masks describe optional operand positions too, so bits beyond numIn
  // are valid when the matching argument-contract alternative omits them.
  if (numIn < 0 || numIn > 16 || overlappingRoles != 0) {
    return false;
  }

  inputStorageTypes.assign(static_cast<size_t>(numIn), {});
  outputStorageTypes.assign(static_cast<size_t>(numOut), {});
  bool requiresF64Accumulator = false;
  auto selectFloat = [&](NDArray* array, std::string& storage) {
    std::string localAccumulator;
    if (array == nullptr ||
        !selectMlirFloatTypes(
            array->dataType(), caps, storage, localAccumulator)) {
      return false;
    }
    if (localAccumulator == "f64") {
      requiresF64Accumulator = true;
    } else if (localAccumulator != "f32") {
      return false;
    }
    return true;
  };

  for (int i = 0; i < numIn; ++i) {
    const uint32_t bit = uint32_t{1} << i;
    if ((specialMask & bit) == 0 &&
        !selectFloat(inputs[i], inputStorageTypes[static_cast<size_t>(i)])) {
      return false;
    }
  }
  for (int i = 0; i < numOut; ++i) {
    if (!selectFloat(outputs[i], outputStorageTypes[static_cast<size_t>(i)])) {
      return false;
    }
  }

  bool haveUniformSpecialType = false;
  std::string uniformSpecialType;
  bool uniformSpecialUnsigned = false;
  unsignedInputMask = 0;
  for (int i = 0; i < numIn; ++i) {
    const uint32_t bit = uint32_t{1} << i;
    if ((specialMask & bit) == 0) continue;
    std::string localAccumulator;
    bool localUnsigned = false;
    auto& storage = inputStorageTypes[static_cast<size_t>(i)];
    if (inputs[i] == nullptr ||
        !selectMlirScalarTypes(
            inputs[i]->dataType(), caps, storage, localAccumulator,
            localUnsigned) ||
        ((integer32Mask & bit) != 0 && localAccumulator != "i32") ||
        ((integer64Mask & bit) != 0 && localAccumulator != "i64") ||
        ((integerIndexMask & bit) != 0 && localAccumulator != "i32" &&
         localAccumulator != "i64") ||
        ((scalarMask & bit) != 0 && localAccumulator != "i32" &&
         localAccumulator != "f32" && localAccumulator != "f64")) {
      return false;
    }
    if (localAccumulator == "f64") requiresF64Accumulator = true;
    if (localUnsigned) unsignedInputMask |= uint16_t{1} << i;
    if (emitter.operandTypeContract.requireUniformSpecialInputs) {
      if (!haveUniformSpecialType) {
        haveUniformSpecialType = true;
        uniformSpecialType = storage;
        uniformSpecialUnsigned = localUnsigned;
      } else if (storage != uniformSpecialType ||
                 localUnsigned != uniformSpecialUnsigned) {
        return false;
      }
    }
  }
  accumulatorType = requiresF64Accumulator ? "f64" : "f32";
  return true;
}

static bool selectMlirIndexedFloatContract(
    NDArray** inputs, int numIn, NDArray** outputs, int numOut,
    const VulkanDeviceCaps& caps,
    std::vector<std::string>& inputStorageTypes,
    std::vector<std::string>& outputStorageTypes,
    std::string& accumulatorType, bool& dataInputUnsigned) {
  if (numIn != 2 || numOut != 1) return false;
  inputStorageTypes.assign(2, {});
  outputStorageTypes.assign(1, {});
  std::string dataAccumulator;
  if (!selectMlirScalarTypes(
          inputs[0]->dataType(), caps, inputStorageTypes[0],
          dataAccumulator, dataInputUnsigned)) {
    return false;
  }
  bool requiresF64Accumulator = dataAccumulator == "f64";
  auto selectFloat = [&](NDArray* array, std::string& storage) {
    std::string localAccumulator;
    if (array == nullptr ||
        !selectMlirFloatTypes(
            array->dataType(), caps, storage, localAccumulator)) {
      return false;
    }
    if (localAccumulator == "f64") {
      requiresF64Accumulator = true;
    } else if (localAccumulator != "f32") {
      return false;
    }
    return true;
  };
  if (!selectFloat(inputs[1], inputStorageTypes[1]) ||
      !selectFloat(outputs[0], outputStorageTypes[0])) {
    return false;
  }
  accumulatorType = requiresF64Accumulator ? "f64" : "f32";
  return true;
}

static std::string emitContractMovementMlir(
    const NativeSlot& slot, NDArray** inputs, int numIn,
    NDArray** outputs, int numOut, const VulkanDeviceCaps& caps) {
  const auto* emitter = emitterForSlot(slot);
  if (emitter == nullptr || numIn < 1 || inputs == nullptr ||
      numOut < 1 || outputs == nullptr || outputs[0] == nullptr) {
    return "";
  }

  auto isStructuralInput = [&](int index) {
    return index >= 0 &&
           vulkanInputIsStructuralIndex(
               *emitter, static_cast<unsigned>(index));
  };
  std::vector<std::string> inputTypes(static_cast<size_t>(numIn));
  std::vector<std::string> outputTypes(static_cast<size_t>(numOut));
  std::vector<std::string> inputMemrefs(static_cast<size_t>(numIn));
  std::vector<std::string> outputMemrefs(static_cast<size_t>(numOut));
  auto select = [&](NDArray* array, std::string& storage,
                    std::string& memref) {
    std::string accumulator;
    bool isUnsigned = false;
    if (array == nullptr ||
        !selectMlirScalarTypes(array->dataType(), caps, storage,
                               accumulator, isUnsigned)) {
      return false;
    }
    memref = mlirMemrefBody(array, storage);
    return !memref.empty();
  };
  for (int i = 0; i < numIn; ++i) {
    if (isStructuralInput(i)) {
      // This operand remains a real descriptor binding, but its values are not
      // part of the static replay contract. An inert i32 view avoids requiring
      // shaderInt64 solely because framework shape tensors commonly use LONG.
      inputTypes[static_cast<size_t>(i)] = "i32";
      inputMemrefs[static_cast<size_t>(i)] =
          mlirMemrefBody(inputs[i], inputTypes[static_cast<size_t>(i)]);
      if (inputMemrefs[static_cast<size_t>(i)].empty()) return "";
    } else if (!select(inputs[i], inputTypes[static_cast<size_t>(i)],
                       inputMemrefs[static_cast<size_t>(i)])) {
      return "";
    }
  }
  for (int i = 0; i < numOut; ++i) {
    if (!select(outputs[i], outputTypes[static_cast<size_t>(i)],
                outputMemrefs[static_cast<size_t>(i)])) {
      return "";
    }
  }

  const bool axisPartition = usesAxisPartitionSchedule(*emitter);
  const bool multiDestinationCopy =
      usesSameShapeCopySchedule(*emitter) &&
      !usesTrailingPayloadCopySchedule(*emitter) && numOut > 1;
  const bool appendAdditionalDestinations =
      axisPartition || multiDestinationCopy;
  std::vector<NDArray*> linalgInputs;
  std::vector<std::string> linalgInputTypes;
  std::vector<std::string> linalgInputMemrefs;
  linalgInputs.reserve(static_cast<size_t>(
      numIn + (appendAdditionalDestinations ? numOut - 1 : 0)));
  linalgInputTypes.reserve(linalgInputs.capacity());
  linalgInputMemrefs.reserve(linalgInputs.capacity());
  for (int i = 0; i < numIn; ++i) {
    linalgInputs.push_back(inputs[i]);
    linalgInputTypes.push_back(inputTypes[static_cast<size_t>(i)]);
    linalgInputMemrefs.push_back(inputMemrefs[static_cast<size_t>(i)]);
  }
  // linalg.generic requires destination shapes and element types to share a
  // loop domain. Additional destinations therefore remain output ABI operands
  // but are represented as inert generic inputs. The contract lowering
  // recognizes and writes all of them inside the same gpu.launch.
  if (appendAdditionalDestinations) {
    for (int i = 1; i < numOut; ++i) {
      linalgInputs.push_back(outputs[i]);
      linalgInputTypes.push_back(outputTypes[static_cast<size_t>(i)]);
      linalgInputMemrefs.push_back(outputMemrefs[static_cast<size_t>(i)]);
    }
  }

  const int loopRank = outputs[0]->rankOf();
  std::ostringstream dimensions;
  std::ostringstream identityResults;
  std::ostringstream iteratorTypes;
  for (int d = 0; d < loopRank; ++d) {
    if (d != 0) {
      dimensions << ", ";
      identityResults << ", ";
      iteratorTypes << ", ";
    }
    dimensions << "d" << d;
    identityResults << "d" << d;
    iteratorTypes << "\"parallel\"";
  }
  const std::string dims = dimensions.str();
  auto indexingMap = [&](int rank, bool identity) {
    std::ostringstream map;
    map << "affine_map<(" << dims << ") -> (";
    for (int d = 0; d < rank; ++d) {
      if (d != 0) map << ", ";
      map << (identity ? "d" + std::to_string(d) : "0");
    }
    map << ")>";
    return map.str();
  };

  const int primaryCopyInput =
      usesTrailingPayloadCopySchedule(*emitter) ? numIn - 1 : 0;
  bool inputFortran = false;
  bool outputFortran = false;
  if (emitter->loweringContract == VulkanLoweringContract::LINEAR_COPY) {
    inputFortran = inputs[primaryCopyInput]->ordering() == 'f';
    outputFortran = outputs[0]->ordering() == 'f';
  } else if (usesLinearConcatSchedule(*emitter)) {
    inputFortran = static_cast<char>(slot.args.iArgs[0]) == 'f';
  }
  const bool scalarExpand =
      emitter->loweringContract == VulkanLoweringContract::LINEAR_COPY &&
      !usesSameShapeCopySchedule(*emitter) &&
      inputs[primaryCopyInput]->lengthOf() == 1 &&
      outputs[0]->lengthOf() > 1;
  int64_t axis = -1;
  if (axisPartition) {
    axis = slot.args.iArgs[0];
    if (axis < 0) axis += inputs[0]->rankOf();
  }

  std::ostringstream ss;
  ss << "module {\n"
     << "  func.func @main(";
  bool first = true;
  for (int i = 0; i < numIn; ++i) {
    if (!first) ss << ", ";
    first = false;
    ss << "%input" << i << ": memref<"
       << inputMemrefs[static_cast<size_t>(i)] << ">";
  }
  for (int i = 0; i < numOut; ++i) {
    if (!first) ss << ", ";
    first = false;
    ss << "%output" << i << ": memref<"
       << outputMemrefs[static_cast<size_t>(i)] << ">";
  }
  ss << ") {\n"
     << "    linalg.generic {" << emitterIdentityAttributes(slot)
     << ", nd4j.contract_movement = true"
     << ", nd4j.input_fortran_order = "
     << (inputFortran ? "true" : "false")
     << ", nd4j.output_fortran_order = "
     << (outputFortran ? "true" : "false")
     << ", nd4j.scalar_expand = "
     << (scalarExpand ? "true" : "false")
     << ", nd4j.axis = " << axis << " : i64"
     << ", nd4j.num_payload_inputs = " << numIn << " : i64"
     << ", nd4j.num_outputs = " << numOut << " : i64,\n"
     << "                    indexing_maps = [";
  for (size_t i = 0; i < linalgInputs.size(); ++i) {
    if (i != 0) ss << ", ";
    ss << indexingMap(linalgInputs[i]->rankOf(), false);
  }
  if (!linalgInputs.empty()) ss << ", ";
  ss << indexingMap(loopRank, true) << "],\n"
     << "                    iterator_types = ["
     << iteratorTypes.str() << "]}\n"
     << "      ins(";
  for (size_t i = 0; i < linalgInputs.size(); ++i) {
    if (i != 0) ss << ", ";
    if (i < static_cast<size_t>(numIn)) {
      ss << "%input" << i;
    } else {
      ss << "%output" << (i - static_cast<size_t>(numIn) + 1);
    }
  }
  ss << " : ";
  for (size_t i = 0; i < linalgInputMemrefs.size(); ++i) {
    if (i != 0) ss << ", ";
    ss << "memref<" << linalgInputMemrefs[i] << ">";
  }
  ss << ")\n"
     << "      outs(%output0 : memref<" << outputMemrefs[0] << ">) {\n"
     << "      ^bb0(";
  for (size_t i = 0; i < linalgInputTypes.size(); ++i) {
    if (i != 0) ss << ", ";
    ss << "%arg" << i << ": " << linalgInputTypes[i];
  }
  if (!linalgInputTypes.empty()) ss << ", ";
  ss << "%outputValue: " << outputTypes[0] << "):\n"
     << "        linalg.yield %outputValue : " << outputTypes[0] << "\n"
     << "    }\n"
     << "    return\n"
     << "  }\n"
     << "}\n";
  return ss.str();
}

/**
 * Emit the ABI-preserving linalg shell for trait-routed non-elementwise
 * schedules. Every operand remains a real MemRef binding; the selected lowering
 * replaces this shell with exactly one gpu.launch.
 */
static std::string emitStructuredScheduleMlir(
    const NativeSlot& slot, NDArray** inputs, int numIn,
    NDArray** outputs, int numOut, const VulkanDeviceCaps& caps) {
  const auto* emitter = emitterForSlot(slot);
  if (emitter == nullptr || inputs == nullptr || outputs == nullptr ||
      numIn < 1 || numOut < 1 || outputs[0] == nullptr) {
    return "";
  }
  const bool batchedMatrixList =
      usesBatchedMatrixListSchedule(*emitter);
  const bool indexedAccumulation =
      usesIndexedAccumulationSchedule(*emitter);
  const bool indexedTadMovement =
      usesIndexedTadMovementSchedule(*emitter);
  if (!batchedMatrixList && !indexedAccumulation &&
      !indexedTadMovement) {
    return "";
  }

  auto isStructuralInput = [&](int index) {
    return vulkanInputIsStructuralIndex(*emitter,
                                        static_cast<unsigned>(index));
  };
  std::vector<std::string> inputTypes(static_cast<size_t>(numIn));
  std::vector<std::string> outputTypes(static_cast<size_t>(numOut));
  std::vector<std::string> inputMemrefs(static_cast<size_t>(numIn));
  std::vector<std::string> outputMemrefs(static_cast<size_t>(numOut));
  bool requiresF64Accumulator = false;
  bool hasFloatingAccumulator = false;
  auto select = [&](NDArray* array, std::string& storage,
                    std::string& memref) {
    std::string accumulator;
    bool isUnsigned = false;
    if (array == nullptr ||
        !selectMlirScalarTypes(array->dataType(), caps, storage,
                               accumulator, isUnsigned)) {
      return false;
    }
    requiresF64Accumulator |= accumulator == "f64";
    hasFloatingAccumulator |= accumulator == "f32" || accumulator == "f64";
    memref = mlirMemrefBody(array, storage);
    return !memref.empty();
  };
  for (int i = 0; i < numIn; ++i) {
    if (isStructuralInput(i)) {
      inputTypes[static_cast<size_t>(i)] = "i32";
      inputMemrefs[static_cast<size_t>(i)] =
          mlirMemrefBody(inputs[i], "i32");
      if (inputMemrefs[static_cast<size_t>(i)].empty()) return "";
    } else if (!select(inputs[i], inputTypes[static_cast<size_t>(i)],
                       inputMemrefs[static_cast<size_t>(i)])) {
      return "";
    }
  }
  for (int i = 0; i < numOut; ++i) {
    if (!select(outputs[i], outputTypes[static_cast<size_t>(i)],
                outputMemrefs[static_cast<size_t>(i)])) {
      return "";
    }
  }

  const bool appendAdditionalDestinations =
      indexedTadMovement && numOut > 1;
  std::vector<NDArray*> linalgInputs;
  std::vector<std::string> linalgInputTypes;
  std::vector<std::string> linalgInputMemrefs;
  linalgInputs.reserve(static_cast<size_t>(
      numIn + (appendAdditionalDestinations ? numOut - 1 : 0)));
  linalgInputTypes.reserve(linalgInputs.capacity());
  linalgInputMemrefs.reserve(linalgInputs.capacity());
  for (int i = 0; i < numIn; ++i) {
    linalgInputs.push_back(inputs[i]);
    linalgInputTypes.push_back(inputTypes[static_cast<size_t>(i)]);
    linalgInputMemrefs.push_back(inputMemrefs[static_cast<size_t>(i)]);
  }
  // Secondary destinations may have different TAD shapes. Keep them as
  // function ABI operands but model them as inert linalg inputs, then recover
  // and write them in the indexed-TAD lowering. This is the same shell contract
  // used by the generic multi-destination movement path.
  if (appendAdditionalDestinations) {
    for (int i = 1; i < numOut; ++i) {
      linalgInputs.push_back(outputs[i]);
      linalgInputTypes.push_back(outputTypes[static_cast<size_t>(i)]);
      linalgInputMemrefs.push_back(outputMemrefs[static_cast<size_t>(i)]);
    }
  }
  const int linalgOutputCount =
      appendAdditionalDestinations ? 1 : numOut;

  int loopRank = 0;
  if (!indexedAccumulation) {
    for (int i = 0; i < numIn; ++i) {
      loopRank = std::max(loopRank, inputs[i]->rankOf());
    }
  }
  // Indexed accumulation launches over the destination domain. Its indices may
  // have a higher rank (the trailing dimension stores index depth), but using
  // that rank for the inert linalg carrier leaves loop dimensions absent from
  // every indexing map and MLIR correctly rejects the non-invertible shell.
  for (int i = 0; i < numOut; ++i) {
    loopRank = std::max(loopRank, outputs[i]->rankOf());
  }
  std::ostringstream dimensions;
  std::ostringstream identityResults;
  std::ostringstream iteratorTypes;
  for (int d = 0; d < loopRank; ++d) {
    if (d != 0) {
      dimensions << ", ";
      identityResults << ", ";
      iteratorTypes << ", ";
    }
    dimensions << "d" << d;
    identityResults << "d" << d;
    iteratorTypes << "\"parallel\"";
  }
  const std::string dims = dimensions.str();
  auto indexingMap = [&](int rank, bool identity) {
    std::ostringstream map;
    map << "affine_map<(" << dims << ") -> (";
    for (int d = 0; d < rank; ++d) {
      if (d != 0) map << ", ";
      map << (identity ? "d" + std::to_string(d) : "0");
    }
    map << ")>";
    return map.str();
  };

  std::ostringstream ss;
  ss << "module {\n"
     << "  func.func @main(";
  bool first = true;
  for (int i = 0; i < numIn; ++i) {
    if (!first) ss << ", ";
    first = false;
    ss << "%input" << i << ": memref<"
       << inputMemrefs[static_cast<size_t>(i)] << ">";
  }
  for (int i = 0; i < numOut; ++i) {
    if (!first) ss << ", ";
    first = false;
    ss << "%output" << i << ": memref<"
       << outputMemrefs[static_cast<size_t>(i)] << ">";
  }
  ss << ") {\n"
     << "    linalg.generic {nd4j.op_hash = "
     << static_cast<long long>(slot.ident.opHash) << " : i64";
  if (hasFloatingAccumulator) {
    ss << ", nd4j.accumulator_type = "
       << (requiresF64Accumulator ? "f64" : "f32");
  }
  if (batchedMatrixList) {
    ss << ", nd4j.batched_matrix_list = true"
       << ", nd4j.transpose_a = "
       << (slot.args.iArgs[0] != 0 ? "true" : "false")
       << ", nd4j.transpose_b = "
       << (slot.args.iArgs[1] != 0 ? "true" : "false")
       << ", nd4j.batch_count = " << numOut << " : i64";
  } else if (indexedAccumulation) {
    ss << ", nd4j.indexed_accumulation = true"
       << ", nd4j.index_depth = "
       << inputs[0]->sizeAt(inputs[0]->rankOf() - 1) << " : i64"
       << ", nd4j.prefix_rank = " << (inputs[0]->rankOf() - 1)
       << " : i64";
  } else {
    ss << ", nd4j.indexed_tad_movement = true";
    switch (emitter->recipe) {
      case VulkanKernelRecipe::PULL_INDEXED_TADS:
        ss << ", nd4j.item_count = " << slot.args.iArgs[0] << " : i64"
           << ", nd4j.tad_dimension = " << slot.args.iArgs[1] << " : i64";
        break;
      case VulkanKernelRecipe::DISJOINT_PAIR_SHUFFLE: {
        const auto dimensionCount = slot.args.iArgs[1];
        ss << ", nd4j.array_count = " << slot.args.iArgs[0] << " : i64"
           << ", nd4j.tad_dimensions = array<i64: ";
        for (sd::LongType index = 0; index < dimensionCount; ++index) {
          if (index != 0) ss << ", ";
          ss << slot.args.iArgs[2 + index];
        }
        ss << ">";
        break;
      }
      default:
        return "";
    }
  }
  ss << ",\n"
     << "                    indexing_maps = [";
  for (size_t i = 0; i < linalgInputs.size(); ++i) {
    if (i != 0) ss << ", ";
    ss << indexingMap(linalgInputs[i]->rankOf(), false);
  }
  for (int i = 0; i < linalgOutputCount; ++i) {
    if (!linalgInputs.empty() || i != 0) ss << ", ";
    ss << indexingMap(outputs[i]->rankOf(), true);
  }
  ss << "],\n"
     << "                    iterator_types = ["
     << iteratorTypes.str() << "]}\n"
     << "      ins(";
  for (size_t i = 0; i < linalgInputs.size(); ++i) {
    if (i != 0) ss << ", ";
    if (i < static_cast<size_t>(numIn)) {
      ss << "%input" << i;
    } else {
      ss << "%output" << (i - static_cast<size_t>(numIn) + 1);
    }
  }
  ss << " : ";
  for (size_t i = 0; i < linalgInputMemrefs.size(); ++i) {
    if (i != 0) ss << ", ";
    ss << "memref<" << linalgInputMemrefs[i] << ">";
  }
  ss << ")\n"
     << "      outs(";
  for (int i = 0; i < linalgOutputCount; ++i) {
    if (i != 0) ss << ", ";
    ss << "%output" << i;
  }
  ss << " : ";
  for (int i = 0; i < linalgOutputCount; ++i) {
    if (i != 0) ss << ", ";
    ss << "memref<" << outputMemrefs[static_cast<size_t>(i)] << ">";
  }
  ss << ") {\n"
     << "      ^bb0(";
  for (size_t i = 0; i < linalgInputTypes.size(); ++i) {
    if (i != 0) ss << ", ";
    ss << "%inputValue" << i << ": " << linalgInputTypes[i];
  }
  for (int i = 0; i < linalgOutputCount; ++i) {
    if (!linalgInputTypes.empty() || i != 0) ss << ", ";
    ss << "%outputValue" << i << ": "
       << outputTypes[static_cast<size_t>(i)];
  }
  ss << "):\n"
     << "        linalg.yield ";
  for (int i = 0; i < linalgOutputCount; ++i) {
    if (i != 0) ss << ", ";
    ss << "%outputValue" << i;
  }
  ss << " : ";
  for (int i = 0; i < linalgOutputCount; ++i) {
    if (i != 0) ss << ", ";
    ss << outputTypes[static_cast<size_t>(i)];
  }
  ss << "\n"
     << "    }\n"
     << "    return\n"
     << "  }\n"
     << "}\n";
  return ss.str();
}

template <typename Policy>
static std::string emitVulkanOp(const NativeSlot& slot,
                                NDArray** inputs, int numIn,
                                NDArray** outputs, int numOut,
                                const VulkanDeviceCaps& caps) {
  if constexpr (Policy::contractMovement) {
    return emitContractMovementMlir(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if constexpr (Policy::batchedMatrixList ||
                Policy::indexedAccumulation ||
                Policy::indexedTadMovement) {
    return emitStructuredScheduleMlir(
        slot, inputs, numIn, outputs, numOut, caps);
  }

  std::string ts;
  std::string accTs;
  bool isUnsigned = false;
  const auto* scalarEmitter =
      emitterForSlot(slot);
  const bool structuralPrimaryInput =
      scalarEmitter != nullptr &&
      vulkanInputIsStructuralIndex(*scalarEmitter, 0);
  NDArray* scalarTypeDonor =
      numIn > 0 && inputs != nullptr && !structuralPrimaryInput
          ? inputs[0]
          : (numOut > 0 && outputs != nullptr ? outputs[0] : nullptr);
  if (scalarTypeDonor == nullptr ||
      !selectMlirScalarTypes(
          scalarTypeDonor->dataType(), caps, ts, accTs, isUnsigned)) {
    return "";
  }

  std::ostringstream ss;

  // ── reusable multi-output elementwise contract ────────────────────────────
  if constexpr (Policy::multiOutputElementwise) {
    std::vector<std::string> inputTypes(3);
    std::vector<std::string> inputAccumulators(3);
    std::vector<bool> inputUnsigned(3, false);
    std::vector<std::string> outputTypes(2);
    std::vector<std::string> outputAccumulators(2);
    std::vector<bool> outputUnsigned(2, false);
    bool requiresF64Accumulator = false;
    for (int i = 0; i < 3; ++i) {
      bool localUnsigned = false;
      if (!selectMlirScalarTypes(
              inputs[i]->dataType(), caps, inputTypes[static_cast<size_t>(i)],
              inputAccumulators[static_cast<size_t>(i)], localUnsigned)) {
        return "";
      }
      inputUnsigned[static_cast<size_t>(i)] = localUnsigned;
      requiresF64Accumulator |=
          inputAccumulators[static_cast<size_t>(i)] == "f64";
    }
    for (int i = 0; i < 2; ++i) {
      bool localUnsigned = false;
      if (!selectMlirScalarTypes(
              outputs[i]->dataType(), caps, outputTypes[static_cast<size_t>(i)],
              outputAccumulators[static_cast<size_t>(i)], localUnsigned) ||
          (outputAccumulators[static_cast<size_t>(i)] != "f32" &&
           outputAccumulators[static_cast<size_t>(i)] != "f64")) {
        return "";
      }
      outputUnsigned[static_cast<size_t>(i)] = localUnsigned;
      requiresF64Accumulator |=
          outputAccumulators[static_cast<size_t>(i)] == "f64";
    }
    accTs = requiresF64Accumulator ? "f64" : "f32";

    const int rank = outputs[0]->rankOf();
    std::ostringstream dimensions;
    std::ostringstream identityResults;
    std::string iteratorTypes;
    for (int d = 0; d < rank; ++d) {
      if (d != 0) {
        dimensions << ", ";
        identityResults << ", ";
        iteratorTypes += ", ";
      }
      dimensions << "d" << d;
      identityResults << "d" << d;
      iteratorTypes += "\"parallel\"";
    }
    std::ostringstream countResults;
    for (int d = 0; d < inputs[0]->rankOf(); ++d) {
      if (d != 0) countResults << ", ";
      countResults << "0";
    }
    const std::string dims = dimensions.str();
    const std::string identityMap =
        "affine_map<(" + dims + ") -> (" + identityResults.str() + ")>";
    const std::string countMap =
        "affine_map<(" + dims + ") -> (" + countResults.str() + ")>";
    const std::string countType = mlirMemrefBody(inputs[0], inputTypes[0]);
    const std::string meanType = mlirMemrefBody(inputs[1], inputTypes[1]);
    const std::string varianceType =
        mlirMemrefBody(inputs[2], inputTypes[2]);
    const std::string outputMeanType =
        mlirMemrefBody(outputs[0], outputTypes[0]);
    const std::string outputVarianceType =
        mlirMemrefBody(outputs[1], outputTypes[1]);
    const double shift =
        slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 0.0;

    ss << "module {\n"
       << "  func.func @main(%counts: memref<" << countType << ">, "
       << "%means: memref<" << meanType << ">, "
       << "%variances: memref<" << varianceType << ">, "
       << "%outputMeans: memref<" << outputMeanType << ">, "
       << "%outputVariances: memref<" << outputVarianceType << ">) {\n"
       << "    linalg.generic {nd4j.op_hash = "
       << static_cast<long long>(slot.ident.opHash)
       << " : i64, nd4j.accumulator_type = " << accTs
       << ", nd4j.multi_output_elementwise = true"
       << ", nd4j.input0_unsigned = "
       << (inputUnsigned[0] ? "true" : "false")
       << ", nd4j.input1_unsigned = "
       << (inputUnsigned[1] ? "true" : "false")
       << ", nd4j.input2_unsigned = "
       << (inputUnsigned[2] ? "true" : "false")
       << ", nd4j.shift = " << std::scientific
       << std::setprecision(std::numeric_limits<double>::max_digits10)
       << shift << " : " << accTs << ",\n"
       << "                    indexing_maps = [" << countMap << ", "
       << identityMap << ", " << identityMap << ", " << identityMap
       << ", " << identityMap << "],\n"
       << "                    iterator_types = [" << iteratorTypes << "]}\n"
       << "      ins(%counts, %means, %variances : memref<" << countType
       << ">, memref<" << meanType << ">, memref<" << varianceType << ">)\n"
       << "      outs(%outputMeans, %outputVariances : memref<"
       << outputMeanType << ">, memref<" << outputVarianceType << ">) {\n"
       << "      ^bb0(%count: " << inputTypes[0] << ", %mean: "
       << inputTypes[1] << ", %variance: " << inputTypes[2]
       << ", %outputMean: " << outputTypes[0] << ", %outputVariance: "
       << outputTypes[1] << "):\n"
       << "        linalg.yield %outputMean, %outputVariance : "
       << outputTypes[0] << ", " << outputTypes[1] << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── matmul / mmul ──────────────────────────────────────────────────────────
  if constexpr (Policy::matmul) {
    const std::string aType = mlirMemrefBody(inputs[0], ts);
    const std::string bType = mlirMemrefBody(inputs[1], ts);
    const std::string cType = mlirMemrefBody(outputs[0], ts);
    ss << "module {\n"
       << "  func.func @main(%A: memref<" << aType << ">, "
       <<                  "%B: memref<" << bType << ">, "
       <<                  "%C: memref<" << cType << ">) {\n"
       << "    linalg." << (inputs[0]->rankOf() == 3 ? "batch_matmul" : "matmul")
       << " {nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash)
       << " : i64, nd4j.accumulator_type = " << accTs
       << "} ins(%A, %B : memref<" << aType << ">, "
       <<                                       "memref<" << bType << ">) "
       <<                  "outs(%C : memref<" << cType << ">)\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── rms_norm / skip_rms_norm ───────────────────────────────────────────────
  if constexpr (Policy::rmsNorm) {
    sd::LongType rows   = inputs[0]->sizeAt(0);
    sd::LongType hidden = inputs[0]->sizeAt(1);
    bool hasGamma = (numIn >= 2 && inputs[1] != nullptr);
    const double eps = slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 1.0e-5;
    const std::string epsLiteral = mlirFloatLiteral(eps);
    std::vector<std::string> inputStorageTypes;
    std::vector<std::string> outputStorageTypes;
    if (!selectMlirFloatingContract(
            inputs, numIn, outputs, numOut, caps, inputStorageTypes,
            outputStorageTypes, accTs)) {
      return "";
    }
    const std::string& xTs = inputStorageTypes[0];
    const std::string& yTs = outputStorageTypes[0];
    const std::string xType = mlirMemrefBody(inputs[0], xTs);
    const std::string yType = mlirMemrefBody(outputs[0], yTs);
    const std::string gammaType =
        hasGamma ? mlirMemrefBody(inputs[1], inputStorageTypes[1]) : "";
    if (hasGamma) {
      ss << "module {\n"
         << "  func.func @main(%X: memref<" << xType << ">, "
         <<                  "%gamma: memref<" << gammaType << ">, "
         <<                  "%Y: memref<" << yType << ">) {\n"
         << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash) << " : i64, nd4j.accumulator_type = " << accTs
         << ", nd4j.epsilon = " << epsLiteral << " : " << accTs << ",\n"
         << "                    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d0, d1)>],\n"
         << "                    iterator_types = [\"parallel\", \"reduction\"]}\n"
         << "      ins(%X, %gamma : memref<" << xType << ">, "
         <<                        "memref<" << gammaType << ">)\n"
         << "      outs(%Y : memref<" << yType << ">) {\n"
         << "      ^bb0(%xv: " << xTs << ", %gv: "
         << inputStorageTypes[1] << ", %yv: " << yTs << "):\n"
         << "        linalg.yield %yv : " << yTs << "\n"
         << "    }\n"
         << "    return\n"
         << "  }\n"
         << "}\n";
    } else {
      ss << "module {\n"
         << "  func.func @main(%X: memref<" << xType << ">, "
         <<                  "%Y: memref<" << yType << ">) {\n"
         << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash) << " : i64, nd4j.accumulator_type = " << accTs
         << ", nd4j.epsilon = " << epsLiteral << " : " << accTs << ",\n"
         << "                    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d0, d1)>],\n"
         << "                    iterator_types = [\"parallel\", \"reduction\"]}\n"
         << "      ins(%X : memref<" << xType << ">)\n"
         << "      outs(%Y : memref<" << yType << ">) {\n"
         << "      ^bb0(%xv: " << xTs << ", %yv: " << yTs << "):\n"
         << "        linalg.yield %yv : " << yTs << "\n"
         << "    }\n"
         << "    return\n"
         << "  }\n"
         << "}\n";
    }
    return ss.str();
  }

  // ── rope / fused_rope ──────────────────────────────────────────────────────
  if constexpr (Policy::rope) {
    sd::LongType B     = inputs[0]->sizeAt(0);
    sd::LongType S     = inputs[0]->sizeAt(1);
    sd::LongType H     = inputs[0]->sizeAt(2);
    sd::LongType D     = inputs[0]->sizeAt(3);
    sd::LongType halfD = D / 2;
    const std::string xType = mlirMemrefBody(inputs[0], ts);
    const std::string cosType = mlirMemrefBody(inputs[1], ts);
    const std::string sinType = mlirMemrefBody(inputs[2], ts);
    const std::string yType = mlirMemrefBody(outputs[0], ts);
    const std::string cacheFeature =
        inputs[1]->sizeAt(1) == halfD ? "d3 floordiv 2" : "d3";
    ss << "module {\n"
       << "  func.func @main(%X: memref<" << xType << ">, "
       <<                  "%cos: memref<" << cosType << ">, "
       <<                  "%sin: memref<" << sinType << ">, "
       <<                  "%Y: memref<" << yType << ">) {\n"
       << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash)
       << " : i64, nd4j.accumulator_type = " << accTs
       << ", nd4j.head_dim = " << D << " : i64, "
       <<                    "nd4j.rotary_dim = " << halfD << " : i64,\n"
       << "                    indexing_maps = [\n"
       << "                      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,\n"
       << "                      affine_map<(d0, d1, d2, d3) -> (d1, "
       << cacheFeature << ")>,\n"
       << "                      affine_map<(d0, d1, d2, d3) -> (d1, "
       << cacheFeature << ")>,\n"
       << "                      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],\n"
       << "                    iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]}\n"
       << "      ins(%X, %cos, %sin : memref<" << xType << ">, "
       <<                            "memref<" << cosType << ">, "
       <<                            "memref<" << sinType << ">)\n"
       << "      outs(%Y : memref<" << yType << ">) {\n"
       << "      ^bb0(%xv: " << ts << ", %cv: " << ts << ", %sv: " << ts << ", %yv: " << ts << "):\n"
       << "        linalg.yield %yv : " << ts << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── elementwise ternary (select) ─────────────────────────────────────────
  if constexpr (Policy::ternary) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || numIn != 3 || numOut != 1) return "";

    std::string conditionTs, conditionAcc;
    std::string trueTs, trueAcc;
    std::string falseTs, falseAcc;
    std::string outputTs, outputAcc;
    bool conditionUnsigned = false;
    bool trueUnsigned = false;
    bool falseUnsigned = false;
    bool outputUnsigned = false;
    if (!selectMlirScalarTypes(inputs[0]->dataType(), caps, conditionTs,
                               conditionAcc, conditionUnsigned) ||
        !selectMlirScalarTypes(inputs[1]->dataType(), caps, trueTs, trueAcc,
                               trueUnsigned) ||
        !selectMlirScalarTypes(inputs[2]->dataType(), caps, falseTs, falseAcc,
                               falseUnsigned) ||
        !selectMlirScalarTypes(outputs[0]->dataType(), caps, outputTs,
                               outputAcc, outputUnsigned) ||
        trueAcc != falseAcc || outputAcc != trueAcc) {
      return "";
    }

    const int rank = outputs[0]->rankOf();
    std::ostringstream dimensions;
    std::string iteratorTypes;
    for (int d = 0; d < rank; ++d) {
      if (d != 0) {
        dimensions << ", ";
        iteratorTypes += ", ";
      }
      dimensions << "d" << d;
      iteratorTypes += "\"parallel\"";
    }
    const std::string dims = dimensions.str();
    auto broadcastMap = [&](NDArray* input) {
      std::ostringstream results;
      const int offset = rank - input->rankOf();
      for (int d = 0; d < input->rankOf(); ++d) {
        if (d != 0) results << ", ";
        results << (input->sizeAt(d) == 1
                        ? "0"
                        : "d" + std::to_string(offset + d));
      }
      return "affine_map<(" + dims + ") -> (" + results.str() + ")>";
    };
    const std::string conditionType = mlirMemrefBody(inputs[0], conditionTs);
    const std::string trueType = mlirMemrefBody(inputs[1], trueTs);
    const std::string falseType = mlirMemrefBody(inputs[2], falseTs);
    const std::string outputType = mlirMemrefBody(outputs[0], outputTs);
    const std::string outputMap =
        "affine_map<(" + dims + ") -> (" + dims + ")>";

    ss << "module {\n"
       << "  func.func @main(%condition: memref<" << conditionType << ">, "
       << "%true_value: memref<" << trueType << ">, "
       << "%false_value: memref<" << falseType << ">, "
       << "%output: memref<" << outputType << ">) {\n"
       << "    linalg.generic {" << emitterIdentityAttributes(slot)
       << ", nd4j.accumulator_type = " << trueAcc
       << ", nd4j.ternary = true, nd4j.input0_unsigned = "
       << (conditionUnsigned ? "true" : "false")
       << ", nd4j.input1_unsigned = "
       << (trueUnsigned ? "true" : "false")
       << ", nd4j.input2_unsigned = "
       << (falseUnsigned ? "true" : "false")
       << ", nd4j.output_unsigned = "
       << (outputUnsigned ? "true" : "false") << ",\n"
       << "                    indexing_maps = ["
       << broadcastMap(inputs[0]) << ", "
       << broadcastMap(inputs[1]) << ", "
       << broadcastMap(inputs[2]) << ", " << outputMap << "],\n"
       << "                    iterator_types = [" << iteratorTypes << "]}\n"
       << "      ins(%condition, %true_value, %false_value : memref<"
       << conditionType << ">, memref<" << trueType << ">, memref<"
       << falseType << ">)\n"
       << "      outs(%output : memref<" << outputType << ">) {\n"
       << "      ^bb0(%condition_value: " << conditionTs
       << ", %true_block: " << trueTs << ", %false_block: " << falseTs
       << ", %output_block: " << outputTs << "):\n"
       << "        linalg.yield %output_block : " << outputTs << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── elementwise binary ───────────────────────────────────────────────────
  if constexpr (Policy::binary) {
    const auto* emitter = emitterForSlot(slot);
    const bool unaryAssign =
        emitter != nullptr &&
        hasVulkanOpTrait(*emitter, sd::ops::OP_TRAIT_IDENTITY) &&
        numIn == 1;
    if (unaryAssign) {
      std::string aTs, aAcc, cTs, cAcc;
      bool aUnsigned = false;
      bool cUnsigned = false;
      if (!selectMlirScalarTypes(inputs[0]->dataType(), caps, aTs, aAcc,
                                 aUnsigned) ||
          !selectMlirScalarTypes(outputs[0]->dataType(), caps, cTs, cAcc,
                                 cUnsigned)) {
        return "";
      }
      const int rank = outputs[0]->rankOf();
      std::ostringstream dimensions;
      std::string iterTypes;
      for (int d = 0; d < rank; ++d) {
        if (d != 0) dimensions << ", ";
        dimensions << "d" << d;
        if (d != 0) iterTypes += ", ";
        iterTypes += "\"parallel\"";
      }
      const std::string dims = dimensions.str();
      const std::string identityMap =
          "affine_map<(" + dims + ") -> (" + dims + ")>";
      const std::string aType = mlirMemrefBody(inputs[0], aTs);
      const std::string cType = mlirMemrefBody(outputs[0], cTs);
      ss << "module {\n"
         << "  func.func @main(%A: memref<" << aType << ">, "
         << "%C: memref<" << cType << ">) {\n"
         << "    linalg.generic {" << emitterIdentityAttributes(slot)
         << ", nd4j.accumulator_type = " << cAcc
         << ", nd4j.binary = true, nd4j.input0_unsigned = "
         << (aUnsigned ? "true" : "false")
         << ", nd4j.output_unsigned = "
         << (cUnsigned ? "true" : "false") << ",\n"
         << "                    indexing_maps = [" << identityMap
         << ", " << identityMap << "],\n"
         << "                    iterator_types = [" << iterTypes << "]}\n"
         << "      ins(%A : memref<" << aType << ">)\n"
         << "      outs(%C : memref<" << cType << ">) {\n"
         << "      ^bb0(%av: " << aTs << ", %cv: " << cTs << "):\n"
         << "        linalg.yield %cv : " << cTs << "\n"
         << "    }\n"
         << "    return\n"
         << "  }\n"
         << "}\n";
      return ss.str();
    }

    const int rank = outputs[0]->rankOf();
    std::string aTs, bTs, cTs;
    std::string aAcc, bAcc, cAcc;
    bool aUnsigned = false;
    bool bUnsigned = false;
    bool cUnsigned = false;
    if (!selectMlirScalarTypes(inputs[0]->dataType(), caps, aTs, aAcc,
                               aUnsigned) ||
        !selectMlirScalarTypes(inputs[1]->dataType(), caps, bTs, bAcc,
                               bUnsigned) ||
        !selectMlirScalarTypes(outputs[0]->dataType(), caps, cTs, cAcc,
                               cUnsigned)) {
      return "";
    }
    if (emitter != nullptr &&
        (emitter->family == VulkanKernelFamily::COMPARISON ||
         emitter->family == VulkanKernelFamily::LOGICAL)) {
      // Comparisons/logical operations compute in the operand domain and
      // convert their i1/i32 result to the byte-addressed BOOL output ABI.
      cAcc = aAcc;
    }
    const std::string aType = mlirMemrefBody(inputs[0], aTs);
    const std::string bType = mlirMemrefBody(inputs[1], bTs);
    const std::string cType = mlirMemrefBody(outputs[0], cTs);
    std::ostringstream semanticAttributes;
    if (emitter != nullptr &&
        hasVulkanScalarArgumentSchema(*emitter)) {
      double parameter = 0.0;
      if (emitter->recipe == VulkanKernelRecipe::ELU_BP) {
        parameter = 1.0;
      } else if (emitter->recipe == VulkanKernelRecipe::LEAKY_RELU_BP) {
        parameter = 0.01;
      }
      if (slot.args.numTArgs == 1) {
        parameter = slot.args.tArgs[0];
        if (emitter->recipe !=
            VulkanKernelRecipe::THRESHOLDED_RELU_BP) {
          parameter = static_cast<float>(parameter);
        }
      }
      semanticAttributes << std::scientific
                         << std::setprecision(
                                std::numeric_limits<double>::max_digits10)
                         << ", nd4j.parameter = " << parameter
                         << " : " << cAcc;
    }

    auto dimList = [&]() {
      std::ostringstream text;
      for (int d = 0; d < rank; ++d) {
        if (d != 0) text << ", ";
        text << "d" << d;
      }
      return text.str();
    };
    const std::string dims = dimList();
    auto broadcastMap = [&](NDArray* input) {
      std::ostringstream results;
      const int offset = rank - input->rankOf();
      for (int d = 0; d < input->rankOf(); ++d) {
        if (d != 0) results << ", ";
        results << (input->sizeAt(d) == 1
                        ? "0"
                        : "d" + std::to_string(offset + d));
      }
      return "affine_map<(" + dims + ") -> (" + results.str() + ")>";
    };
    const std::string affineA = broadcastMap(inputs[0]);
    const std::string affineB = broadcastMap(inputs[1]);
    const std::string affineC =
        "affine_map<(" + dims + ") -> (" + dims + ")>";
    std::string iterTypes;
    for (int d = 0; d < rank; ++d) {
      if (d > 0) iterTypes += ", ";
      iterTypes += "\"parallel\"";
    }

    ss << "module {\n"
       << "  func.func @main(%A: memref<" << aType << ">, "
       <<                  "%B: memref<" << bType << ">, "
       <<                  "%C: memref<" << cType << ">) {\n"
       << "    linalg.generic {" << emitterIdentityAttributes(slot)
       << ", nd4j.accumulator_type = " << cAcc
       << ", nd4j.binary = true, nd4j.input0_unsigned = "
       << (aUnsigned ? "true" : "false")
       << ", nd4j.input1_unsigned = "
       << (bUnsigned ? "true" : "false")
       << ", nd4j.output_unsigned = "
       << (cUnsigned ? "true" : "false")
       << semanticAttributes.str() << ",\n"
       << "                    indexing_maps = [" << affineA << ", " << affineB << ", " << affineC << "],\n"
       << "                    iterator_types = [" << iterTypes << "]}\n"
       << "      ins(%A, %B : memref<" << aType << ">, memref<" << bType << ">)\n"
       << "      outs(%C : memref<" << cType << ">) {\n"
       << "      ^bb0(%av: " << aTs << ", %bv: " << bTs << ", %cv: " << cTs << "):\n"
       << "        linalg.yield %cv : " << cTs << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── unary elementwise ops ────────────────────────────────────────────────
  if constexpr (Policy::unary) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr) return "";
    int rank = inputs[0]->rankOf();
    std::string yTs;
    std::string yAccTs;
    bool outputUnsigned = false;
    if (!selectMlirScalarTypes(outputs[0]->dataType(), caps, yTs, yAccTs,
                               outputUnsigned)) {
      return "";
    }
    std::string unaryAccTs = accTs;
    if (unaryProducesFloatingOutput(*emitter)) {
      if (accTs == "f64" || yAccTs == "f64") {
        unaryAccTs = "f64";
      } else if (accTs == "f32" || yAccTs == "f32") {
        unaryAccTs = "f32";
      } else if (accTs == "f16" || yAccTs == "f16") {
        unaryAccTs = "f16";
      } else {
        return "";
      }
    }
    const std::string xType = mlirMemrefBody(inputs[0], ts);
    const std::string yType = mlirMemrefBody(outputs[0], yTs);

    if (vulkanArgumentContractAcceptsInputCount(*emitter, 3) &&
        numIn == 3) {
      std::string lowerTs, lowerAcc, upperTs, upperAcc;
      bool lowerUnsigned = false;
      bool upperUnsigned = false;
      if (!selectMlirScalarTypes(inputs[1]->dataType(), caps, lowerTs,
                                 lowerAcc, lowerUnsigned) ||
          !selectMlirScalarTypes(inputs[2]->dataType(), caps, upperTs,
                                 upperAcc, upperUnsigned)) {
        return "";
      }
      const std::string lowerType = mlirMemrefBody(inputs[1], lowerTs);
      const std::string upperType = mlirMemrefBody(inputs[2], upperTs);
      std::ostringstream dimensions;
      std::string iterTypes;
      for (int d = 0; d < rank; ++d) {
        if (d != 0) dimensions << ", ";
        dimensions << "d" << d;
        if (d != 0) iterTypes += ", ";
        iterTypes += "\"parallel\"";
      }
      const std::string dims = dimensions.str();
      const std::string identityMap =
          "affine_map<(" + dims + ") -> (" + dims + ")>";
      auto scalarMap = [&](NDArray* bound) {
        std::ostringstream zeros;
        // NDArray's shape accessors are historically non-const even though
        // querying the frozen view metadata does not mutate the array.
        const int boundRank = const_cast<NDArray*>(bound)->rankOf();
        for (int d = 0; d < boundRank; ++d) {
          if (d != 0) zeros << ", ";
          zeros << "0";
        }
        return "affine_map<(" + dims + ") -> (" + zeros.str() + ")>";
      };
      ss << "module {\n"
         << "  func.func @main(%X: memref<" << xType << ">, "
         << "%L: memref<" << lowerType << ">, "
         << "%U: memref<" << upperType << ">, "
         << "%Y: memref<" << yType << ">) {\n"
         << "    linalg.generic {nd4j.op_hash = "
         << static_cast<long long>(slot.ident.opHash)
         << " : i64, nd4j.accumulator_type = " << accTs
         << ", nd4j.unary = true, nd4j.bounds_from_inputs = true"
         << ", nd4j.input0_unsigned = "
         << (isUnsigned ? "true" : "false")
         << ", nd4j.input1_unsigned = "
         << (lowerUnsigned ? "true" : "false")
         << ", nd4j.input2_unsigned = "
         << (upperUnsigned ? "true" : "false")
         << ", nd4j.output_unsigned = "
         << (outputUnsigned ? "true" : "false") << ",\n"
         << "                    indexing_maps = [" << identityMap << ", "
         << scalarMap(inputs[1]) << ", " << scalarMap(inputs[2]) << ", "
         << identityMap << "],\n"
         << "                    iterator_types = [" << iterTypes << "]}\n"
         << "      ins(%X, %L, %U : memref<" << xType << ">, memref<"
         << lowerType << ">, memref<" << upperType << ">)\n"
         << "      outs(%Y : memref<" << yType << ">) {\n"
         << "      ^bb0(%xv: " << ts << ", %lv: " << lowerTs
         << ", %uv: " << upperTs << ", %yv: " << yTs << "):\n"
         << "        linalg.yield %yv : " << yTs << "\n"
         << "    }\n"
         << "    return\n"
         << "  }\n"
         << "}\n";
      return ss.str();
    }

    double scalar0 = 0.0;
    double scalar1 = 0.0;
    switch (emitter->recipe) {
      case VulkanKernelRecipe::LEAKY_RELU:
        scalar0 = slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 0.01;
        break;
      case VulkanKernelRecipe::ELU:
        scalar0 = slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 1.0;
        break;
      case VulkanKernelRecipe::SCALE:
        scalar0 = slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 1.0;
        break;
      case VulkanKernelRecipe::RELU:
      case VulkanKernelRecipe::THRESHOLDED_RELU:
      case VulkanKernelRecipe::RELU6:
        scalar0 = slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 0.0;
        break;
      case VulkanKernelRecipe::CLIP_BY_VALUE:
        scalar0 = slot.args.tArgs[0];
        scalar1 = slot.args.tArgs[1];
        break;
      default:
        break;
    }

    auto dimList = [&]() -> std::string {
      std::ostringstream s;
      for (int d = 0; d < rank; ++d) { if (d > 0) s << ", "; s << "d" << d; }
      return s.str();
    };
    std::string dims = dimList();
    std::string affineId = "affine_map<(" + dims + ") -> (" + dims + ")>";
    std::string iterTypes;
    for (int d = 0; d < rank; ++d) { if (d > 0) iterTypes += ", "; iterTypes += "\"parallel\""; }

    ss << "module {\n"
       << "  func.func @main(%X: memref<" << xType << ">, "
       <<                  "%Y: memref<" << yType << ">) {\n"
       << "    linalg.generic {" << emitterIdentityAttributes(slot)
       << ", nd4j.accumulator_type = " << unaryAccTs
       << ", nd4j.unary = true, nd4j.scalar0 = "
       << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10)
       << scalar0 << " : f64, nd4j.scalar1 = " << scalar1
       << " : f64, nd4j.input0_unsigned = "
       << (isUnsigned ? "true" : "false")
       << ", nd4j.output_unsigned = "
       << (outputUnsigned ? "true" : "false") << ",\n"
       << "                    indexing_maps = [" << affineId << ", " << affineId << "],\n"
       << "                    iterator_types = [" << iterTypes << "]}\n"
       << "      ins(%X : memref<" << xType << ">)\n"
       << "      outs(%Y : memref<" << yType << ">) {\n"
       << "      ^bb0(%xv: " << ts << ", %yv: " << yTs << "):\n"
       << "        linalg.yield %yv : " << yTs << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── softmax ───────────────────────────────────────────────────────────────
  if constexpr (Policy::softmax) {
    sd::LongType rows = inputs[0]->sizeAt(0);
    sd::LongType dim  = inputs[0]->sizeAt(1);
    const std::string xType = mlirMemrefBody(inputs[0], ts);
    const std::string yType = mlirMemrefBody(outputs[0], ts);

    ss << "module {\n"
       << "  func.func @main(%X: memref<" << xType << ">, "
       <<                  "%Y: memref<" << yType << ">) {\n"
       << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash)
       << " : i64, nd4j.accumulator_type = " << accTs << ",\n"
       << "                    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,\n"
       << "                                     affine_map<(d0, d1) -> (d0, d1)>],\n"
       << "                    iterator_types = [\"parallel\", \"reduction\"]}\n"
       << "      ins(%X : memref<" << xType << ">)\n"
       << "      outs(%Y : memref<" << yType << ">) {\n"
       << "      ^bb0(%xv: " << ts << ", %yv: " << ts << "):\n"
       << "        linalg.yield %yv : " << ts << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── layer_norm ────────────────────────────────────────────────────────────
  if constexpr (Policy::layerNorm) {
    sd::LongType rows   = inputs[0]->sizeAt(0);
    sd::LongType hidden = inputs[0]->sizeAt(1);
    bool hasGamma = (numIn >= 2 && inputs[1] != nullptr);
    bool hasBeta  = (numIn >= 3 && inputs[2] != nullptr);
    const auto* emitter = emitterForSlot(slot);
    const double eps =
        emitter != nullptr &&
                hasVulkanEmitterTrait(
                    *emitter, VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER) &&
                slot.args.numTArgs == 1
            ? slot.args.tArgs[0]
            : 1.0e-5;
    const std::string epsLiteral = mlirFloatLiteral(eps);
    std::vector<std::string> inputStorageTypes;
    std::vector<std::string> outputStorageTypes;
    if (!selectMlirFloatingContract(
            inputs, numIn, outputs, numOut, caps, inputStorageTypes,
            outputStorageTypes, accTs)) {
      return "";
    }
    const std::string& xTs = inputStorageTypes[0];
    const std::string& yTs = outputStorageTypes[0];
    const std::string xType = mlirMemrefBody(inputs[0], xTs);
    const std::string yType = mlirMemrefBody(outputs[0], yTs);
    const std::string gammaType =
        hasGamma ? mlirMemrefBody(inputs[1], inputStorageTypes[1]) : "";
    const std::string betaType =
        hasBeta ? mlirMemrefBody(inputs[2], inputStorageTypes[2]) : "";

    if (hasGamma && hasBeta) {
      ss << "module {\n"
         << "  func.func @main(%X: memref<" << xType << ">, "
         <<                  "%gamma: memref<" << gammaType << ">, "
         <<                  "%beta: memref<" << betaType << ">, "
         <<                  "%Y: memref<" << yType << ">) {\n"
         << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash) << " : i64, nd4j.accumulator_type = " << accTs
         << ", nd4j.epsilon = " << epsLiteral << " : " << accTs << ",\n"
         << "                    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d0, d1)>],\n"
         << "                    iterator_types = [\"parallel\", \"reduction\"]}\n"
         << "      ins(%X, %gamma, %beta : memref<" << xType << ">, "
         <<                               "memref<" << gammaType << ">, "
         <<                               "memref<" << betaType << ">)\n"
         << "      outs(%Y : memref<" << yType << ">) {\n"
         << "      ^bb0(%xv: " << xTs << ", %gv: "
         << inputStorageTypes[1] << ", %bv: " << inputStorageTypes[2]
         << ", %yv: " << yTs << "):\n"
         << "        linalg.yield %yv : " << yTs << "\n"
         << "    }\n"
         << "    return\n"
         << "  }\n"
         << "}\n";
    } else if (hasGamma) {
      ss << "module {\n"
         << "  func.func @main(%X: memref<" << xType << ">, "
         <<                  "%gamma: memref<" << gammaType << ">, "
         <<                  "%Y: memref<" << yType << ">) {\n"
         << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash) << " : i64, nd4j.accumulator_type = " << accTs
         << ", nd4j.epsilon = " << epsLiteral << " : " << accTs << ",\n"
         << "                    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d0, d1)>],\n"
         << "                    iterator_types = [\"parallel\", \"reduction\"]}\n"
         << "      ins(%X, %gamma : memref<" << xType << ">, "
         <<                         "memref<" << gammaType << ">)\n"
         << "      outs(%Y : memref<" << yType << ">) {\n"
         << "      ^bb0(%xv: " << xTs << ", %gv: "
         << inputStorageTypes[1] << ", %yv: " << yTs << "):\n"
         << "        linalg.yield %yv : " << yTs << "\n"
         << "    }\n"
         << "    return\n"
         << "  }\n"
         << "}\n";
    } else {
      ss << "module {\n"
         << "  func.func @main(%X: memref<" << xType << ">, "
         <<                  "%Y: memref<" << yType << ">) {\n"
         << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash) << " : i64, nd4j.accumulator_type = " << accTs
         << ", nd4j.epsilon = " << epsLiteral << " : " << accTs << ",\n"
         << "                    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,\n"
         << "                                     affine_map<(d0, d1) -> (d0, d1)>],\n"
         << "                    iterator_types = [\"parallel\", \"reduction\"]}\n"


         << "      ins(%X : memref<" << xType << ">)\n"
         << "      outs(%Y : memref<" << yType << ">) {\n"
         << "      ^bb0(%xv: " << xTs << ", %yv: " << yTs << "):\n"
         << "        linalg.yield %yv : " << yTs << "\n"
         << "    }\n"
         << "    return\n"
         << "  }\n"
         << "}\n";
    }
    return ss.str();
  }

  // ── gather / embedding_lookup (axis-0 rank-N table, vector indices) ─────
  if constexpr (Policy::gather) {
    std::string its;
    if (!selectMlirIndexType(inputs[1]->dataType(), its)) return "";
    const std::string& fty = ts;
    const std::string tableType = mlirMemrefBody(inputs[0], fty);
    const std::string indicesType = mlirMemrefBody(inputs[1], its);
    const std::string outputType = mlirMemrefBody(outputs[0], fty);

    std::ostringstream dimensions;
    std::ostringstream iterators;
    for (int d = 0; d < outputs[0]->rankOf(); ++d) {
      if (d != 0) {
        dimensions << ", ";
        iterators << ", ";
      }
      dimensions << "d" << d;
      iterators << "\"parallel\"";
    }
    const std::string dims = dimensions.str();
    std::ostringstream tableDimensions;
    for (int d = 0; d < inputs[0]->rankOf(); ++d) {
      if (d != 0) tableDimensions << ", ";
      tableDimensions << (d == 0 ? "0" : "d" + std::to_string(d));
    }
    const std::string tableMap =
        "affine_map<(" + dims + ") -> (" + tableDimensions.str() + ")>";
    const std::string indicesMap =
        "affine_map<(" + dims + ") -> (d0)>";
    const std::string outputMap =
        "affine_map<(" + dims + ") -> (" + dims + ")>";

    ss << "module {\n"
       << "  func.func @main("
       <<   "%table: memref<" << tableType << ">, "
       <<   "%indices: memref<" << indicesType << ">, "
       <<   "%out: memref<" << outputType << ">) {\n"
       << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash) << " : i64, nd4j.axis = 0 : i64,\n"
       << "                    nd4j.index_unsigned = "
       << (inputs[1]->dataType() == DataType::UINT32 ? "true" : "false")
       << ",\n                    indexing_maps = [" << tableMap
       << ", " << indicesMap << ", " << outputMap << "],\n"
       << "                    iterator_types = [" << iterators.str()
       << "]}\n"
       << "      ins(%table, %indices : memref<" << tableType << ">, "
       <<                              "memref<" << indicesType << ">)\n"
       << "      outs(%out : memref<" << outputType << ">) {\n"
       << "      ^bb0(%tv: " << fty << ", %iv: " << its << ", %ov: " << fty << "):\n"
       << "        linalg.yield %ov : " << fty << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── concat ────────────────────────────────────────────────────────────────
  if constexpr (Policy::concat) {
    int rank = inputs[0]->rankOf();
    const std::string& fty = ts;

    int64_t axis = -1;
    if (!normalizeAxis(slot.args.iArgs[0], rank, axis)) return "";

    auto buildShape = [&](NDArray* array) {
      return mlirMemrefBody(array, fty);
    };
    const std::string outShape = mlirMemrefBody(outputs[0], fty);

    auto dimList = [&]() -> std::string {
      std::ostringstream s;
      for (int d = 0; d < rank; ++d) { if (d > 0) s << ", "; s << "d" << d; }
      return s.str();
    };
    auto zeroList = [&]() -> std::string {
      std::ostringstream s;
      for (int d = 0; d < rank; ++d) { if (d > 0) s << ", "; s << "0"; }
      return s.str();
    };
    std::string dims = dimList();
    // The linalg.generic is an ABI carrier. Concat's trait-selected lowering
    // owns the real per-input index equations, so inputs use the same inert
    // zero maps as the shared descriptor-driven movement carrier.
    std::string affineInput =
        "affine_map<(" + dims + ") -> (" + zeroList() + ")>";
    std::string affineOutput =
        "affine_map<(" + dims + ") -> (" + dims + ")>";
    std::string iterTypes;
    for (int d = 0; d < rank; ++d) { if (d > 0) iterTypes += ", "; iterTypes += "\"parallel\""; }

    ss << "module {\n"
       << "  func.func @main(";
    for (int i = 0; i < numIn; ++i) {
      if (i > 0) ss << ", ";
      ss << "%in" << i << ": memref<" << buildShape(inputs[i]) << ">";
    }
    ss << ", %out: memref<" << outShape << ">) {\n"
       << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash) << " : i64, nd4j.axis = " << axis << " : i64,\n"

       << "                    nd4j.num_inputs = " << numIn << " : i64,\n"
       << "                    indexing_maps = [";

    for (int i = 0; i < numIn; ++i) {
      if (i > 0) ss << ", ";
      ss << affineInput;
    }

    ss << ", " << affineOutput << "],\n"
       << "                    iterator_types = [" << iterTypes << "]}\n"
       << "      ins(";
    for (int i = 0; i < numIn; ++i) {
      if (i > 0) ss << ", ";
      ss << "%in" << i;
    }
    ss << " : ";
    for (int i = 0; i < numIn; ++i) {
      if (i > 0) ss << ", ";
      ss << "memref<" << buildShape(inputs[i]) << ">";
    }
    ss << ")\n"
       << "      outs(%out : memref<" << outShape << ">) {\n"
       << "      ^bb0(";
    for (int i = 0; i < numIn; ++i) {
      if (i > 0) ss << ", ";
      ss << "%iv" << i << ": " << fty;
    }
    ss << ", %ov: " << fty << "):\n"
       << "        linalg.yield %ov : " << fty << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── transpose / permute ──────────────────────────────────────────────────
  if constexpr (Policy::transpose) {
    int rank = inputs[0]->rankOf();
    const std::string& fty = ts;
    std::vector<int64_t> perm;
    if (!permutationForSlot<Policy>(slot, rank, perm)) return "";

    // Build perm vector attribute string: [p0, p1, ...]
    auto buildShape = [&](NDArray* array) {
      return mlirMemrefBody(array, fty);
    };

    auto dimList = [&]() -> std::string {
      std::ostringstream s;
      for (int d = 0; d < rank; ++d) { if (d > 0) s << ", "; s << "d" << d; }
      return s.str();
    };
    std::string dims = dimList();
    // linalg.generic iterates the output domain.  Map those output indices back
    // to the input with the inverse permutation so MLIR can verify operands
    // whose dimensions are reordered.  The trait-selected lowering consumes
    // the same permutation attribute when it emits the device index equations.
    std::vector<int64_t> inversePermutation(static_cast<size_t>(rank));
    for (int d = 0; d < rank; ++d) {
      inversePermutation[static_cast<size_t>(perm[static_cast<size_t>(d)])] = d;
    }
    std::ostringstream inputDimensions;
    for (int d = 0; d < rank; ++d) {
      if (d > 0) inputDimensions << ", ";
      inputDimensions << "d"
                      << inversePermutation[static_cast<size_t>(d)];
    }
    std::string affineInput =
        "affine_map<(" + dims + ") -> (" + inputDimensions.str() + ")>";
    std::string affineOutput =
        "affine_map<(" + dims + ") -> (" + dims + ")>";
    std::string iterTypes;
    for (int d = 0; d < rank; ++d) { if (d > 0) iterTypes += ", "; iterTypes += "\"parallel\""; }

    std::string permStr;
    {
      std::ostringstream ps;
      ps << "array<i64";
      if (rank > 0) ps << ": ";
      for (int d = 0; d < rank; ++d) {
        if (d > 0) ps << ", ";
        ps << perm[d];
      }
      ps << ">";
      permStr = ps.str();
    }

    ss << "module {\n"
       << "  func.func @main("
       <<   "%X: memref<" << buildShape(inputs[0]) << ">, "
       <<   "%Y: memref<" << buildShape(outputs[0]) << ">) {\n"
       << "    linalg.generic {" << "nd4j.op_hash = " << static_cast<long long>(slot.ident.opHash) << " : i64,\n"

       << "                    nd4j.permutation = " << permStr << ",\n"
       << "                    indexing_maps = [" << affineInput << ", " << affineOutput << "],\n"
       << "                    iterator_types = [" << iterTypes << "]}\n"
       << "      ins(%X : memref<" << buildShape(inputs[0]) << ">)\n"
       << "      outs(%Y : memref<" << buildShape(outputs[0]) << ">) {\n"
       << "      ^bb0(%xv: " << fty << ", %yv: " << fty << "):\n"
       << "        linalg.yield %yv : " << fty << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  // ── Static equal split / unstack ─────────────────────────────────────────
  if constexpr (Policy::split) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || numIn != 1 || numOut < 1) return "";
    std::string outputTs;
    std::string outputAccTs;
    bool outputUnsigned = false;
    if (!selectMlirScalarTypes(outputs[0]->dataType(), caps, outputTs,
                               outputAccTs, outputUnsigned) ||
        outputTs != ts || outputUnsigned != isUnsigned) {
      return "";
    }
    for (int i = 1; i < numOut; ++i) {
      std::string localTs;
      std::string localAccTs;
      bool localUnsigned = false;
      if (!selectMlirScalarTypes(outputs[i]->dataType(), caps, localTs,
                                 localAccTs, localUnsigned) ||
          localTs != outputTs || localUnsigned != outputUnsigned) {
        return "";
      }
    }
    int64_t axis = 0;
    const bool removesAxis =
        outputs[0]->rankOf() + 1 == inputs[0]->rankOf();
    if (!removesAxis) {
      if (outputs[0]->rankOf() != inputs[0]->rankOf() ||
          (slot.args.numIArgs == 2 &&
           !normalizeAxis(slot.args.iArgs[1], inputs[0]->rankOf(), axis))) {
        return "";
      }
    } else if (!normalizeAxis(
                   slot.args.iArgs[0], inputs[0]->rankOf(), axis)) {
      return "";
    }
    const int loopRank = outputs[0]->rankOf();
    std::ostringstream dimensions;
    std::ostringstream iterators;
    for (int d = 0; d < loopRank; ++d) {
      if (d != 0) {
        dimensions << ", ";
        iterators << ", ";
      }
      dimensions << "d" << d;
      iterators << "\"parallel\"";
    }
    const std::string dims = dimensions.str();
    const std::string outputMap =
        "affine_map<(" + dims + ") -> (" + dims + ")>";
    std::ostringstream inputResults;
    for (int d = 0; d < inputs[0]->rankOf(); ++d) {
      if (d != 0) inputResults << ", ";
      inputResults << "0";
    }
    const std::string inputMap =
        "affine_map<(" + dims + ") -> (" + inputResults.str() + ")>";

    ss << "module {\n  func.func @main(%in: memref<"
       << mlirMemrefBody(inputs[0], ts) << ">";
    for (int i = 0; i < numOut; ++i) {
      ss << ", %out" << i << ": memref<"
         << mlirMemrefBody(outputs[i], outputTs) << ">";
    }
    ss << ") {\n    linalg.generic {nd4j.op_hash = "
       << static_cast<long long>(slot.ident.opHash)
       << " : i64, nd4j.axis = " << axis << " : i64"
       << ", nd4j.num_outputs = " << numOut << " : i64,\n"
       << "                    indexing_maps = [" << inputMap;
    for (int i = 0; i < numOut; ++i) ss << ", " << outputMap;
    ss << "],\n                    iterator_types = [" << iterators.str()
       << "]}\n      ins(%in : memref<" << mlirMemrefBody(inputs[0], ts)
       << ">)\n      outs(";
    for (int i = 0; i < numOut; ++i) {
      if (i != 0) ss << ", ";
      ss << "%out" << i;
    }
    ss << " : ";
    for (int i = 0; i < numOut; ++i) {
      if (i != 0) ss << ", ";
      ss << "memref<" << mlirMemrefBody(outputs[i], outputTs) << ">";
    }
    ss << ") {\n      ^bb0(%iv: " << ts;
    for (int i = 0; i < numOut; ++i) {
      ss << ", %ov" << i << ": " << outputTs;
    }
    ss << "):\n        linalg.yield ";
    for (int i = 0; i < numOut; ++i) {
      if (i != 0) ss << ", ";
      ss << "%ov" << i;
    }
    ss << " : ";
    for (int i = 0; i < numOut; ++i) {
      if (i != 0) ss << ", ";
      ss << outputTs;
    }
    ss << "\n    }\n    return\n  }\n}\n";
    return ss.str();
  }

  // ── Replay-safe constant generation ─────────────────────────────────────
  if constexpr (Policy::constant) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || numOut != 1) return "";
    std::string outputTs;
    std::string outputAccTs;
    bool outputUnsigned = false;
    if (!selectMlirScalarTypes(outputs[0]->dataType(), caps, outputTs,
                               outputAccTs, outputUnsigned)) {
      return "";
    }
    const int loopRank = outputs[0]->rankOf();
    std::ostringstream dimensions;
    std::ostringstream iterators;
    for (int d = 0; d < loopRank; ++d) {
      if (d != 0) {
        dimensions << ", ";
        iterators << ", ";
      }
      dimensions << "d" << d;
      iterators << "\"parallel\"";
    }
    const std::string dims = dimensions.str();
    const std::string identity =
        "affine_map<(" + dims + ") -> (" + dims + ")>";
    if (hasVulkanEmitterTrait(
            *emitter, VULKAN_EMITTER_TRAIT_RANDOM_STATE)) {
      if (numIn != 0 || slot.args.numTArgs != 2) return "";
      const std::string stateMap =
          "affine_map<(" + dims + ") -> (0)>";
      std::ostringstream attributes;
      attributes << ", nd4j.accumulator_type = " << outputAccTs
                 << ", nd4j.random_from = "
                 << mlirFloatLiteral(slot.args.tArgs[0]) << " : " << outputAccTs
                 << ", nd4j.random_to = "
                 << mlirFloatLiteral(slot.args.tArgs[1]) << " : " << outputAccTs;
      ss << "module {\n  func.func @main(%rng: memref<"
         << kVulkanRandomStateWordCount << "xi32>, %out: memref<"
         << mlirMemrefBody(outputs[0], outputTs) << ">) {\n"
         << "    linalg.generic {" << emitterIdentityAttributes(slot)
         << attributes.str() << ",\n"
         << "                    indexing_maps = [" << stateMap << ", "
         << identity << "],\n                    iterator_types = ["
         << iterators.str() << "]}\n"
         << "      ins(%rng : memref<" << kVulkanRandomStateWordCount
         << "xi32>)\n"
         << "      outs(%out : memref<"
         << mlirMemrefBody(outputs[0], outputTs) << ">) {\n"
         << "      ^bb0(%rngWord: i32, %ov: " << outputTs << "):\n"
         << "        linalg.yield %ov : " << outputTs
         << "\n    }\n    return\n  }\n}\n";
      return ss.str();
    }
    if (hasVulkanEmitterTrait(
            *emitter, VULKAN_EMITTER_TRAIT_ARGUMENT_GENERATED)) {
      if (numIn != 0) return "";
      std::ostringstream attributes;
      if (emitter->recipe == VulkanKernelRecipe::EYE) {
        attributes << ", nd4j.eye = true, nd4j.output_unsigned = "
                   << (outputUnsigned ? "true" : "false");
      } else if (emitter->recipe ==
                 VulkanKernelRecipe::MIN_MAX_DATATYPE) {
        double value = 0.0;
        if (!supportedScalarLimit(
                outputs[0]->dataType(), slot.args.iArgs[1] != 0, value)) {
          return "";
        }
        std::ostringstream literal;
        literal << std::scientific
                << std::setprecision(
                       std::numeric_limits<double>::max_digits10)
                << value;
        attributes << ", nd4j.constant_value = " << literal.str()
                   << " : f64, nd4j.output_unsigned = "
                   << (outputUnsigned ? "true" : "false");
      } else if (emitter->recipe == VulkanKernelRecipe::LIN_SPACE) {
        const sd::LongType steps = slot.args.iArgs[0];
        const double start = slot.args.tArgs[0];
        const bool endSpecified =
            slot.args.numBArgs == 1 && slot.args.bArgs[0];
        const double step =
            steps == 1
                ? 0.0
                : (endSpecified
                       ? (slot.args.tArgs[1] - start) /
                             (static_cast<double>(steps) - 1.0)
                       : slot.args.tArgs[1]);
        const std::string computeType =
            DataTypeUtils::isR(outputs[0]->dataType()) ? outputAccTs : "f64";
        if (computeType == "f64" && !caps.fp64) return "";
        auto floatLiteral = [](double value) {
          std::ostringstream literal;
          literal << std::scientific
                  << std::setprecision(
                         std::numeric_limits<double>::max_digits10)
                  << value;
          return literal.str();
        };
        attributes << ", nd4j.accumulator_type = " << computeType
                   << ", nd4j.linspace_start = " << floatLiteral(start)
                   << " : " << computeType
                   << ", nd4j.linspace_step = " << floatLiteral(step)
                   << " : " << computeType
                   << ", nd4j.output_unsigned = "
                   << (outputUnsigned ? "true" : "false");
      } else {
        StaticRangeSpec range;
        if (!readStaticRangeSpec(slot, range)) return "";
        auto floatLiteral = [](double value) {
          std::ostringstream literal;
          literal << std::scientific
                  << std::setprecision(
                         std::numeric_limits<double>::max_digits10)
                  << value;
          return literal.str();
        };
        if (DataTypeUtils::isR(outputs[0]->dataType())) {
          attributes << ", nd4j.accumulator_type = " << outputAccTs;
        }
        attributes << ", nd4j.range_start = "
                   << floatLiteral(range.valueStart) << " : f64"
                   << ", nd4j.range_delta = "
                   << floatLiteral(range.valueDelta) << " : f64"
                   << ", nd4j.output_unsigned = "
                   << (outputUnsigned ? "true" : "false");
      }
      ss << "module {\n  func.func @main(%out: memref<"
         << mlirMemrefBody(outputs[0], outputTs) << ">) {\n"
         << "    linalg.generic {" << emitterIdentityAttributes(slot)
         << attributes.str() << ",\n"
         << "                    indexing_maps = [" << identity
         << "],\n                    iterator_types = ["
         << iterators.str() << "]}\n"
         << "      outs(%out : memref<"
         << mlirMemrefBody(outputs[0], outputTs) << ">) {\n"
         << "      ^bb0(%ov: " << outputTs << "):\n"
         << "        linalg.yield %ov : " << outputTs
         << "\n    }\n    return\n  }\n}\n";
      return ss.str();
    }
    if (numIn != 1) return "";
    const bool structuralInput =
        vulkanInputIsStructuralIndex(*emitter, 0);
    const std::string inputTs = structuralInput ? "i32" : ts;
    auto inputMap = [&](const std::vector<std::string>& results) {
      std::ostringstream map;
      map << "affine_map<(" << dims << ") -> (";
      for (size_t i = 0; i < results.size(); ++i) {
        if (i != 0) map << ", ";
        map << results[i];
      }
      map << ")>";
      return map.str();
    };
    std::vector<std::string> inputResults;
    std::ostringstream semanticAttributes;
    switch (emitter->recipe) {
      case VulkanKernelRecipe::FILL_AS: {
        for (int d = 0; d < inputs[0]->rankOf(); ++d) {
          inputResults.push_back("d" + std::to_string(d));
        }
        const double value =
            slot.args.numIArgs == 1
                ? static_cast<double>(slot.args.iArgs[0])
                : slot.args.tArgs[0];
        std::ostringstream literal;
        literal << std::scientific
                << std::setprecision(
                       std::numeric_limits<double>::max_digits10)
                << value;
        semanticAttributes
            << ", nd4j.constant_value = " << literal.str() << " : f64"
            << ", nd4j.output_unsigned = "
            << (outputUnsigned ? "true" : "false");
        break;
      }
      case VulkanKernelRecipe::ZEROS_AS:
      case VulkanKernelRecipe::ONES_AS:
        if (structuralInput) {
          inputResults.assign(
              static_cast<size_t>(inputs[0]->rankOf()), "0");
        } else {
          for (int d = 0; d < inputs[0]->rankOf(); ++d) {
            inputResults.push_back("d" + std::to_string(d));
          }
        }
        semanticAttributes
            << ", nd4j.constant_fill = true, nd4j.fill_one = "
            << (emitter->recipe == VulkanKernelRecipe::ONES_AS
                    ? "true"
                    : "false");
        break;
      case VulkanKernelRecipe::SHAPE_OF: {
        inputResults.assign(static_cast<size_t>(inputs[0]->rankOf()), "0");
        semanticAttributes << ", nd4j.shape_values = array<i64: ";
        for (int d = 0; d < inputs[0]->rankOf(); ++d) {
          if (d != 0) semanticAttributes << ", ";
          semanticAttributes << inputs[0]->sizeAt(d);
        }
        semanticAttributes << ">, nd4j.output_unsigned = "
                           << (outputUnsigned ? "true" : "false");
        break;
      }
      case VulkanKernelRecipe::RANK_OF:
      case VulkanKernelRecipe::SIZE_OF:
      case VulkanKernelRecipe::SIZE_AT: {
        inputResults.assign(static_cast<size_t>(inputs[0]->rankOf()), "0");
        sd::LongType value = 0;
        if (emitter->recipe == VulkanKernelRecipe::SIZE_AT) {
          int64_t axis = -1;
          if (!normalizeAxis(slot.args.iArgs[0], inputs[0]->rankOf(), axis)) {
            return "";
          }
          value = inputs[0]->sizeAt(static_cast<int>(axis));
        } else {
          value = emitter->recipe == VulkanKernelRecipe::RANK_OF
                      ? static_cast<sd::LongType>(inputs[0]->rankOf())
                      : inputs[0]->lengthOf();
        }
        semanticAttributes << ", nd4j.scalar_metadata = " << value
                           << " : i64, nd4j.output_unsigned = "
                           << (outputUnsigned ? "true" : "false");
        break;
      }
      case VulkanKernelRecipe::ONE_HOT: {
        int64_t axis = -1;
        if (!normalizeAxis(slot.args.iArgs[0], loopRank, axis)) return "";
        inputResults.reserve(static_cast<size_t>(inputs[0]->rankOf()));
        for (int d = 0; d < loopRank; ++d) {
          if (d != axis) {
            inputResults.push_back("d" + std::to_string(d));
          }
        }
        const double on =
            slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 1.0;
        const double off =
            slot.args.numTArgs > 1 ? slot.args.tArgs[1] : 0.0;
        std::ostringstream onValue;
        std::ostringstream offValue;
        onValue << std::scientific
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << on;
        offValue << std::scientific
                 << std::setprecision(std::numeric_limits<double>::max_digits10)
                 << off;
        semanticAttributes
            << ", nd4j.axis = " << axis << " : i64"
            << ", nd4j.depth = " << slot.args.iArgs[1] << " : i64"
            << ", nd4j.on_value = " << onValue.str() << " : f64"
            << ", nd4j.off_value = " << offValue.str() << " : f64"
            << ", nd4j.index_unsigned = "
            << (isUnsigned ? "true" : "false")
            << ", nd4j.output_unsigned = "
            << (outputUnsigned ? "true" : "false");
        break;
      }
      default:
        return "";
    }
    ss << "module {\n  func.func @main(%in: memref<"
       << mlirMemrefBody(inputs[0], inputTs) << ">, %out: memref<"
       << mlirMemrefBody(outputs[0], outputTs) << ">) {\n"
       << "    linalg.generic {nd4j.op_hash = "
       << static_cast<long long>(slot.ident.opHash)
       << " : i64" << semanticAttributes.str() << ",\n"
       << "                    indexing_maps = [" << inputMap(inputResults)
       << ", "
       << identity << "],\n                    iterator_types = ["
       << iterators.str() << "]}\n"
       << "      ins(%in : memref<" << mlirMemrefBody(inputs[0], inputTs)
       << ">)\n      outs(%out : memref<"
       << mlirMemrefBody(outputs[0], outputTs) << ">) {\n"
       << "      ^bb0(%iv: " << inputTs << ", %ov: " << outputTs << "):\n"
       << "        linalg.yield %ov : " << outputTs
       << "\n    }\n    return\n  }\n}\n";
    return ss.str();
  }

  // ── Static rank-N data movement ─────────────────────────────────────────
  if constexpr (Policy::movement) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr || numOut != 1) return "";

    std::vector<std::string> inputStorageTypes(static_cast<size_t>(numIn));
    std::vector<bool> inputUnsigned(static_cast<size_t>(numIn), false);
    for (int i = 0; i < numIn; ++i) {
      std::string localAccumulator;
      bool localUnsigned = false;
      if (!selectMlirScalarTypes(
              inputs[i]->dataType(), caps,
              inputStorageTypes[static_cast<size_t>(i)], localAccumulator,
              localUnsigned)) {
        return "";
      }
      inputUnsigned[static_cast<size_t>(i)] = localUnsigned;
    }
    std::string outputStorageType;
    std::string outputAccumulator;
    bool outputUnsigned = false;
    if (!selectMlirScalarTypes(outputs[0]->dataType(), caps,
                               outputStorageType, outputAccumulator,
                               outputUnsigned) ||
        inputStorageTypes[0] != outputStorageType ||
        inputUnsigned[0] != outputUnsigned) {
      return "";
    }
    if (!hasVulkanOpTrait(*emitter, sd::ops::OP_TRAIT_GATHER_ND)) {
      for (int i = 1; i < numIn; ++i) {
        const auto inputIndex = static_cast<unsigned>(i);
        const bool hasSpecialStorageRole =
            vulkanInputUsesScalar32Storage(*emitter, inputIndex) ||
            vulkanInputUsesInteger32Storage(*emitter, inputIndex) ||
            vulkanInputUsesInteger64Storage(*emitter, inputIndex) ||
            vulkanInputUsesIntegerIndexStorage(*emitter, inputIndex) ||
            vulkanInputIsStructuralIndex(*emitter, inputIndex);
        if (!hasSpecialStorageRole &&
            (inputStorageTypes[static_cast<size_t>(i)] != outputStorageType ||
             inputUnsigned[static_cast<size_t>(i)] != outputUnsigned)) {
          return "";
        }
      }
    }

    const int loopRank = outputs[0]->rankOf();
    auto dimensionList = [](int rank) {
      std::ostringstream result;
      for (int d = 0; d < rank; ++d) {
        if (d != 0) result << ", ";
        result << "d" << d;
      }
      return result.str();
    };
    const std::string loopDimensions = dimensionList(loopRank);
    auto affineMap = [&](const std::vector<std::string>& results) {
      std::ostringstream map;
      map << "affine_map<(" << loopDimensions << ") -> (";
      for (size_t i = 0; i < results.size(); ++i) {
        if (i != 0) map << ", ";
        map << results[i];
      }
      map << ")>";
      return map.str();
    };
    auto identityResults = [](int rank) {
      std::vector<std::string> results;
      results.reserve(static_cast<size_t>(rank));
      for (int d = 0; d < rank; ++d) {
        results.push_back("d" + std::to_string(d));
      }
      return results;
    };
    auto integerArray = [](const std::vector<sd::LongType>& values) {
      std::ostringstream attr;
      attr << "array<i64";
      if (!values.empty()) attr << ": ";
      for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) attr << ", ";
        attr << values[i];
      }
      attr << ">";
      return attr.str();
    };

    std::vector<std::string> maps;
    maps.reserve(static_cast<size_t>(numIn + 1));
    std::ostringstream semanticAttributes;
    switch (emitter->recipe) {
      case VulkanKernelRecipe::GATHER_ND: {
        const int inputRank = inputs[0]->rankOf();
        const int indicesRank = inputs[1]->rankOf();
        const int indexedRank =
            static_cast<int>(inputs[1]->sizeAt(indicesRank - 1));
        std::vector<std::string> inputMap;
        for (int d = 0; d < inputRank; ++d) {
          inputMap.push_back(
              d < indexedRank
                  ? "0"
                  : "d" + std::to_string(
                                indicesRank - 1 + d - indexedRank));
        }
        std::vector<std::string> indicesMap;
        for (int d = 0; d < indicesRank; ++d) {
          indicesMap.push_back(
              d == indicesRank - 1 ? "0" : "d" + std::to_string(d));
        }
        maps.push_back(affineMap(inputMap));
        maps.push_back(affineMap(indicesMap));
        semanticAttributes << ", nd4j.index_unsigned = "
                           << (inputUnsigned[1] ? "true" : "false");
        break;
      }
      case VulkanKernelRecipe::TILE: {
        maps.push_back(affineMap(std::vector<std::string>(
            static_cast<size_t>(inputs[0]->rankOf()), "0")));
        std::vector<sd::LongType> repetitions(
            slot.args.iArgs, slot.args.iArgs + slot.args.numIArgs);
        semanticAttributes << ", nd4j.repetitions = "
                           << integerArray(repetitions);
        break;
      }
      case VulkanKernelRecipe::REPEAT: {
        maps.push_back(affineMap(std::vector<std::string>(
            static_cast<size_t>(inputs[0]->rankOf()), "0")));
        int64_t axis = -1;
        if (!normalizeAxis(slot.args.iArgs[slot.args.numIArgs - 1],
                           inputs[0]->rankOf(), axis)) {
          return "";
        }
        std::vector<sd::LongType> repetitions(
            slot.args.iArgs,
            slot.args.iArgs + slot.args.numIArgs - 1);
        semanticAttributes << ", nd4j.axis = " << axis << " : i64"
                           << ", nd4j.repetitions = "
                           << integerArray(repetitions);
        break;
      }
      case VulkanKernelRecipe::REVERSE: {
        maps.push_back(affineMap(identityResults(inputs[0]->rankOf())));
        std::set<sd::LongType> normalizedAxes;
        for (int i = 0; i < slot.args.numIArgs; ++i) {
          int64_t axis = -1;
          if (!normalizeAxis(slot.args.iArgs[i], inputs[0]->rankOf(), axis)) {
            return "";
          }
          normalizedAxes.insert(axis);
        }
        std::vector<sd::LongType> axes(
            normalizedAxes.begin(), normalizedAxes.end());
        semanticAttributes << ", nd4j.reverse_axes = "
                           << integerArray(axes);
        break;
      }
      case VulkanKernelRecipe::ROLL: {
        maps.push_back(affineMap(identityResults(inputs[0]->rankOf())));
        semanticAttributes
            << ", nd4j.roll_input_fortran = "
            << (inputs[0]->ordering() == 'f' ? "true" : "false")
            << ", nd4j.roll_output_fortran = "
            << (outputs[0]->ordering() == 'f' ? "true" : "false");
        if (numIn == 1) {
          if (slot.args.numIArgs == 1) {
            sd::LongType shift = 0;
            if (!normalizeRollShift(slot.args.iArgs[0],
                                    inputs[0]->lengthOf(), shift)) {
              return "";
            }
            semanticAttributes << ", nd4j.roll_linear_shift = "
                               << shift << " : i64";
          } else {
            std::vector<sd::LongType> dimensionShifts(
                static_cast<size_t>(inputs[0]->rankOf()), 0);
            for (int i = 1; i < slot.args.numIArgs; ++i) {
              int64_t axis = -1;
              if (!normalizeAxis(slot.args.iArgs[i], inputs[0]->rankOf(),
                                 axis)) {
                return "";
              }
              const sd::LongType dimension =
                  inputs[0]->sizeAt(static_cast<int>(axis));
              sd::LongType shift = 0;
              if (!normalizeRollShift(slot.args.iArgs[0], dimension, shift)) {
                return "";
              }
              const size_t index = static_cast<size_t>(axis);
              const uint64_t combined =
                  static_cast<uint64_t>(dimensionShifts[index]) +
                  static_cast<uint64_t>(shift);
              dimensionShifts[index] = static_cast<sd::LongType>(
                  combined % static_cast<uint64_t>(dimension));
            }
            semanticAttributes << ", nd4j.roll_dimension_shifts = "
                               << integerArray(dimensionShifts);
          }
        } else {
          for (int i = 1; i < numIn; ++i) {
            maps.push_back(affineMap(std::vector<std::string>(
                static_cast<size_t>(inputs[i]->rankOf()), "0")));
          }
          semanticAttributes
              << ", nd4j.roll_tensor_controls = true"
              << ", nd4j.roll_has_axes = "
              << (numIn == 3 ? "true" : "false")
              << ", nd4j.roll_shift_unsigned = "
              << (inputUnsigned[1] ? "true" : "false")
              << ", nd4j.roll_shift_fortran = "
              << (inputs[1]->ordering() == 'f' ? "true" : "false");
          if (numIn == 3) {
            semanticAttributes
                << ", nd4j.roll_axes_unsigned = "
                << (inputUnsigned[2] ? "true" : "false")
                << ", nd4j.roll_axes_fortran = "
                << (inputs[2]->ordering() == 'f' ? "true" : "false");
          }
        }
        break;
      }
      case VulkanKernelRecipe::SLICE: {
        std::vector<std::string> inputMap(
            static_cast<size_t>(inputs[0]->rankOf()), "0");
        maps.push_back(affineMap(inputMap));
        std::vector<sd::LongType> begin;
        std::vector<sd::LongType> sizes;
        begin.reserve(static_cast<size_t>(inputs[0]->rankOf()));
        sizes.reserve(static_cast<size_t>(inputs[0]->rankOf()));
        for (int d = 0; d < inputs[0]->rankOf(); ++d) {
          begin.push_back(slot.args.iArgs[d]);
          sd::LongType size = slot.args.iArgs[inputs[0]->rankOf() + d];
          if (size == -1) size = inputs[0]->sizeAt(d) - begin.back();
          sizes.push_back(size);
        }
        semanticAttributes << ", nd4j.slice_begin = "
                           << integerArray(begin)
                           << ", nd4j.slice_sizes = "
                           << integerArray(sizes);
        break;
      }
      case VulkanKernelRecipe::STRIDED_SLICE: {
        maps.push_back(affineMap(std::vector<std::string>(
            static_cast<size_t>(inputs[0]->rankOf()), "0")));
        const int rank = inputs[0]->rankOf();
        std::vector<sd::LongType> begin(
            slot.args.iArgs + 5, slot.args.iArgs + 5 + rank);
        std::vector<sd::LongType> end(
            slot.args.iArgs + 5 + rank,
            slot.args.iArgs + 5 + 2 * rank);
        std::vector<sd::LongType> strides(
            slot.args.iArgs + 5 + 2 * rank,
            slot.args.iArgs + 5 + 3 * rank);
        semanticAttributes << ", nd4j.slice_begin = "
                           << integerArray(begin)
                           << ", nd4j.slice_end = "
                           << integerArray(end)
                           << ", nd4j.slice_strides = "
                           << integerArray(strides);
        break;
      }
      case VulkanKernelRecipe::STACK: {
        int64_t axis = 0;
        if (slot.args.numIArgs == 1 &&
            !normalizeAxis(slot.args.iArgs[0], inputs[0]->rankOf() + 1,
                           axis)) {
          return "";
        }
        for (int i = 0; i < numIn; ++i) {
          std::vector<std::string> inputMap;
          if (inputs[i]->rankOf() == 1 && inputs[0]->rankOf() == 0) {
            inputMap.push_back("0");
          } else {
            for (int d = 0; d < inputs[i]->rankOf(); ++d) {
              inputMap.push_back(
                  "d" + std::to_string(d < axis ? d : d + 1));
            }
          }
          maps.push_back(affineMap(inputMap));
        }
        semanticAttributes << ", nd4j.axis = " << axis << " : i64"
                           << ", nd4j.num_inputs = " << numIn << " : i64";
        break;
      }
      case VulkanKernelRecipe::TRIU: {
        maps.push_back(affineMap(identityResults(inputs[0]->rankOf())));
        const sd::LongType diagonal =
            slot.args.numIArgs == 1 ? slot.args.iArgs[0] : 0;
        semanticAttributes << ", nd4j.diagonal = " << diagonal << " : i64";
        break;
      }
      case VulkanKernelRecipe::TRIU_BP: {
        maps.push_back(affineMap(identityResults(inputs[0]->rankOf())));
        maps.push_back(affineMap(identityResults(inputs[1]->rankOf())));
        const sd::LongType diagonal =
            slot.args.numIArgs == 1 ? slot.args.iArgs[0] : 0;
        semanticAttributes << ", nd4j.diagonal = " << diagonal << " : i64";
        break;
      }
      default:
        return "";
    }
    maps.push_back(affineMap(identityResults(loopRank)));
    if (maps.size() != static_cast<size_t>(numIn + 1)) return "";

    ss << "module {\n  func.func @main(";
    for (int i = 0; i < numIn; ++i) {
      if (i != 0) ss << ", ";
      ss << "%in" << i << ": memref<"
         << mlirMemrefBody(
                inputs[i], inputStorageTypes[static_cast<size_t>(i)])
         << ">";
    }
    if (numIn != 0) ss << ", ";
    ss << "%out: memref<" << mlirMemrefBody(outputs[0], outputStorageType)
       << ">) {\n    linalg.generic {nd4j.op_hash = "
       << static_cast<long long>(slot.ident.opHash) << " : i64"
       << semanticAttributes.str() << ",\n"
       << "                    indexing_maps = [";
    for (size_t i = 0; i < maps.size(); ++i) {
      if (i != 0) ss << ", ";
      ss << maps[i];
    }
    ss << "],\n                    iterator_types = [";
    for (int d = 0; d < loopRank; ++d) {
      if (d != 0) ss << ", ";
      ss << "\"parallel\"";
    }
    ss << "]}\n      ins(";
    for (int i = 0; i < numIn; ++i) {
      if (i != 0) ss << ", ";
      ss << "%in" << i;
    }
    ss << " : ";
    for (int i = 0; i < numIn; ++i) {
      if (i != 0) ss << ", ";
      ss << "memref<"
         << mlirMemrefBody(
                inputs[i], inputStorageTypes[static_cast<size_t>(i)])
         << ">";
    }
    ss << ")\n      outs(%out : memref<"
       << mlirMemrefBody(outputs[0], outputStorageType)
       << ">) {\n      ^bb0(";
    for (int i = 0; i < numIn; ++i) {
      if (i != 0) ss << ", ";
      ss << "%iv" << i << ": "
         << inputStorageTypes[static_cast<size_t>(i)];
    }
    if (numIn != 0) ss << ", ";
    ss << "%ov: " << outputStorageType << "):\n"
       << "        linalg.yield %ov : " << outputStorageType << "\n"
       << "    }\n    return\n  }\n}\n";
    return ss.str();
  }

  // ── Structured compute kernels ────────────────────────────────────────────────────
  if constexpr (Policy::structuredCompute) {
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr) return "";

    std::vector<std::string> inputStorageTypes;
    std::vector<std::string> outputStorageTypes;
    uint16_t unsignedInputMask = 0;
    bool positionUnsigned = false;
    bool tokenUnsigned = false;
    bool dataInputUnsigned = false;
    bool typeContractSelected = false;
    if (hasVulkanMixedOperandTypeContract(*emitter)) {
      typeContractSelected = selectMlirMixedOperandContract(
          inputs, numIn, outputs, numOut, caps, *emitter,
          inputStorageTypes, outputStorageTypes, accTs, unsignedInputMask);
      positionUnsigned = (unsignedInputMask & (uint16_t{1} << 1)) != 0;
      tokenUnsigned = (unsignedInputMask & (uint16_t{1} << 2)) != 0;
    } else if (usesBroadcastBinarySchedule(*emitter)) {
      typeContractSelected = selectMlirIndexedFloatContract(
          inputs, numIn, outputs, numOut, caps, inputStorageTypes,
          outputStorageTypes, accTs, dataInputUnsigned);
    } else {
      typeContractSelected = selectMlirFloatingContract(
          inputs, numIn, outputs, numOut, caps, inputStorageTypes,
          outputStorageTypes, accTs);
    }
    if (!typeContractSelected) {
      return "";
    }

    std::vector<std::string> maps;
    std::vector<std::string> iterators;
    switch (emitter->recipe) {
      case VulkanKernelRecipe::WINDOW_PARTITION: {
        const sd::LongType window = slot.args.iArgs[0];
        const sd::LongType heightBlocks = inputs[0]->sizeAt(1) / window;
        const sd::LongType widthBlocks = inputs[0]->sizeAt(2) / window;
        const sd::LongType windowsPerBatch = heightBlocks * widthBlocks;
        const std::string dims = "d0, d1, d2, d3";
        maps = {
            "affine_map<(" + dims + ") -> (d0 floordiv " +
                std::to_string(windowsPerBatch) +
                ", ((d0 mod " + std::to_string(windowsPerBatch) +
                ") floordiv " + std::to_string(widthBlocks) + ") * " +
                std::to_string(window) + " + d1, (d0 mod " +
                std::to_string(widthBlocks) + ") * " +
                std::to_string(window) + " + d2, d3)>",
            "affine_map<(" + dims + ") -> (d0, d1, d2, d3)>"};
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"parallel\"", "\"parallel\""};
        break;
      }
      case VulkanKernelRecipe::WINDOW_UNPARTITION: {
        const sd::LongType window = slot.args.iArgs[0];
        const sd::LongType heightBlocks = slot.args.iArgs[1] / window;
        const sd::LongType widthBlocks = slot.args.iArgs[2] / window;
        const std::string dims = "d0, d1, d2, d3";
        maps = {
            "affine_map<(" + dims + ") -> ((d0 * " +
                std::to_string(heightBlocks) + " + d1 floordiv " +
                std::to_string(window) + ") * " +
                std::to_string(widthBlocks) + " + d2 floordiv " +
                std::to_string(window) + ", d1 mod " +
                std::to_string(window) + ", d2 mod " +
                std::to_string(window) + ", d3)>",
            "affine_map<(" + dims + ") -> (d0, d1, d2, d3)>"};
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"parallel\"", "\"parallel\""};
        break;
      }
      case VulkanKernelRecipe::BIAS_ADD: {
        const int rank = outputs[0]->rankOf();
        const bool nchw =
            slot.args.numBArgs == 1 && slot.args.bArgs[0];
        const int channelAxis = nchw ? 1 : rank - 1;
        std::ostringstream dimensions;
        for (int d = 0; d < rank; ++d) {
          if (d != 0) dimensions << ", ";
          dimensions << "d" << d;
          iterators.push_back("\"parallel\"");
        }
        const std::string dims = dimensions.str();
        const std::string identity =
            "affine_map<(" + dims + ") -> (" + dims + ")>";
        maps = {
            identity,
            "affine_map<(" + dims + ") -> (d" +
                std::to_string(channelAxis) + ")>",
            identity};
        break;
      }
      case VulkanKernelRecipe::PRELU: {
        const int rank = outputs[0]->rankOf();
        const int alphaRank = inputs[1]->rankOf();
        std::ostringstream allDimensions;
        std::ostringstream dataDimensions;
        std::ostringstream alphaDimensions;
        for (int d = 0; d < rank + alphaRank; ++d) {
          if (d != 0) allDimensions << ", ";
          allDimensions << "d" << d;
          iterators.push_back(
              d < rank ? "\"parallel\"" : "\"reduction\"");
          if (d < rank) {
            if (d != 0) dataDimensions << ", ";
            dataDimensions << "d" << d;
          } else {
            if (d != rank) alphaDimensions << ", ";
            alphaDimensions << "d" << d;
          }
        }
        const std::string all = allDimensions.str();
        const std::string dataMap =
            "affine_map<(" + all + ") -> (" +
            dataDimensions.str() + ")>";
        const std::string alphaMap =
            "affine_map<(" + all + ") -> (" +
            alphaDimensions.str() + ")>";
        maps = {dataMap, alphaMap, dataMap};
        break;
      }
      case VulkanKernelRecipe::BATCH_NORM: {
        const int rank = outputs[0]->rankOf();
        std::vector<int64_t> axes;
        if (slot.args.numIArgs == 2) {
          axes.push_back(rank - 1);
        } else {
          for (int i = 2; i < slot.args.numIArgs; ++i) {
            int64_t axis = -1;
            if (!normalizeAxis(slot.args.iArgs[i], rank, axis)) return "";
            axes.push_back(axis);
          }
        }
        std::set<int64_t> axisSet(axes.begin(), axes.end());
        std::ostringstream dimensions;
        for (int d = 0; d < rank; ++d) {
          if (d != 0) dimensions << ", ";
          dimensions << "d" << d;
          iterators.push_back("\"parallel\"");
        }
        const std::string dims = dimensions.str();
        const std::string identity =
            "affine_map<(" + dims + ") -> (" + dims + ")>";
        std::ostringstream parameterResult;
        if (axes.size() == 1) {
          parameterResult << "d" << axes[0];
        } else {
          for (int d = 0; d < rank; ++d) {
            if (d != 0) parameterResult << ", ";
            parameterResult <<
                (axisSet.count(d) != 0 ? "d" + std::to_string(d) : "0");
          }
        }
        const std::string parameterMap =
            "affine_map<(" + dims + ") -> (" +
            parameterResult.str() + ")>";
        maps.push_back(identity);
        for (int i = 1; i < numIn; ++i) maps.push_back(parameterMap);
        maps.push_back(identity);
        break;
      }
      case VulkanKernelRecipe::RMS_NORM_BP: {
        const int rank = outputs[0]->rankOf();
        std::ostringstream dimensions;
        for (int d = 0; d < rank; ++d) {
          if (d != 0) dimensions << ", ";
          dimensions << "d" << d;
          iterators.push_back("\"parallel\"");
        }
        const std::string dims = dimensions.str();
        const std::string identity =
            "affine_map<(" + dims + ") -> (" + dims + ")>";
        maps.assign(static_cast<size_t>(numIn + numOut), identity);
        break;
      }
      case VulkanKernelRecipe::FUSED_LAYER_NORM_BP: {
        const std::string identity =
            "affine_map<(d0) -> (d0)>";
        maps.assign(static_cast<size_t>(numIn + numOut), identity);
        iterators = {"\"parallel\""};
        break;
      }
      case VulkanKernelRecipe::VISION_EMBEDDING_MERGE:
        maps = {
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>"};
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"parallel\"", "\"reduction\""};
        break;
      case VulkanKernelRecipe::APPLY_ALIBI:
        maps = {
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>"};
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"parallel\"", "\"parallel\""};
        break;
      case VulkanKernelRecipe::ROPE:
      case VulkanKernelRecipe::ROPE_BP: {
        const int rank = outputs[0]->rankOf();
        std::ostringstream dimensions;
        for (int d = 0; d < rank; ++d) {
          if (d != 0) dimensions << ", ";
          dimensions << "d" << d;
          iterators.push_back("\"parallel\"");
        }
        const std::string dims = dimensions.str();
        const std::string identity =
            "affine_map<(" + dims + ") -> (" + dims + ")>";
        maps.assign(static_cast<size_t>(numIn + numOut), identity);
        break;
      }
      case VulkanKernelRecipe::FUSED_MROPE:
        maps = {
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>"};
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"parallel\"", "\"parallel\""};
        break;
      case VulkanKernelRecipe::SKIP_RMS_NORM: {
        const std::string identity =
            "affine_map<(d0, d1) -> (d0, d1)>";
        const std::string feature =
            "affine_map<(d0, d1) -> (d1)>";
        maps = {identity, identity, feature};
        if (numIn == 4) maps.push_back(feature);
        for (int i = 0; i < numOut; ++i) maps.push_back(identity);
        iterators = {"\"parallel\"", "\"reduction\""};
        break;
      }
      case VulkanKernelRecipe::RMS_NORM_LINEAR:
        maps = {
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d2)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>"};
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"reduction\""};
        break;
      case VulkanKernelRecipe::FUSED_GEMM_SWIGLU:
        maps = {
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>"};
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"reduction\""};
        break;
      case VulkanKernelRecipe::FUSED_RMS_NORM_SWIGLU:
        maps = {
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>",
            "affine_map<(d0, d1, d2, d3) -> (d3)>",
            "affine_map<(d0, d1, d2, d3) -> (d3, d2)>",
            "affine_map<(d0, d1, d2, d3) -> (d3, d2)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>"};
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"parallel\"", "\"reduction\""};
        break;
      case VulkanKernelRecipe::DOT_PRODUCT_ATTENTION:
        if (inputs[0]->rankOf() == 3) {
          maps = {
              "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>",
              "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>",
              "affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2)>"};
          if (numIn == 4) {
            maps.push_back(
                "affine_map<(d0, d1, d2, d3, d4) -> (d0, d2)>");
          }
          maps.push_back(
              "affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>");
          if (numOut == 2) {
            maps.push_back(
                "affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3)>");
          }
          iterators = {"\"parallel\"", "\"reduction\"",
                       "\"reduction\"", "\"parallel\"",
                       "\"parallel\""};
        } else {
          maps = {
              "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>",
              "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>",
              "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>"};
          if (numIn == 4) {
            maps.push_back(
                "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3)>");
          }
          maps.push_back(
              "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>");
          if (numOut == 2) {
            maps.push_back(
                "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>");
          }
          iterators = {"\"parallel\"", "\"parallel\"",
                       "\"reduction\"", "\"reduction\"",
                       "\"parallel\"", "\"parallel\""};
        }
        break;
      case VulkanKernelRecipe::FLASH_ATTENTION:
      case VulkanKernelRecipe::GROUPED_QUERY_ATTENTION:
        if (inputs[0]->rankOf() == 3) {
          maps = {
              "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>",
              "affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>",
              "affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>",
              "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>"};
          iterators = {"\"parallel\"", "\"parallel\"",
                       "\"reduction\"", "\"parallel\""};
        } else {
          const sd::LongType headsPerGroup =
              inputs[0]->sizeAt(2) / inputs[1]->sizeAt(2);
          const std::string kvHead =
              headsPerGroup == 1
                  ? "d2"
                  : "d2 floordiv " + std::to_string(headsPerGroup);
          maps = {
              "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>",
              "affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, " +
                  kvHead + ", d4)>",
              "affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, " +
                  kvHead + ", d4)>",
              "affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>"};
          iterators = {"\"parallel\"", "\"parallel\"",
                       "\"parallel\"", "\"reduction\"",
                       "\"parallel\""};
        }
        break;
      case VulkanKernelRecipe::FUSED_ELEMENTWISE_CHAIN: {
        const int rank = outputs[0]->rankOf();
        std::ostringstream dimensions;
        for (int d = 0; d < rank; ++d) {
          if (d != 0) dimensions << ", ";
          dimensions << "d" << d;
          iterators.push_back("\"parallel\"");
        }
        const std::string dims = dimensions.str();
        const std::string identity =
            "affine_map<(" + dims + ") -> (" + dims + ")>";
        maps.assign(static_cast<size_t>(numIn + numOut), identity);
        break;
      }
      case VulkanKernelRecipe::SWISH_MUL_BP: {
        const int rank = outputs[0]->rankOf();
        std::ostringstream dimensions;
        for (int d = 0; d < rank; ++d) {
          if (d != 0) dimensions << ", ";
          dimensions << "d" << d;
          iterators.push_back("\"parallel\"");
        }
        const std::string dims = dimensions.str();
        const std::string identity =
            "affine_map<(" + dims + ") -> (" + dims + ")>";
        maps.assign(static_cast<size_t>(numIn + numOut), identity);
        break;
      }
      case VulkanKernelRecipe::FUSED_BIAS_DROPOUT_RESIDUAL: {
        const int rank = outputs[0]->rankOf();
        std::ostringstream dimensions;
        for (int d = 0; d < rank; ++d) {
          if (d != 0) dimensions << ", ";
          dimensions << "d" << d;
          iterators.push_back("\"parallel\"");
        }
        const std::string dims = dimensions.str();
        const std::string identity =
            "affine_map<(" + dims + ") -> (" + dims + ")>";
        const std::string feature =
            "affine_map<(" + dims + ") -> (d" +
            std::to_string(rank - 1) + ")>";
        maps = {identity, feature, identity, identity};
        break;
      }
      case VulkanKernelRecipe::FUSED_ATTENTION_PROJECTION:
        if (inputs[0]->rankOf() == 3) {
          maps = {
              "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>",
              "affine_map<(d0, d1, d2, d3) -> (d3, d2)>"};
        } else {
          const sd::LongType headDim = inputs[0]->sizeAt(3);
          maps = {
              "affine_map<(d0, d1, d2, d3) -> (d0, d1, "
                  "d3 floordiv " + std::to_string(headDim) +
                  ", d3 mod " + std::to_string(headDim) + ")>",
              "affine_map<(d0, d1, d2, d3) -> (d3, d2)>"};
        }
        if (numIn == 3) {
          maps.push_back("affine_map<(d0, d1, d2, d3) -> (d2)>");
        }
        maps.push_back(
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>");
        iterators = {"\"parallel\"", "\"parallel\"",
                     "\"parallel\"", "\"reduction\""};
        break;
      case VulkanKernelRecipe::SWIGLU:
      case VulkanKernelRecipe::GEGLU:
      case VulkanKernelRecipe::REGLU: {
        // Traverse the full input width. Modulo the half-width maps the gate
        // element and its corresponding up-projection element onto the same
        // output coordinate, expressing the true two-halves dependency while
        // keeping the concatenated affine maps invertible.
        const int rank = outputs[0]->rankOf();
        const sd::LongType half = outputs[0]->sizeAt(rank - 1);
        std::ostringstream dimensions;
        std::ostringstream outputDimensions;
        for (int d = 0; d < rank; ++d) {
          if (d != 0) {
            dimensions << ", ";
            outputDimensions << ", ";
          }
          dimensions << "d" << d;
          outputDimensions
              << (d == rank - 1
                      ? "d" + std::to_string(d) + " mod " +
                            std::to_string(half)
                      : "d" + std::to_string(d));
          iterators.push_back("\"parallel\"");
        }
        const std::string dims = dimensions.str();
        const std::string inputMap =
            "affine_map<(" + dims + ") -> (" + dims + ")>";
        const std::string outputMap =
            "affine_map<(" + dims + ") -> (" + outputDimensions.str() + ")>";
        maps = {inputMap, outputMap};
        break;
      }
      default:
        return "";
    }

    if (maps.size() != static_cast<size_t>(numIn + numOut)) return "";
    const double epsilon =
        slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 1.0e-5;
    auto inputMemrefBody = [&](int index) {
      return mlirMemrefBody(
          inputs[index], inputStorageTypes[static_cast<size_t>(index)]);
    };
    auto outputMemrefBody = [&](int index) {
      return mlirMemrefBody(
          outputs[index], outputStorageTypes[static_cast<size_t>(index)]);
    };

    ss << "module {\n  func.func @main(";
    bool firstArgument = true;
    for (int i = 0; i < numIn; ++i) {
      if (!firstArgument) ss << ", ";
      firstArgument = false;
      ss << "%in" << i << ": memref<" << inputMemrefBody(i) << ">";
    }
    for (int i = 0; i < numOut; ++i) {
      if (!firstArgument) ss << ", ";
      firstArgument = false;
      ss << "%out" << i << ": memref<" << outputMemrefBody(i) << ">";
    }
    ss << ") {\n    linalg.generic {nd4j.op_hash = "
       << static_cast<long long>(slot.ident.opHash)
       << " : i64, nd4j.accumulator_type = " << accTs;
    if (emitter->recipe == VulkanKernelRecipe::WINDOW_PARTITION) {
      ss << ", nd4j.window_size = " << slot.args.iArgs[0] << " : i64";
    } else if (emitter->recipe ==
               VulkanKernelRecipe::WINDOW_UNPARTITION) {
      ss << ", nd4j.window_size = " << slot.args.iArgs[0] << " : i64"
         << ", nd4j.output_height = " << slot.args.iArgs[1] << " : i64"
         << ", nd4j.output_width = " << slot.args.iArgs[2] << " : i64";
    } else if (emitter->recipe == VulkanKernelRecipe::BIAS_ADD) {
      const bool nchw =
          slot.args.numBArgs == 1 && slot.args.bArgs[0];
      const int channelAxis = nchw ? 1 : inputs[0]->rankOf() - 1;
      ss << ", nd4j.channel_axis = " << channelAxis << " : i64"
         << ", nd4j.input0_unsigned = "
         << (dataInputUnsigned ? "true" : "false");
    } else if (emitter->recipe == VulkanKernelRecipe::PRELU) {
      ss << ", nd4j.shared_axes = array<i64";
      if (slot.args.numIArgs > 0) ss << ": ";
      for (int i = 0; i < slot.args.numIArgs; ++i) {
        if (i != 0) ss << ", ";
        sd::LongType axis = slot.args.iArgs[i];
        if (axis <= 0) axis += inputs[0]->rankOf() - 1;
        ss << axis;
      }
      ss << ">, nd4j.input0_unsigned = "
         << (dataInputUnsigned ? "true" : "false");
    } else if (emitter->recipe == VulkanKernelRecipe::BATCH_NORM) {
      const int rank = inputs[0]->rankOf();
      ss << ", nd4j.apply_scale = "
         << (slot.args.iArgs[0] != 0 ? "true" : "false")
         << ", nd4j.apply_offset = "
         << (slot.args.iArgs[1] != 0 ? "true" : "false")
         << ", nd4j.normalization_axes = array<i64: ";
      if (slot.args.numIArgs == 2) {
        ss << rank - 1;
      } else {
        for (int i = 2; i < slot.args.numIArgs; ++i) {
          if (i != 2) ss << ", ";
          int64_t axis = -1;
          if (!normalizeAxis(slot.args.iArgs[i], rank, axis)) return "";
          ss << axis;
        }
      }
      ss << ">, nd4j.epsilon = " << std::scientific
         << std::setprecision(std::numeric_limits<double>::max_digits10)
         << slot.args.tArgs[0] << " : " << accTs;
    } else if (emitter->recipe ==
        VulkanKernelRecipe::VISION_EMBEDDING_MERGE) {
      const sd::LongType targetTokenId = slot.args.iArgs[0];
      const bool targetInRange =
          tokenUnsigned
              ? targetTokenId >= 0 &&
                    static_cast<uint64_t>(targetTokenId) <=
                        std::numeric_limits<uint32_t>::max()
              : targetTokenId >= std::numeric_limits<int32_t>::min() &&
                    targetTokenId <= std::numeric_limits<int32_t>::max();
      ss << ", nd4j.target_token_id = " << targetTokenId << " : i64"
         << ", nd4j.token_unsigned = "
         << (tokenUnsigned ? "true" : "false")
         << ", nd4j.target_in_range = "
         << (targetInRange ? "true" : "false");
    } else if (emitter->recipe == VulkanKernelRecipe::APPLY_ALIBI) {
      const sd::LongType heads = inputs[0]->sizeAt(1);
      ss << ", nd4j.num_heads = " << heads << " : i64";
    } else if (emitter->recipe == VulkanKernelRecipe::DOT_PRODUCT_ATTENTION) {
      ss << ", nd4j.attention_normalize = "
         << (slot.args.iArgs[0] != 0 ? "true" : "false")
         << ", nd4j.output_weights = "
         << (slot.args.iArgs[1] != 0 ? "true" : "false");
    } else if (emitter->recipe == VulkanKernelRecipe::FLASH_ATTENTION ||
               emitter->recipe ==
                   VulkanKernelRecipe::GROUPED_QUERY_ATTENTION) {
      const double scale =
          slot.args.numTArgs == 1 ? slot.args.tArgs[0] : 0.0;
      const bool causal =
          slot.args.numBArgs == 0 || slot.args.bArgs[0];
      ss << ", nd4j.attention_scale = " << std::scientific
         << std::setprecision(std::numeric_limits<double>::max_digits10)
         << scale << " : " << accTs
         << ", nd4j.attention_causal = "
         << (causal ? "true" : "false");
    } else if (emitter->recipe == VulkanKernelRecipe::ROPE ||
               emitter->recipe == VulkanKernelRecipe::ROPE_BP) {
      const sd::LongType ropeType =
          slot.args.numIArgs > 0 ? slot.args.iArgs[0] : 0;
      const sd::LongType positionOffset =
          slot.args.numIArgs > 1 ? slot.args.iArgs[1] : 0;
      const sd::LongType requestedRotaryDimensions =
          slot.args.numIArgs > 2 ? slot.args.iArgs[2] : 0;
      const sd::LongType headDimension =
          inputs[0]->sizeAt(inputs[0]->rankOf() - 1);
      const sd::LongType rotaryDimensions =
          requestedRotaryDimensions > 0 &&
                  requestedRotaryDimensions < headDimension
              ? requestedRotaryDimensions
              : headDimension;
      const double frequencyBase =
          slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 10000.0;
      const double frequencyScale =
          slot.args.numTArgs > 1 ? slot.args.tArgs[1] : 1.0;
      ss << ", nd4j.rope_type = " << ropeType << " : i64"
         << ", nd4j.position_offset = " << positionOffset << " : i64"
         << ", nd4j.rotary_dims = " << rotaryDimensions << " : i64"
         << ", nd4j.frequency_base = " << std::scientific
         << std::setprecision(std::numeric_limits<double>::max_digits10)
         << frequencyBase << " : " << accTs
         << ", nd4j.frequency_scale = " << frequencyScale
         << " : " << accTs;
    } else if (emitter->recipe ==
               VulkanKernelRecipe::FUSED_ELEMENTWISE_CHAIN) {
      ss << ", nd4j.chain_ops = array<i64: ";
      for (int i = 0; i < slot.args.numIArgs; ++i) {
        if (i != 0) ss << ", ";
        ss << slot.args.iArgs[i];
      }
      ss << ">";
      if (slot.args.numTArgs == 2) {
        ss << ", nd4j.clip_min = " << std::scientific
           << std::setprecision(std::numeric_limits<double>::max_digits10)
           << slot.args.tArgs[0] << " : " << accTs
           << ", nd4j.clip_max = " << slot.args.tArgs[1]
           << " : " << accTs;
      }
    } else if (emitter->recipe == VulkanKernelRecipe::FUSED_MROPE) {
      const sd::LongType sectionT =
          slot.args.numIArgs > 0 ? slot.args.iArgs[0] : 24;
      const sd::LongType sectionH =
          slot.args.numIArgs > 1 ? slot.args.iArgs[1] : 20;
      const sd::LongType sectionW =
          slot.args.numIArgs > 2 ? slot.args.iArgs[2] : 20;
      const bool interleaved =
          slot.args.numIArgs > 3 && slot.args.iArgs[3] != 0;
      const double frequencyBase =
          slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 10000.0;
      ss << ", nd4j.section_t = " << sectionT << " : i64"
         << ", nd4j.section_h = " << sectionH << " : i64"
         << ", nd4j.section_w = " << sectionW << " : i64"
         << ", nd4j.interleaved = "
         << (interleaved ? "true" : "false")
         << ", nd4j.frequency_base = " << std::scientific
         << std::setprecision(std::numeric_limits<double>::max_digits10)
         << frequencyBase << " : " << accTs
         << ", nd4j.position_unsigned = "
         << (positionUnsigned ? "true" : "false");
    } else if (hasVulkanEmitterTrait(
                   *emitter, VULKAN_EMITTER_TRAIT_EPSILON_PARAMETER)) {
      ss << ", nd4j.epsilon = " << std::scientific
         << std::setprecision(std::numeric_limits<double>::max_digits10)
         << epsilon << " : " << accTs;
    }
    ss << ",\n                    indexing_maps = [";
    for (size_t i = 0; i < maps.size(); ++i) {
      if (i != 0) ss << ", ";
      ss << maps[i];
    }
    ss << "],\n                    iterator_types = [";
    for (size_t i = 0; i < iterators.size(); ++i) {
      if (i != 0) ss << ", ";
      ss << iterators[i];
    }
    ss << "]}\n      ins(";
    for (int i = 0; i < numIn; ++i) {
      if (i != 0) ss << ", ";
      ss << "%in" << i;
    }
    ss << " : ";
    for (int i = 0; i < numIn; ++i) {
      if (i != 0) ss << ", ";
      ss << "memref<" << inputMemrefBody(i) << ">";
    }
    ss << ")\n      outs(";
    for (int i = 0; i < numOut; ++i) {
      if (i != 0) ss << ", ";
      ss << "%out" << i;
    }
    ss << " : ";
    for (int i = 0; i < numOut; ++i) {
      if (i != 0) ss << ", ";
      ss << "memref<" << outputMemrefBody(i) << ">";
    }
    ss << ") {\n      ^bb0(";
    bool firstRegionArgument = true;
    for (int i = 0; i < numIn; ++i) {
      if (!firstRegionArgument) ss << ", ";
      firstRegionArgument = false;
      ss << "%iv" << i << ": "
         << inputStorageTypes[static_cast<size_t>(i)];
    }
    for (int i = 0; i < numOut; ++i) {
      if (!firstRegionArgument) ss << ", ";
      firstRegionArgument = false;
      ss << "%ov" << i << ": "
         << outputStorageTypes[static_cast<size_t>(i)];
    }
    ss << "):\n        linalg.yield ";
    for (int i = 0; i < numOut; ++i) {
      if (i != 0) ss << ", ";
      ss << "%ov" << i;
    }
    ss << " : ";
    for (int i = 0; i < numOut; ++i) {
      if (i != 0) ss << ", ";
      ss << outputStorageTypes[static_cast<size_t>(i)];
    }
    ss << "\n    }\n    return\n  }\n}\n";
    return ss.str();
  }

  if constexpr (Policy::reduction) {
    const int rank = inputs[0]->rankOf();
    std::vector<int64_t> axes;
    bool keepDims = false;
    bool biasCorrected = false;
    const auto* emitter = emitterForSlot(slot);
    if (emitter == nullptr) return "";
    if (hasVulkanEmitterTrait(*emitter, VULKAN_EMITTER_TRAIT_IMPLICIT_LAST_AXIS)) {
      axes.push_back(rank - 1);
      keepDims = slot.args.numIArgs == 0 || slot.args.iArgs[0] != 0;
    } else if (!reductionForSlot(
                   slot, inputs[0], *emitter, outputs, numOut, axes, keepDims,
                   biasCorrected)) {
      return "";
    }

    std::string outputTs;
    std::string outputAccTs;
    bool outputUnsigned = false;
    if (!selectMlirScalarTypes(outputs[0]->dataType(), caps, outputTs,
                               outputAccTs, outputUnsigned)) {
      return "";
    }
    std::string reductionAccTs = accTs;
    if (reductionProducesFloatingOutput(*emitter)) {
      if (accTs == "f64" || outputAccTs == "f64") {
        reductionAccTs = "f64";
      } else if (accTs == "f32" || outputAccTs == "f32") {
        reductionAccTs = "f32";
      } else if (accTs == "f16" || outputAccTs == "f16") {
        reductionAccTs = "f16";
      } else {
        return "";
      }
    }
    auto buildShape = [&](NDArray* array, const std::string& storageType) {
      return mlirMemrefBody(array, storageType);
    };
    std::ostringstream axesText;
    axesText << "array<i64";
    if (!axes.empty()) axesText << ": ";
    for (size_t i = 0; i < axes.size(); ++i) {
      if (i != 0) axesText << ", ";
      axesText << axes[i];

    }
    axesText << ">";

    std::set<int64_t> reduced(axes.begin(), axes.end());
    std::ostringstream dims;
    std::ostringstream iterators;
    for (int d = 0; d < rank; ++d) {
      if (d != 0) {
        dims << ", ";
        iterators << ", ";
      }
      dims << "d" << d;
      iterators << (reduced.count(d) ? "\"reduction\"" : "\"parallel\"");
    }
    const std::string dimList = dims.str();
    const std::string inputMap = "affine_map<(" + dimList + ") -> (" + dimList + ")>";
    std::ostringstream outputMap;
    outputMap << "affine_map<(" << dimList << ") -> (";
    bool first = true;
    for (int d = 0; d < rank; ++d) {
      if (reduced.count(d) == 0 || keepDims) {
        if (!first) outputMap << ", ";
        outputMap << (reduced.count(d) == 0 ? "d" + std::to_string(d) : "0");
        first = false;
      }
    }
    outputMap << ")>";

    ss << "module {\n"
       << "  func.func @main(%X: memref<" << buildShape(inputs[0], ts) << ">, "
       << "%Y: memref<" << buildShape(outputs[0], outputTs) << ">) {\n"
       << "    linalg.generic {" << emitterIdentityAttributes(slot)
       << ", nd4j.accumulator_type = " << reductionAccTs << ",\n"
       << "                    nd4j.nd_reduce = true,\n"
       << "                    nd4j.input0_unsigned = "
       << (isUnsigned ? "true" : "false") << ",\n"
       << "                    nd4j.output_unsigned = "
       << (outputUnsigned ? "true" : "false") << ",\n"
       << "                    nd4j.reduce_axes = " << axesText.str()
       << ",\n"
       << "                    nd4j.keep_dims = " << (keepDims ? "true" : "false") << ",\n"
       << "                    nd4j.bias_corrected = "
       << (biasCorrected ? "true" : "false") << ",\n"
       << "                    indexing_maps = [" << inputMap << ", " << outputMap.str() << "],\n"
       << "                    iterator_types = [" << iterators.str() << "]}\n"
       << "      ins(%X : memref<" << buildShape(inputs[0], ts) << ">)\n"
       << "      outs(%Y : memref<" << buildShape(outputs[0], outputTs) << ">) {\n"
       << "      ^bb0(%xv: " << ts << ", %yv: " << outputTs << "):\n"
       << "        linalg.yield %yv : " << outputTs << "\n"
       << "    }\n"
       << "    return\n"
       << "  }\n"
       << "}\n";
    return ss.str();
  }

  return "";  // Not handled — caller treats empty string as failure.
}

static std::string emitCatalogOp(const NativeSlot& slot,
                                 NDArray** inputs, int numIn,
                                 NDArray** outputs, int numOut,
                                 const VulkanDeviceCaps& caps) {
  const auto* emitter = emitterForSlot(slot);
  if (emitter == nullptr) return "";

  if (usesStructuredComputeSchedule(*emitter)) {
    return emitVulkanOp<StructuredComputePolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }

  switch (emitter->loweringContract) {
    case VulkanLoweringContract::SOFTMAX:
      return emitVulkanOp<SoftmaxPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanLoweringContract::LAYER_NORM:
      return emitVulkanOp<LayerNormPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanLoweringContract::FUSED_LLM:
    case VulkanLoweringContract::DEFAULT:
    case VulkanLoweringContract::LINEAR_COPY:
    case VulkanLoweringContract::INDEXED_TAD_MOVEMENT:
      break;
  }

  if (usesBatchedMatrixListSchedule(*emitter)) {
    return emitVulkanOp<BatchedMatrixListPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesIndexedAccumulationSchedule(*emitter)) {
    return emitVulkanOp<IndexedAccumulationPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesIndexedTadMovementSchedule(*emitter)) {
    return emitVulkanOp<IndexedTadMovementPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesContractMovementSchedule(*emitter)) {
    return emitVulkanOp<ContractMovementPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesRowwiseEpsilonNormalizationSchedule(*emitter)) {
    return emitVulkanOp<RmsNormPolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesMultiOutputNormalizationSchedule(*emitter)) {
    return emitVulkanOp<MultiOutputElementwisePolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesCachedRotarySchedule(*emitter)) {
    return emitVulkanOp<RopePolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }
  if (usesRankPermutationSchedule(*emitter)) {
    if (usesDefaultReversePermutationSchedule(*emitter)) {
      return emitVulkanOp<TransposePolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    }
    return emitVulkanOp<PermutePolicy>(
        slot, inputs, numIn, outputs, numOut, caps);
  }

  switch (emitter->family) {
    case VulkanKernelFamily::MATMUL:
      return emitVulkanOp<MatmulPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::ELEMENTWISE_BINARY:
      return emitVulkanOp<BinaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::ELEMENTWISE_UNARY:
      return emitVulkanOp<UnaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::COMPARISON:
    case VulkanKernelFamily::LOGICAL:
      if (emitter->recipe == VulkanKernelRecipe::BOOLEAN_NOT) {
        return emitVulkanOp<UnaryPolicy>(
            slot, inputs, numIn, outputs, numOut, caps);
      }
      return emitVulkanOp<BinaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::TERNARY:
      return emitVulkanOp<TernaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::NORMALIZATION:
      return "";
    case VulkanKernelFamily::DATA_MOVEMENT:
      if (usesIndexedLookupSchedule(*emitter)) {
        return emitVulkanOp<GatherPolicy>(
            slot, inputs, numIn, outputs, numOut, caps);
      }
      if (usesVariadicAxisConcatSchedule(*emitter)) {
        return emitVulkanOp<ConcatPolicy>(
            slot, inputs, numIn, outputs, numOut, caps);
      }
      if (usesMultiOutputPartitionSchedule(*emitter)) {
        return emitVulkanOp<SplitPolicy>(
            slot, inputs, numIn, outputs, numOut, caps);
      }
      return emitVulkanOp<MovementPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::REDUCTION:
      return emitVulkanOp<ReductionPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::CONSTANT_GENERATION:
      return emitVulkanOp<ConstantPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::CAST:
      return emitVulkanOp<UnaryPolicy>(
          slot, inputs, numIn, outputs, numOut, caps);
    case VulkanKernelFamily::UNKNOWN:
    case VulkanKernelFamily::ATTENTION:
      return "";
  }
  return "";
}

}  // anonymous namespace

std::string VulkanSegmentRecorder::emitMlirModule(const NativeSlot& slot,
                                                   NDArray** inputs, int numIn,
                                                   NDArray** outputs, int numOut) const {
  const VulkanDeviceCaps* caps =
      handle_ != nullptr ? handle_->getDeviceCaps() : nullptr;
  const VulkanOpHandler* handler = findVulkanHandler(slot);
  if (caps == nullptr || handler == nullptr) return "";
  return handler->emitMlir(slot, inputs, numIn, outputs, numOut, *caps);
}

// ─────────────────────────────────────────────────────────────────────────────
//  allocateBinding — borrow the DataBuffer's pool-owned Vulkan storage buffer
// ─────────────────────────────────────────────────────────────────────────────

bool VulkanSegmentRecorder::allocateBinding(NDArray* arr, bool readBeforeWrite,
                                             OperandBinding& binding) {
  if (handle_ == nullptr || arr == nullptr || arr->dataBuffer() == nullptr) {
    return false;
  }

  DataBuffer* dataBuffer = arr->dataBuffer();
  const sd::LongType len = dataBuffer->getLenInBytes();
  if (len <= 0) {
    sd_printf("VulkanSegmentRecorder: zero-length DataBuffer cannot be bound\n");
    return false;
  }

  try {
    if (readBeforeWrite) {
      // Capture the exact value an eager CUDA/device op would see. Graph
      // execution suppresses implicit syncs, so force the normal DataBuffer H2D
      // path only when the host side is newer.
      if (!dataBuffer->isSpecialActual()) {
        dataBuffer->syncToSpecial(/*forceSync=*/true);
      } else {
        dataBuffer->allocateSpecial();
      }
    } else {
      // Output-only values need device storage but no host materialization.
      dataBuffer->allocateSpecial();
    }
  } catch (const std::exception& error) {
    sd_printf("VulkanSegmentRecorder: DataBuffer device allocation/sync failed: %s\n",
              error.what());
    return false;
  }

  void* specialToken = dataBuffer->special();
  VulkanAllocRecord record;
  VulkanMemoryPool& pool = VulkanMemoryPool::getInstance();
  const VkDeviceSize requiredBytes = static_cast<VkDeviceSize>(len);
  if (specialToken == nullptr || !pool.queryRecord(specialToken, record) ||
      record.buffer == VK_NULL_HANDLE ||
      record.logicalDevice != handle_->getDevice() ||
      record.deviceId != handle_->getDeviceId() ||
      record.logicalSize < requiredBytes) {
    sd_printf("VulkanSegmentRecorder: DataBuffer special allocation is not a "
              "compatible Vulkan storage buffer (device=%d bytes=%llu)\n",
              handle_->getDeviceId(),
              static_cast<unsigned long long>(requiredBytes));
    return false;
  }

  if (readBeforeWrite && !dataBuffer->isSpecialActual()) {
    sd_printf("VulkanSegmentRecorder: input DataBuffer has no actual device value\n");
    return false;
  }

  binding.dataBuffer = dataBuffer;
  binding.specialToken = specialToken;
  binding.buffer = record.buffer;
  // CUDA kernels receive the backing allocation's base pointer, so strided
  // views can address their full backing span. Vulkan descriptors must expose
  // that same allocation extent explicitly instead of truncating the range to
  // the view's logical DataBuffer length.
  binding.bytes = record.logicalSize;
  binding.readBeforeWrite = readBeforeWrite;
  binding.writtenBySegment = false;
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  recordDispatch — allocate descriptor set, write buffers, cmd dispatch
// ─────────────────────────────────────────────────────────────────────────────

bool VulkanSegmentRecorder::recordDispatch(
    VkPipeline pipeline,
    VkPipelineLayout pipelineLayout,
    VkDescriptorSetLayout descSetLayout,
    const std::vector<uint32_t>& descriptorBindings,
    const std::vector<OperandBinding*>& operands,
    uint32_t groupCountX,
    uint32_t groupCountY,
    uint32_t groupCountZ) {
  if (descriptorBindings.empty() ||
      descriptorBindings.size() != operands.size() ||
      operands.size() > std::numeric_limits<uint32_t>::max()) {
    sd_printf("VulkanSegmentRecorder: descriptor ABI/operand mismatch "
              "(bindings=%zu operands=%zu)\n",
              descriptorBindings.size(), operands.size());
    return false;
  }

  VkDevice dev = handle_->getDevice();

  // Allocate exactly the descriptors declared by this shader interface.
  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = static_cast<uint32_t>(operands.size());


  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;

  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(dev, &poolInfo, nullptr, &descriptorPool) !=
      VK_SUCCESS) {
    sd_printf("VulkanSegmentRecorder: vkCreateDescriptorPool failed\n");

    return false;
  }

  VkDescriptorSetAllocateInfo dsAllocInfo = {};
  dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAllocInfo.descriptorPool = descriptorPool;
  dsAllocInfo.descriptorSetCount = 1;
  dsAllocInfo.pSetLayouts = &descSetLayout;

  VkDescriptorSet ds = VK_NULL_HANDLE;
  if (vkAllocateDescriptorSets(dev, &dsAllocInfo, &ds) != VK_SUCCESS) {
    sd_printf("VulkanSegmentRecorder: vkAllocateDescriptorSets failed\n");
    vkDestroyDescriptorPool(dev, descriptorPool, nullptr);
    return false;
  }
  descriptorPools_.push_back(descriptorPool);

  std::vector<VkDescriptorBufferInfo> bufInfos(operands.size());
  std::vector<VkWriteDescriptorSet> writes(operands.size());
  for (size_t i = 0; i < operands.size(); ++i) {
    bufInfos[i].buffer = operands[i]->buffer;
    bufInfos[i].offset = 0;
    bufInfos[i].range = operands[i]->bytes;

    writes[i] = {};
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = descriptorBindings[i];
    writes[i].dstArrayElement = 0;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufInfos[i];
  }
  vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()),
                         writes.data(), 0, nullptr);

  // Record the exact workgroup grid encoded by this operation's GPU kernel.
  handle_->recordDispatch(pipeline, pipelineLayout, ds,
                          groupCountX, groupCountY, groupCountZ);

  // Barrier between ops so writes from this dispatch are visible to the next.
  handle_->recordComputeBarrier();

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  recordOp — main entry point called once per slot during capture
// ─────────────────────────────────────────────────────────────────────────────

bool VulkanSegmentRecorder::recordOp(const NativeSlot& slot,
                                      NDArray** inputs, int numIn,
                                      NDArray** outputs, int numOut,
                                      RandomGenerator* randomState) {
  const std::string& opName = slot.ident.opName;
  const VulkanOpHandler* handler = findVulkanHandler(slot);
  const auto* emitter = emitterForSlot(slot);
  if (handler == nullptr || emitter == nullptr) return false;
  const bool usesRandomState = hasVulkanEmitterTrait(
      *emitter, VULKAN_EMITTER_TRAIT_RANDOM_STATE);
  if (usesRandomState &&
      (randomState == nullptr || numOut != 1 || outputs == nullptr ||
       outputs[0] == nullptr)) {
    DSP_DIAG(GRAPH_REPLAY,
             "VulkanSegmentRecorder: runtime state contract is incomplete "
             "for opName=%s",
             opName.c_str());
    return false;
  }
  DSP_DIAG(GRAPH_REPLAY,
           "VulkanSegmentRecorder::recordOp opName=%s numIn=%d numOut=%d",
           opName.c_str(), numIn, numOut);

  // Emit MLIR module text.
  std::string mlir = emitMlirModule(slot, inputs, numIn, outputs, numOut);
  if (mlir.empty()) {
    DSP_DIAG(GRAPH_REPLAY,
             "VulkanSegmentRecorder: emitMlirModule returned empty for opName=%s",
             opName.c_str());
    return false;
  }

  // Bind inputs first, then outputs so descriptor order matches the MLIR
  // function signature. A graph value must retain one VkBuffer across producer
  // and consumer dispatches, just as CUDA graph capture retains one device
  // pointer. Intern by DataBuffer identity instead of allocating per op.
  // Reserve the maximum number of new bindings up front so pointers collected
  // below remain stable while this op is assembled.
  bindings_.reserve(bindings_.size() + static_cast<size_t>(numIn + numOut));

  auto findBinding = [&](NDArray* array) -> OperandBinding* {
    if (array == nullptr || array->dataBuffer() == nullptr) return nullptr;
    for (auto& existing : bindings_) {
      if (existing.dataBuffer == array->dataBuffer()) {
        return &existing;
      }
    }
    return nullptr;
  };

  std::vector<OperandBinding*> operands;
  operands.reserve(static_cast<size_t>(numIn + numOut +
                                       (usesRandomState ? 1 : 0)));

  if (usesRandomState) {
    randomStateBindings_.emplace_back();
    RandomStateBinding& stateBinding = randomStateBindings_.back();
    stateBinding.hostState = randomState;
    stateBinding.steps = outputs[0]->lengthOf();

    VulkanMemoryPool& pool = VulkanMemoryPool::getInstance();
    const VkDeviceSize stateBytes = sizeof(VulkanRandomStateWords);
    try {
      stateBinding.operand.specialToken =
          pool.allocate(handle_->getDeviceId(), stateBytes);
    } catch (const std::exception& error) {
      DSP_DIAG(GRAPH_REPLAY,
               "VulkanSegmentRecorder: random-state allocation failed: %s",
               error.what());
      randomStateBindings_.pop_back();
      return false;
    }

    VulkanAllocRecord record;
    if (stateBinding.operand.specialToken == nullptr ||
        !pool.queryRecord(stateBinding.operand.specialToken, record) ||
        record.buffer == VK_NULL_HANDLE ||
        record.logicalDevice != handle_->getDevice() ||
        record.deviceId != handle_->getDeviceId() ||
        record.logicalSize < stateBytes) {
      if (stateBinding.operand.specialToken != nullptr) {
        pool.freeImmediate(stateBinding.operand.specialToken);
      }
      randomStateBindings_.pop_back();
      DSP_DIAG(GRAPH_REPLAY,
               "VulkanSegmentRecorder: random-state buffer is incompatible "
               "with replay device %d",
               handle_->getDeviceId());
      return false;
    }

    stateBinding.operand.buffer = record.buffer;
    stateBinding.operand.bytes = stateBytes;
    stateBinding.operand.readBeforeWrite = true;
    operands.push_back(&stateBinding.operand);
  }

  for (int i = 0; i < numIn; ++i) {
    OperandBinding* binding = findBinding(inputs[i]);
    if (binding == nullptr) {
      bindings_.emplace_back();
      binding = &bindings_.back();
      if (!allocateBinding(inputs[i], /*readBeforeWrite=*/true, *binding)) {
        DSP_DIAG(GRAPH_REPLAY,
                 "VulkanSegmentRecorder: allocateBinding failed for input %d opName=%s",
                 i, opName.c_str());
        bindings_.pop_back();
        return false;
      }
    } else if (!binding->writtenBySegment) {
      // This value originates outside the recorded command buffer.
      binding->readBeforeWrite = true;
    }
    operands.push_back(binding);
  }
  for (int i = 0; i < numOut; ++i) {
    OperandBinding* binding = findBinding(outputs[i]);
    if (binding == nullptr) {
      bindings_.emplace_back();
      binding = &bindings_.back();
      if (!allocateBinding(outputs[i], /*readBeforeWrite=*/false, *binding)) {
        DSP_DIAG(GRAPH_REPLAY,
                 "VulkanSegmentRecorder: allocateBinding failed for output %d opName=%s",
                 i, opName.c_str());
        bindings_.pop_back();
        return false;
      }
    }
    binding->writtenBySegment = true;
    operands.push_back(binding);
  }

  // Compile (or retrieve cached) pipeline via the handle's pipeline cache.
  VulkanPipelineCache* cache = handle_->getPipelineCache();
  if (cache == nullptr) {
    DSP_DIAG(GRAPH_REPLAY,
             "VulkanSegmentRecorder: null pipeline cache on handle, opName=%s",
             opName.c_str());
    return false;
  }

  const std::string pipelineKey = emitterPipelineKey(slot);
  if (pipelineKey.empty()) return false;
  VkDevice dev = handle_->getDevice();
  VkPipeline pipeline =
      cache->getOrCompile(pipelineKey, mlir, dev, compilationPolicy_);
  if (pipeline == VK_NULL_HANDLE) {
    DSP_DIAG(GRAPH_REPLAY,
             "VulkanSegmentRecorder: pipeline compilation failed opName=%s",
             opName.c_str());
    return false;
  }

  VkPipelineLayout pipelineLayout = cache->getPipelineLayout(pipelineKey, mlir);
  VkDescriptorSetLayout descSetLayout = cache->getDescriptorSetLayout(pipelineKey, mlir);
  std::vector<uint32_t> descriptorBindings =
      cache->getDescriptorBindings(pipelineKey, mlir);
  if (pipelineLayout == VK_NULL_HANDLE || descSetLayout == VK_NULL_HANDLE ||
      descriptorBindings.empty()) {
    DSP_DIAG(GRAPH_REPLAY,
             "VulkanSegmentRecorder: invalid compiled descriptor ABI "
             "opName=%s bindings=%d signatureOperands=%d",
             opName.c_str(), static_cast<int>(descriptorBindings.size()),
             static_cast<int>(operands.size()));
    return false;
  }

  // MLIR GPU outlining captures only the function arguments used by a kernel.
  // Descriptor binding numbers retain their original function-signature indices,
  // so select the corresponding runtime operands instead of forcing unused
  // arguments into the shader interface.
  std::vector<OperandBinding*> capturedOperands;
  capturedOperands.reserve(descriptorBindings.size());
  for (uint32_t binding : descriptorBindings) {
    if (binding >= operands.size()) {
      DSP_DIAG(GRAPH_REPLAY,
               "VulkanSegmentRecorder: descriptor binding outside function "
               "signature opName=%s binding=%u signatureOperands=%d",
               opName.c_str(), binding, static_cast<int>(operands.size()));
      return false;
    }
    capturedOperands.push_back(operands[binding]);
  }

  DispatchGeometry geometry;
  if (!handler->dispatchGeometry(
          slot, inputs, numIn, outputs, numOut, geometry)) {
    DSP_DIAG(GRAPH_REPLAY,
             "VulkanSegmentRecorder: invalid dispatch geometry opName=%s",
             opName.c_str());
    return false;
  }

  // Record the dispatch + barrier into the command buffer.
  if (!recordDispatch(pipeline, pipelineLayout, descSetLayout,
                      descriptorBindings, capturedOperands,
                      geometry.x, geometry.y, geometry.z)) {
    DSP_DIAG(GRAPH_REPLAY,
             "VulkanSegmentRecorder: recordDispatch failed opName=%s",
             opName.c_str());
    return false;
  }

  DSP_DIAG(GRAPH_REPLAY,
           "VulkanSegmentRecorder: recorded opName=%s bindings=%d "
           "signatureOperands=%d",
           opName.c_str(), static_cast<int>(capturedOperands.size()),
           static_cast<int>(operands.size()));
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device actuality around replay
// ─────────────────────────────────────────────────────────────────────────────

bool VulkanSegmentRecorder::prepareReplayInputs(VulkanExecutionStream* stream) {
  if (handle_ == nullptr || stream == nullptr || !stream->isActive() ||
      stream->deviceId() != handle_->getDeviceId()) {
    sd_printf("VulkanSegmentRecorder: replay requires the captured exact-device "
              "execution stream\n");
    return false;
  }

  for (auto& binding : randomStateBindings_) {
    binding.replayPending = false;
  }

  VulkanMemoryPool& pool = VulkanMemoryPool::getInstance();

  auto bindingStillValid = [&](const OperandBinding& binding) -> bool {
    if (binding.dataBuffer == nullptr ||
        binding.dataBuffer->special() != binding.specialToken) {
      return false;
    }

    VulkanAllocRecord record;
    return pool.queryRecord(binding.specialToken, record) &&
           record.buffer == binding.buffer &&
           record.logicalDevice == handle_->getDevice() &&
           record.deviceId == handle_->getDeviceId() &&
           record.logicalSize >= binding.bytes;
  };

  for (auto& binding : bindings_) {
    if (!binding.readBeforeWrite) continue;

    try {
      if (!bindingStillValid(binding)) {
        sd_printf("VulkanSegmentRecorder: frozen input device binding changed "
                  "before replay\n");
        return false;
      }

      if (!binding.dataBuffer->isSpecialActual()) {
        binding.dataBuffer->syncToSpecial(/*forceSync=*/true);
      }
      if (!binding.dataBuffer->isSpecialActual() ||
          !bindingStillValid(binding)) {
        sd_printf("VulkanSegmentRecorder: input could not be synchronized to "
                  "its captured Vulkan buffer\n");
        return false;
      }
      binding.dataBuffer->readSpecial();
    } catch (const std::exception& error) {
      sd_printf("VulkanSegmentRecorder: input device synchronization failed: %s\n",
                error.what());
      return false;
    }
  }

  std::unordered_map<RandomGenerator*, RandomGenerator> stagedStates;
  for (auto& binding : randomStateBindings_) {
    VulkanAllocRecord record;
    if (binding.hostState == nullptr ||
        binding.operand.specialToken == nullptr ||
        !pool.queryRecord(binding.operand.specialToken, record) ||
        record.buffer != binding.operand.buffer ||
        record.logicalDevice != handle_->getDevice() ||
        record.deviceId != handle_->getDeviceId() ||
        record.logicalSize < binding.operand.bytes) {
      sd_printf("VulkanSegmentRecorder: captured random-state binding changed "
                "before replay\n");
      for (auto& pending : randomStateBindings_) {
        pending.replayPending = false;
      }
      return false;
    }

    auto staged = stagedStates.emplace(binding.hostState, *binding.hostState);
    VulkanRandomStateWords words = randomStateWords(staged.first->second);
    if (!pool.copyHostToDeviceAsync(binding.operand.specialToken, &words,
                                    sizeof(words), stream)) {
      sd_printf("VulkanSegmentRecorder: random-state upload failed\n");
      for (auto& pending : randomStateBindings_) {
        pending.replayPending = false;
      }
      return false;
    }

    binding.replayPending = true;
    staged.first->second.rewindH(static_cast<uint64_t>(binding.steps));
  }

  return true;
}

void VulkanSegmentRecorder::markReplayOutputs() {
  for (auto& binding : bindings_) {
    if (binding.writtenBySegment && binding.dataBuffer != nullptr) {
      binding.dataBuffer->writeSpecial();
    }
  }

  // Advance caller-owned state only after replay completion. This mirrors CUDA's
  // successful-launch lifecycle and makes duplicate post-replay fixups harmless.
  for (auto& binding : randomStateBindings_) {
    if (binding.replayPending && binding.hostState != nullptr) {
      binding.hostState->rewindH(static_cast<uint64_t>(binding.steps));
      binding.replayPending = false;
    }
  }
}

}  // namespace graph
}  // namespace sd

#endif  // defined(HAVE_VULKAN) && HAVE_VULKAN
