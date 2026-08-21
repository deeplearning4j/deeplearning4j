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

#if HAVE_NNAPI

#include <graph/cpu/NnapiGraphBackend.h>
#include <graph/DspDiagnostics.h>
#include <graph/gpu/OpCategoryTable.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <system/Environment.h>
#include <types/float16.h>

#include <android/NeuralNetworks.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>

namespace sd {
namespace graph {

// ─── Helpers ────────────────────────────────────────────────────────────────

static int getAndroidApiLevel() {
  char value[PROP_VALUE_MAX] = {};
  __system_property_get("ro.build.version.sdk", value);
  return atoi(value);
}

static std::string toLower(const std::string& s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return r;
}

static bool isDenseCOrder(NDArray* arr) {
  return arr != nullptr && arr->ordering() == 'c' &&
         shape::strideDescendingCAscendingF(arr->shapeInfo());
}

// NNAPI vendor compilation is synchronous on Android. Keep admission bounded
// so a large transformer graph cannot monopolize the runtime until the IPC
// watchdog kills the worker. Larger graphs must be split or explicitly replayed.
// 32 keeps individual vendor compilations small enough for the Tensor G3 deadline
// while still allowing useful arithmetic islands to remain on the accelerator.
static constexpr int kMaxNnapiSegmentOps = 32;
static constexpr size_t kNnapiOutputGuardBytes = 256;
static constexpr uint8_t kNnapiOutputGuardValue = 0xa5;
static constexpr size_t kNnapiCacheTokenBytes = 32;

struct EmbeddingLookupContract {
  bool valid = false;
  bool flattenLookups = false;
  LongType lookupCount = 0;
  std::vector<LongType> flattenedOutputDimensions;
};

static EmbeddingLookupContract getEmbeddingLookupContract(
    NDArray* values, NDArray* lookups, NDArray* output, int axis) {
  EmbeddingLookupContract contract;
  if (values == nullptr || lookups == nullptr || output == nullptr || axis != 0 ||
      values->rankOf() < 2 || lookups->rankOf() < 1 ||
      (lookups->dataType() != DataType::INT32 &&
       lookups->dataType() != DataType::INT64) ||
      output->dataType() != values->dataType() ||
      output->rankOf() != lookups->rankOf() + values->rankOf() - 1) {
    return contract;
  }

  for (int dimension = 0; dimension < lookups->rankOf(); ++dimension) {
    if (output->sizeAt(dimension) != lookups->sizeAt(dimension)) {
      return contract;
    }
  }
  for (int dimension = 1; dimension < values->rankOf(); ++dimension) {
    const int outputDimension = lookups->rankOf() + dimension - 1;
    if (output->sizeAt(outputDimension) != values->sizeAt(dimension)) {
      return contract;
    }
  }

  const LongType lookupCount = lookups->lengthOf();
  if (lookupCount <= 0 ||
      static_cast<unsigned long long>(lookupCount) >
          static_cast<unsigned long long>(std::numeric_limits<int32_t>::max())) {
    return contract;
  }

  contract.valid = true;
  contract.flattenLookups = lookups->rankOf() != 1;
  contract.lookupCount = lookupCount;
  contract.flattenedOutputDimensions.push_back(lookupCount);
  for (int dimension = 1; dimension < values->rankOf(); ++dimension) {
    contract.flattenedOutputDimensions.push_back(values->sizeAt(dimension));
  }
  return contract;
}

static std::array<uint8_t, kNnapiCacheTokenBytes> makeNnapiCacheToken(
    const std::string& modelKey, int startSlot, int endSlot, LongType shapeKey) {
  const std::string material = modelKey + ":" + std::to_string(startSlot) + ":" +
                               std::to_string(endSlot) + ":" +
                               std::to_string(static_cast<long long>(shapeKey));
  std::array<uint8_t, kNnapiCacheTokenBytes> token{};
  for (size_t lane = 0; lane < 4; ++lane) {
    uint64_t hash = 1469598103934665603ULL ^
                    (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(lane + 1));
    for (unsigned char value : material) {
      hash ^= static_cast<uint64_t>(value);
      hash *= 1099511628211ULL;
    }
    for (size_t byte = 0; byte < sizeof(hash); ++byte) {
      token[lane * sizeof(hash) + byte] =
          static_cast<uint8_t>((hash >> (byte * 8)) & 0xffU);
    }
  }
  return token;
}

#if defined(SD_NNAPI_ACCELERATOR_ONLY)
#define SD_NNAPI_STRINGIFY_INNER(value) #value
#define SD_NNAPI_STRINGIFY(value) SD_NNAPI_STRINGIFY_INNER(value)
#if defined(SD_NNAPI_REQUIRED_DEVICE_NAME)
static constexpr const char* kRequiredNnapiAcceleratorDevice =
    SD_NNAPI_STRINGIFY(SD_NNAPI_REQUIRED_DEVICE_NAME);
#else
static constexpr const char* kRequiredNnapiAcceleratorDevice = "google-edgetpu";
#endif
#endif

// ─── Data type support ──────────────────────────────────────────────────────

int32_t NnapiGraphBackend::toNnapiOperandType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32: return ANEURALNETWORKS_TENSOR_FLOAT32;
    case DataType::HALF:    return ANEURALNETWORKS_TENSOR_FLOAT16;
    case DataType::INT32:   return ANEURALNETWORKS_TENSOR_INT32;
    case DataType::BOOL:    return ANEURALNETWORKS_TENSOR_BOOL8;
    // INT8/UINT8 are deliberately not mapped. This backend has no quantization
    // parameter plumbing (scale/zero-point), and declaring QUANT8 operands with
    // scale = 0 would be an invalid model. Unsupported non-floating types are
    // rejected during model construction instead of silently reinterpreted.
    default:                return -1;
  }
}

bool NnapiGraphBackend::isNnapiSupportedType(DataType dt) {
  return toNnapiOperandType(dt) >= 0;
}

// ─── Op mapping table ───────────────────────────────────────────────────────

int NnapiGraphBackend::getNnapiOpCode(const std::string& opName) {
  std::string name = toLower(opName);

  // ── NNAPI 1.0 (API 27+) ──────────────────────────────────────────────

  // Binary elementwise
  if (name == "add")         return ANEURALNETWORKS_ADD;
  if (name == "subtract" || name == "sub") return ANEURALNETWORKS_SUB;
  if (name == "multiply" || name == "mul") return ANEURALNETWORKS_MUL;
  if (name == "divide" || name == "div" || name == "realdiv")
                             return ANEURALNETWORKS_DIV;

  // Activations
  if (name == "relu")        return ANEURALNETWORKS_RELU;
  if (name == "relu6")       return ANEURALNETWORKS_RELU6;
  if (name == "sigmoid" || name == "logistic")
                             return ANEURALNETWORKS_LOGISTIC;
  if (name == "tanh")        return ANEURALNETWORKS_TANH;

  // Normalization
  if (name == "softmax")     return ANEURALNETWORKS_SOFTMAX;
  if (name == "lrn" || name == "local_response_normalization")
                             return ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION;

  // Convolution
  if (name == "conv2d")      return ANEURALNETWORKS_CONV_2D;
  if (name == "depthwise_conv2d" || name == "sconv2d")
                             return ANEURALNETWORKS_DEPTHWISE_CONV_2D;

  // Pooling
  if (name == "avgpool2d" || name == "avgpool")
                             return ANEURALNETWORKS_AVERAGE_POOL_2D;
  if (name == "maxpool2d" || name == "maxpool")
                             return ANEURALNETWORKS_MAX_POOL_2D;

  // FC
  if (name == "xw_plus_b" || name == "fully_connected")
                             return ANEURALNETWORKS_FULLY_CONNECTED;

  // Shape / data movement (basic)
  if (name == "reshape")     return ANEURALNETWORKS_RESHAPE;
  if (name == "concat" || name == "concatenate")
                             return ANEURALNETWORKS_CONCATENATION;
  if (name == "floor")       return ANEURALNETWORKS_FLOOR;

  // Space transforms
  if (name == "space_to_depth")
                             return ANEURALNETWORKS_SPACE_TO_DEPTH;
  if (name == "depth_to_space")
                             return ANEURALNETWORKS_DEPTH_TO_SPACE;

  // ── NNAPI 1.1 (API 28+) ──────────────────────────────────────────────
  if (name == "batch_to_space" || name == "batch_to_space_nd")
                             return ANEURALNETWORKS_BATCH_TO_SPACE_ND;
  if (name == "space_to_batch" || name == "space_to_batch_nd")
                             return ANEURALNETWORKS_SPACE_TO_BATCH_ND;
  if (name == "squeeze")     return ANEURALNETWORKS_SQUEEZE;
  if (name == "strided_slice" || name == "slice")
                             return ANEURALNETWORKS_STRIDED_SLICE;
  if (name == "transpose" || name == "permute")
                             return ANEURALNETWORKS_TRANSPOSE;
  if (name == "pad")         return ANEURALNETWORKS_PAD;
  if (name == "reduce_mean" || name == "mean")
                             return ANEURALNETWORKS_MEAN;

  // ── NNAPI 1.2 (API 29+) ──────────────────────────────────────────────

  // Unary elementwise
  if (name == "abs")         return ANEURALNETWORKS_ABS;
  if (name == "exp")         return ANEURALNETWORKS_EXP;
  if (name == "log")         return ANEURALNETWORKS_LOG;
  if (name == "neg")         return ANEURALNETWORKS_NEG;
  if (name == "sqrt")        return ANEURALNETWORKS_SQRT;
  if (name == "rsqrt")       return ANEURALNETWORKS_RSQRT;
  if (name == "sin")         return ANEURALNETWORKS_SIN;

  // Binary elementwise
  if (name == "maximum")     return ANEURALNETWORKS_MAXIMUM;
  if (name == "minimum")     return ANEURALNETWORKS_MINIMUM;
  if (name == "pow")         return ANEURALNETWORKS_POW;

  // Comparison
  if (name == "less")        return ANEURALNETWORKS_LESS;
  if (name == "less_equal")  return ANEURALNETWORKS_LESS_EQUAL;
  if (name == "greater")     return ANEURALNETWORKS_GREATER;
  if (name == "greater_equal") return ANEURALNETWORKS_GREATER_EQUAL;
  if (name == "equals" || name == "equal")
                             return ANEURALNETWORKS_EQUAL;
  if (name == "not_equals" || name == "not_equal")
                             return ANEURALNETWORKS_NOT_EQUAL;

  // Logical
  if (name == "boolean_and") return ANEURALNETWORKS_LOGICAL_AND;
  if (name == "boolean_or")  return ANEURALNETWORKS_LOGICAL_OR;
  if (name == "boolean_not") return ANEURALNETWORKS_LOGICAL_NOT;

  // Reduction
  if (name == "reduce_sum")  return ANEURALNETWORKS_REDUCE_SUM;
  if (name == "reduce_max")  return ANEURALNETWORKS_REDUCE_MAX;
  if (name == "reduce_min")  return ANEURALNETWORKS_REDUCE_MIN;
  if (name == "reduce_prod") return ANEURALNETWORKS_REDUCE_PROD;
  if (name == "reduce_any")  return ANEURALNETWORKS_REDUCE_ANY;
  if (name == "reduce_all")  return ANEURALNETWORKS_REDUCE_ALL;

  // ArgMax/ArgMin
  if (name == "argmax")      return ANEURALNETWORKS_ARGMAX;
  if (name == "argmin")      return ANEURALNETWORKS_ARGMIN;

  // Data type
  if (name == "cast")        return ANEURALNETWORKS_CAST;

  // Select/where
  if (name == "where" || name == "select")
                             return ANEURALNETWORKS_SELECT;

  // Data movement
  if (name == "gather")      return ANEURALNETWORKS_GATHER;
  if (name == "expand_dims") return ANEURALNETWORKS_EXPAND_DIMS;
  if (name == "tile")        return ANEURALNETWORKS_TILE;
  if (name == "split" || name == "split_v")
                             return ANEURALNETWORKS_SPLIT;

  // Resize
  if (name == "resize_bilinear")
                             return ANEURALNETWORKS_RESIZE_BILINEAR;
  if (name == "resize_nearest" || name == "resize_nearest_neighbor")
                             return ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;

  // Transpose conv
  if (name == "deconv2d" || name == "conv2d_transpose")
                             return ANEURALNETWORKS_TRANSPOSE_CONV_2D;

  // ── NNAPI 1.3 (API 30+) ──────────────────────────────────────────────
  if (name == "matmul" || name == "mmul")
                             return ANEURALNETWORKS_BATCH_MATMUL;

  return -1;  // Unmappable
}

// ─── API level requirements ─────────────────────────────────────────────────

int NnapiGraphBackend::getMinApiLevel(const std::string& opName) {
  std::string name = toLower(opName);

  // NNAPI 1.3 (API 30)
  if (name == "matmul" || name == "mmul") return 30;

  // NNAPI 1.2 (API 29)
  static const std::unordered_set<std::string> api29Ops = {
      "abs", "exp", "log", "neg", "sqrt", "rsqrt", "sin",
      "maximum", "minimum", "pow",
      "less", "less_equal", "greater", "greater_equal",
      "equals", "equal", "not_equals", "not_equal",
      "boolean_and", "boolean_or", "boolean_not",
      "reduce_sum", "reduce_max", "reduce_min", "reduce_prod",
      "reduce_any", "reduce_all",
      "argmax", "argmin",
      "cast",
      "where", "select",
      "gather", "expand_dims", "tile", "split", "split_v",
      "resize_bilinear", "resize_nearest", "resize_nearest_neighbor",
      "deconv2d", "conv2d_transpose",
  };
  if (api29Ops.count(name)) return 29;

  // NNAPI 1.1 (API 28)
  static const std::unordered_set<std::string> api28Ops = {
      "batch_to_space", "batch_to_space_nd",
      "space_to_batch", "space_to_batch_nd",
      "squeeze", "strided_slice", "slice",
      "transpose", "permute", "pad",
      "reduce_mean", "mean",
  };
  if (api28Ops.count(name)) return 28;

  // NNAPI 1.0 (API 27) — everything else we map
  return 27;
}

// ─── Concrete lowering contract ──────────────────────────────────────────────

bool NnapiGraphBackend::validateSlotContract(const NativeSlot& slot, int nnapiOp,
                                             std::string& reason) {
  const auto& wiring = slot.wiring;
  const auto& args = slot.args;

  if (wiring.numInputs < 0 || wiring.numOutputs < 0 || args.numIArgs < 0 ||
      args.numTArgs < 0 || args.numBArgs < 0 || args.numDArgs < 0 ||
      args.numSArgs < 0) {
    reason = "negative wiring or argument count";
    return false;
  }
  if ((wiring.numInputs > 0 && wiring.inputSourceIndices == nullptr) ||
      (wiring.numOutputs > 0 && wiring.outputSlotIndices == nullptr) ||
      (args.numIArgs > 0 && args.iArgs == nullptr) ||
      (args.numTArgs > 0 && args.tArgs == nullptr) ||
      (args.numBArgs > 0 && args.bArgs == nullptr) ||
      (args.numDArgs > 0 && args.dArgs == nullptr) ||
      (args.numSArgs > 0 && args.sArgs == nullptr)) {
    reason = "non-zero count has a null backing buffer";
    return false;
  }

  auto fail = [&](const char* message) {
    reason = message;
    return false;
  };
  auto oneOutput = [&]() {
    return wiring.numOutputs == 1 || fail("lowering requires exactly one output");
  };
  auto noArgs = [&]() {
    return (args.numIArgs == 0 && args.numTArgs == 0 && args.numBArgs == 0 &&
            args.numDArgs == 0 && args.numSArgs == 0) ||
           fail("unexpected op arguments are not consumed by the lowering");
  };
  auto exactInputs = [&](int count) {
    return wiring.numInputs == count || fail("unexpected data-input count");
  };

  const bool binary =
      nnapiOp == ANEURALNETWORKS_ADD || nnapiOp == ANEURALNETWORKS_SUB ||
      nnapiOp == ANEURALNETWORKS_MUL || nnapiOp == ANEURALNETWORKS_DIV ||
      nnapiOp == ANEURALNETWORKS_MAXIMUM || nnapiOp == ANEURALNETWORKS_MINIMUM ||
      nnapiOp == ANEURALNETWORKS_POW || nnapiOp == ANEURALNETWORKS_LESS ||
      nnapiOp == ANEURALNETWORKS_LESS_EQUAL || nnapiOp == ANEURALNETWORKS_GREATER ||
      nnapiOp == ANEURALNETWORKS_GREATER_EQUAL || nnapiOp == ANEURALNETWORKS_EQUAL ||
      nnapiOp == ANEURALNETWORKS_NOT_EQUAL || nnapiOp == ANEURALNETWORKS_LOGICAL_AND ||
      nnapiOp == ANEURALNETWORKS_LOGICAL_OR;
  if (binary) return exactInputs(2) && oneOutput() && noArgs();

  const bool unary =
      nnapiOp == ANEURALNETWORKS_RELU || nnapiOp == ANEURALNETWORKS_RELU6 ||
      nnapiOp == ANEURALNETWORKS_LOGISTIC || nnapiOp == ANEURALNETWORKS_TANH ||
      nnapiOp == ANEURALNETWORKS_FLOOR || nnapiOp == ANEURALNETWORKS_ABS ||
      nnapiOp == ANEURALNETWORKS_EXP || nnapiOp == ANEURALNETWORKS_LOG ||
      nnapiOp == ANEURALNETWORKS_NEG || nnapiOp == ANEURALNETWORKS_SQRT ||
      nnapiOp == ANEURALNETWORKS_RSQRT || nnapiOp == ANEURALNETWORKS_SIN;
  if (unary) return exactInputs(1) && oneOutput() && noArgs();

  if (nnapiOp == ANEURALNETWORKS_LOGICAL_NOT)
    return exactInputs(1) && oneOutput() && noArgs();
  if (nnapiOp == ANEURALNETWORKS_SELECT)
    return exactInputs(3) && oneOutput() && noArgs();

  if (nnapiOp == ANEURALNETWORKS_SOFTMAX) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numIArgs != 0 || args.numBArgs != 0 || args.numDArgs != 0 ||
        args.numSArgs != 0 || args.numTArgs != 0)
      return fail("softmax lowering accepts only default-axis softmax without parameters");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_BATCH_MATMUL) {
    if (!exactInputs(2) || !oneOutput()) return false;
    if (args.numBArgs > 2 || args.numIArgs != 0 || args.numTArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("batch matmul lowering accepts at most two transpose flags");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_FULLY_CONNECTED)
    return exactInputs(3) && oneOutput() && noArgs();

  if (nnapiOp == ANEURALNETWORKS_CONCATENATION) {
    if (wiring.numInputs < 2 || !oneOutput()) return false;
    if (args.numIArgs != 1 || args.numTArgs != 0 || args.numBArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("concat lowering requires exactly one integer axis");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_RESHAPE) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numTArgs != 0 || args.numBArgs != 0 || args.numDArgs != 0 ||
        args.numSArgs != 0)
      return fail("reshape lowering cannot consume non-integer parameters");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_TRANSPOSE) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numIArgs <= 0 || args.numTArgs != 0 || args.numBArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("transpose lowering requires an explicit permutation");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_GATHER) {
    if (!exactInputs(2) || !oneOutput()) return false;
    if (args.numIArgs != 1 || args.numTArgs != 0 || args.numBArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("gather lowering requires exactly one integer axis");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_SQUEEZE) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numTArgs != 0 || args.numBArgs != 0 || args.numDArgs != 0 ||
        args.numSArgs != 0)
      return fail("squeeze lowering accepts only integer axes");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_EXPAND_DIMS) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numIArgs != 1 || args.numTArgs != 0 || args.numBArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("expand_dims lowering requires exactly one integer axis");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_MEAN || nnapiOp == ANEURALNETWORKS_REDUCE_SUM ||
      nnapiOp == ANEURALNETWORKS_REDUCE_MAX || nnapiOp == ANEURALNETWORKS_REDUCE_MIN ||
      nnapiOp == ANEURALNETWORKS_REDUCE_PROD || nnapiOp == ANEURALNETWORKS_REDUCE_ANY ||
      nnapiOp == ANEURALNETWORKS_REDUCE_ALL) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numIArgs <= 0 || args.numBArgs > 1 || args.numTArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("reduce lowering requires explicit axes and an optional keep-dims flag");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_ARGMAX || nnapiOp == ANEURALNETWORKS_ARGMIN) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numIArgs != 1 || args.numTArgs != 0 || args.numBArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("arg reduction lowering requires exactly one integer axis");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_TILE) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numIArgs <= 0 || args.numTArgs != 0 || args.numBArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("tile lowering requires integer multiples");
    return true;
  }

  if (nnapiOp == ANEURALNETWORKS_SPACE_TO_DEPTH ||
      nnapiOp == ANEURALNETWORKS_DEPTH_TO_SPACE) {
    if (!exactInputs(1) || !oneOutput()) return false;
    if (args.numIArgs != 1 || args.numTArgs != 0 || args.numBArgs != 0 ||
        args.numDArgs != 0 || args.numSArgs != 0)
      return fail("space/depth lowering requires one integer block size");
    return true;
  }

  // These mappings are retained for explicit replay and future lowerers, but
  // their current parameter forms are not byte-for-byte equivalent to nd4j
  // (layout, dilation, masks, runtime shape tensors, or output-type operands).
  // Rejecting them here prevents a vendor compiler from receiving a plausible
  // but semantically wrong operation.
  reason = "mapped operation has no verified concrete NNAPI parameter contract";
  return false;
}

// ─── Construction / singleton ───────────────────────────────────────────────

bool NnapiGraphBackend::resolveRequiredAcceleratorDevice() {
#if defined(SD_NNAPI_ACCELERATOR_ONLY) && defined(__ANDROID_API__) && __ANDROID_API__ >= 29
  requiredDeviceName_ = kRequiredNnapiAcceleratorDevice;

  uint32_t deviceCount = 0;
  int result = ANeuralNetworks_getDeviceCount(&deviceCount);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    DSP_DIAG(BACKEND,
             "NNAPI_DEVICE_SELECTION_FAILED required=%s getDeviceCount=%d",
             requiredDeviceName_.c_str(), result);
    return false;
  }

  for (uint32_t deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
    const ANeuralNetworksDevice* device = nullptr;
    if (ANeuralNetworks_getDevice(deviceIndex, &device) != ANEURALNETWORKS_NO_ERROR ||
        device == nullptr) {
      continue;
    }

    const char* deviceName = nullptr;
    const char* deviceVersion = nullptr;
    int32_t deviceType = ANEURALNETWORKS_DEVICE_UNKNOWN;
    int64_t featureLevel = 0;
    if (ANeuralNetworksDevice_getName(device, &deviceName) != ANEURALNETWORKS_NO_ERROR ||
        deviceName == nullptr ||
        ANeuralNetworksDevice_getType(device, &deviceType) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksDevice_getFeatureLevel(device, &featureLevel) !=
            ANEURALNETWORKS_NO_ERROR) {
      continue;
    }
    (void)ANeuralNetworksDevice_getVersion(device, &deviceVersion);

    DSP_DIAG(BACKEND,
             "NNAPI_DEVICE_DISCOVERED index=%u name=%s type=%d version=%s feature_level=%lld",
             deviceIndex, deviceName, deviceType,
             deviceVersion != nullptr ? deviceVersion : "unknown",
             static_cast<long long>(featureLevel));

    if (toLower(deviceName) != toLower(requiredDeviceName_) ||
        deviceType != ANEURALNETWORKS_DEVICE_ACCELERATOR) {
      continue;
    }

    requiredDevice_ = device;
    selectedDeviceName_ = deviceName;
    selectedDeviceVersion_ = deviceVersion != nullptr ? deviceVersion : "unknown";
    selectedDeviceType_ = deviceType;
    selectedDeviceFeatureLevel_ = featureLevel;
    DSP_DIAG(BACKEND,
             "NNAPI_DEVICE_SELECTED required=%s name=%s type=%d version=%s feature_level=%lld",
             requiredDeviceName_.c_str(), selectedDeviceName_.c_str(),
             selectedDeviceType_, selectedDeviceVersion_.c_str(),
             static_cast<long long>(selectedDeviceFeatureLevel_));
    return true;
  }

  DSP_DIAG(BACKEND,
           "NNAPI_DEVICE_SELECTION_FAILED required=%s discovered=%u reason=missing_accelerator",
           requiredDeviceName_.c_str(), deviceCount);
  return false;
#else
  requiredDeviceName_.clear();
  return true;
#endif
}

NnapiGraphBackend::NnapiGraphBackend() {
  apiLevel_ = getAndroidApiLevel();
  nnapiAvailable_ = (apiLevel_ >= 27);
  if (nnapiAvailable_ && !resolveRequiredAcceleratorDevice()) {
    nnapiAvailable_ = false;
  }
  if (nnapiAvailable_) {
    DSP_DIAG(BACKEND, "NnapiGraphBackend: NNAPI available (API level %d)", apiLevel_);
  } else {
    DSP_DIAG(BACKEND,
             "NnapiGraphBackend: NNAPI unavailable or required accelerator missing (API level %d)",
             apiLevel_);
  }
}

NnapiGraphBackend::~NnapiGraphBackend() {
  invalidateCache();
}

NnapiGraphBackend& NnapiGraphBackend::getInstance() {
  static NnapiGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new NnapiGraphBackend();
  });
  return *instance;
}

bool NnapiGraphBackend::isAvailable() const {
  return nnapiAvailable_;
}

bool NnapiGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_NNAPI ||
         request.executionMode == GraphExecutionMode::GEM_ARM_HYBRID ||
         request.executionMode == GraphExecutionMode::GEM_AUTO ||
         request.executionMode == GraphExecutionMode::GEM_PORTABLE_REPLAY;
}

int NnapiGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  if (request.executionMode == GraphExecutionMode::GEM_NNAPI ||
      request.executionMode == GraphExecutionMode::GEM_ARM_HYBRID) {
    return 1000;
  }
  return 300;
}

GraphBackendPlanningPolicy NnapiGraphBackend::planningPolicy(
    const GraphBackendRequest& request) const {
  GraphBackendPlanningPolicy policy;
  policy.requiresShapePrePass = true;
  policy.requiresSuccessfulShapePrePass = true;
  policy.precompileBeforeFirstExecution = true;
  policy.allowsShapeOnlyWarmup = true;
  policy.requiresCapabilityPartitioning = true;
  policy.requiresCompleteLowering =
      request.executionMode == GraphExecutionMode::GEM_NNAPI;
  policy.preferredMaxSegmentOps = kMaxNnapiSegmentOps;
  return policy;
}

// ─── Segment analysis ───────────────────────────────────────────────────────

bool NnapiGraphBackend::isSlotResolvable(NativeSlot* slots,
                                         int slotIndex) const {
  if (slots == nullptr || slotIndex < 0) return false;
  const int opCode = getNnapiOpCode(slots[slotIndex].ident.opName);
  if (opCode < 0 || apiLevel_ < getMinApiLevel(slots[slotIndex].ident.opName)) {
    return false;
  }

  std::string contractReason;
  if (!validateSlotContract(slots[slotIndex], opCode, contractReason)) {
    DSP_DIAG(BACKEND, "NNAPI admission rejected slot %d (%s): %s",
             slotIndex, slots[slotIndex].ident.opName.c_str(),
             contractReason.c_str());
    return false;
  }
  return true;
}

bool NnapiGraphBackend::canResolveSlot(const GraphBackendRequest& request,
                                       NativeSlot* slots, int slotIndex) {
  (void)request;
  return isSlotResolvable(slots, slotIndex);
}

bool NnapiGraphBackend::canResolveSegment(const GraphBackendRequest& request,
                                          NativeSlot* slots, int start,
                                          int end) {
  // ARM hybrid is still a compiler-required accelerator contract. Capability
  // partitioning may legitimately isolate a supported operation into a
  // one-slot segment (for example, at a boundary between NNAPI-supported and
  // MLIR/ACL-supported operations). NNAPI can lower that operation directly;
  // requiring a fused pair here leaves the segment unresolved before lowering
  // and makes the compiler-required seal fail even though the slot is valid.
  if (start == end &&
      (request.executionMode == GraphExecutionMode::GEM_NNAPI ||
       request.executionMode == GraphExecutionMode::GEM_ARM_HYBRID)) {
    return isSlotResolvable(slots, start);
  }
  return canFuseSegment(slots, start, end);
}

bool NnapiGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (end < start) return false;
  const int segmentOps = end - start + 1;
  if (segmentOps < 2) return false;
  if (segmentOps > kMaxNnapiSegmentOps) {
    DSP_DIAG(BACKEND,
             "NNAPI admission rejected seg[%d-%d]: %d ops exceeds bounded limit %d; explicit replay",
             start, end, segmentOps, kMaxNnapiSegmentOps);
    return false;
  }

  bool hasGather = false;
  for (int i = start; i <= end; i++) {
    if (!isSlotResolvable(slots, i)) return false;
    hasGather |= toLower(slots[i].ident.opName) == "gather";
  }

  // A gather has value-dependent indexing and is only verified here as a
  // standalone NNAPI operation. Do not advertise a fused range containing it:
  // the vendor compiler may reject the otherwise syntactically valid graph
  // (for example, the [gather,multiply] range seen on Tensor G3). Range-level
  // admission must describe a concrete lowering contract, not just a list of
  // individually supported op codes.
  if (hasGather) {
    DSP_DIAG(BACKEND,
             "NNAPI admission rejected seg[%d-%d]: gather is a standalone "
             "compiler segment boundary",
             start, end);
    return false;
  }
  return true;
}

// ─── Operand helpers ────────────────────────────────────────────────────────

bool NnapiGraphBackend::addOperand(ANeuralNetworksModel* model, NDArray* arr,
                                    uint32_t& nextOperand,
                                    std::vector<std::unique_ptr<NDArray>>& contiguousCopies,
                                    uint32_t* outIdx) {
  uint32_t idx = nextOperand++;

  // NNAPI requires contiguous memory. If the array is a view or has
  // non-trivial strides, dup() it into a contiguous copy.
  NDArray* contiguous = arr;
  if (!isDenseCOrder(arr)) {
    auto copy = std::make_unique<NDArray>(arr->dup('c'));
    contiguous = copy.get();
    contiguousCopies.push_back(std::move(copy));
  }

  auto dt = contiguous->dataType();
  int32_t nnapiType = toNnapiOperandType(dt);
  if (nnapiType < 0) {
    if (!DataTypeUtils::isR(dt)) {
      // INT64 and other non-floating types have no NNAPI representation. A
      // silent conversion to FLOAT32 would change integer semantics (exact
      // only below 2^24) — reject the operand so the whole segment falls
      // back instead.
      DSP_DIAG(COMPILE,
               "NnapiGraphBackend: unsupported non-floating operand type %d "
               "cannot be represented in NNAPI",
               static_cast<int>(dt));
      return false;
    }
    // Unsupported floating type (e.g. BFLOAT16) — cast to FLOAT32.
    // This is a real value conversion, not a silent reinterpret.
    auto cast = std::make_unique<NDArray>(contiguous->cast(DataType::FLOAT32));
    contiguous = cast.get();
    contiguousCopies.push_back(std::move(cast));
    nnapiType = ANEURALNETWORKS_TENSOR_FLOAT32;
  }

  auto shape = contiguous->shapeOf();
  int rank = contiguous->rankOf();

  // Scalars: NNAPI doesn't support rank-0 tensors, represent as rank-1 with dim=1
  std::vector<uint32_t> dims;
  if (rank == 0) {
    dims.push_back(1);
  } else {
    dims.resize(rank);
    for (int d = 0; d < rank; d++) {
      dims[d] = static_cast<uint32_t>(shape[d]);
    }
  }

  ANeuralNetworksOperandType operandType;
  operandType.type = nnapiType;
  operandType.dimensionCount = static_cast<uint32_t>(dims.size());
  operandType.dimensions = dims.data();
  operandType.scale = 0.0f;
  operandType.zeroPoint = 0;

  const int result = ANeuralNetworksModel_addOperand(model, &operandType);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    DSP_DIAG(COMPILE, "NnapiGraphBackend: addOperand failed: %d", result);
    return false;
  }
  *outIdx = idx;
  return true;
}

uint32_t NnapiGraphBackend::addScalarOperand(ANeuralNetworksModel* model,
                                              int32_t value, uint32_t& nextOperand) {
  uint32_t idx = nextOperand++;

  ANeuralNetworksOperandType type;
  type.type = ANEURALNETWORKS_INT32;
  type.dimensionCount = 0;
  type.dimensions = nullptr;
  type.scale = 0.0f;
  type.zeroPoint = 0;

  ANeuralNetworksModel_addOperand(model, &type);
  ANeuralNetworksModel_setOperandValue(model, idx, &value, sizeof(int32_t));
  return idx;
}

uint32_t NnapiGraphBackend::addFloatOperand(ANeuralNetworksModel* model,
                                             float value, uint32_t& nextOperand) {
  uint32_t idx = nextOperand++;

  ANeuralNetworksOperandType type;
  type.type = ANEURALNETWORKS_FLOAT32;
  type.dimensionCount = 0;
  type.dimensions = nullptr;
  type.scale = 0.0f;
  type.zeroPoint = 0;

  ANeuralNetworksModel_addOperand(model, &type);
  ANeuralNetworksModel_setOperandValue(model, idx, &value, sizeof(float));
  return idx;
}

uint32_t NnapiGraphBackend::addBoolOperand(ANeuralNetworksModel* model,
                                            bool value, uint32_t& nextOperand) {
  uint32_t idx = nextOperand++;

  // NNAPI BOOL8 scalar
  ANeuralNetworksOperandType type;
  type.type = ANEURALNETWORKS_BOOL;
  type.dimensionCount = 0;
  type.dimensions = nullptr;
  type.scale = 0.0f;
  type.zeroPoint = 0;

  ANeuralNetworksModel_addOperand(model, &type);
  int8_t bval = value ? 1 : 0;
  ANeuralNetworksModel_setOperandValue(model, idx, &bval, sizeof(int8_t));
  return idx;
}

uint32_t NnapiGraphBackend::addIntVectorOperand(ANeuralNetworksModel* model,
                                                 const LongType* data, int count,
                                                 uint32_t& nextOperand,
                                                 std::vector<std::vector<int32_t>>& vectorStorage) {
  uint32_t idx = nextOperand++;

  // Convert LongType (int64) to int32 and store persistently
  vectorStorage.emplace_back(count);
  auto& vec = vectorStorage.back();
  for (int i = 0; i < count; i++) {
    vec[i] = static_cast<int32_t>(data[i]);
  }

  uint32_t dims[1] = {static_cast<uint32_t>(count)};
  ANeuralNetworksOperandType type;
  type.type = ANEURALNETWORKS_TENSOR_INT32;
  type.dimensionCount = 1;
  type.dimensions = dims;
  type.scale = 0.0f;
  type.zeroPoint = 0;

  ANeuralNetworksModel_addOperand(model, &type);
  ANeuralNetworksModel_setOperandValue(model, idx, vec.data(),
                                        count * sizeof(int32_t));
  return idx;
}

uint32_t NnapiGraphBackend::addShapeOperand(ANeuralNetworksModel* model, NDArray* arr,
                                             uint32_t& nextOperand,
                                             std::vector<std::vector<int32_t>>& vectorStorage) {
  int rank = arr->rankOf();
  auto shape = arr->shapeOf();

  vectorStorage.emplace_back(rank);
  auto& vec = vectorStorage.back();
  for (int d = 0; d < rank; d++) {
    vec[d] = static_cast<int32_t>(shape[d]);
  }

  uint32_t idx = nextOperand++;
  uint32_t dims[1] = {static_cast<uint32_t>(rank)};

  ANeuralNetworksOperandType type;
  type.type = ANEURALNETWORKS_TENSOR_INT32;
  type.dimensionCount = 1;
  type.dimensions = dims;
  type.scale = 0.0f;
  type.zeroPoint = 0;

  ANeuralNetworksModel_addOperand(model, &type);
  ANeuralNetworksModel_setOperandValue(model, idx, vec.data(),
                                        rank * sizeof(int32_t));
  return idx;
}

// ─── Implicit parameter injection ───────────────────────────────────────────
//
// Each NNAPI op has a fixed signature. The nd4j slot only provides the data
// tensor operands via inputSourceIndices. NNAPI also requires scalar/tensor
// parameters (padding mode, strides, axes, etc.) that come from iArgs/tArgs/bArgs.
// This function injects those extra operands.

bool NnapiGraphBackend::addImplicitParams(ANeuralNetworksModel* model, NativeSlot& slot,
                                           int nnapiOp, std::vector<uint32_t>& inputOperands,
                                           uint32_t& nextOperand,
                                           NDArray** externalInputs, int numExternalInputs,
                                           NDArray** outputSlots, int totalOutputSlots,
                                           std::vector<std::vector<int32_t>>& vectorStorage) {
  // ── Binary arithmetic: ADD, SUB, MUL, DIV need fused activation code ──
  if (nnapiOp == ANEURALNETWORKS_ADD || nnapiOp == ANEURALNETWORKS_SUB ||
      nnapiOp == ANEURALNETWORKS_MUL || nnapiOp == ANEURALNETWORKS_DIV) {
    inputOperands.push_back(addScalarOperand(model, ANEURALNETWORKS_FUSED_NONE, nextOperand));
    return true;
  }

  // ── Softmax: beta must match the input tensor type — FLOAT16 for FLOAT16
  // inputs, FLOAT32 otherwise. The admission contract rejects softmax slots
  // carrying any tArgs (nd4j softmax has no beta parameter), so beta is
  // always the neutral 1.0. ──
  if (nnapiOp == ANEURALNETWORKS_SOFTMAX) {
    NDArray* softmaxInput = nullptr;
    if (slot.wiring.numInputs > 0 && slot.wiring.inputSourceIndices != nullptr) {
      const int srcIdx = slot.wiring.inputSourceIndices[0];
      if (srcIdx < 0) {
        const int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs != nullptr) {
          softmaxInput = externalInputs[extIdx];
        }
      } else if (srcIdx < totalOutputSlots && outputSlots != nullptr) {
        softmaxInput = outputSlots[srcIdx];
      }
    }
    if (softmaxInput != nullptr && softmaxInput->dataType() == DataType::HALF) {
      // NNAPI requires the beta scalar to be FLOAT16 when the input is
      // FLOAT16. A FLOAT32 beta on a FLOAT16 input is a model-contract
      // violation that vendors may reject or misinterpret.
      const uint32_t betaIdx = nextOperand++;
      ANeuralNetworksOperandType type;
      type.type = ANEURALNETWORKS_FLOAT16;
      type.dimensionCount = 0;
      type.dimensions = nullptr;
      type.scale = 0.0f;
      type.zeroPoint = 0;
      ANeuralNetworksModel_addOperand(model, &type);
      const float16 beta16(1.0f);
      ANeuralNetworksModel_setOperandValue(model, betaIdx, &beta16, sizeof(beta16));
      inputOperands.push_back(betaIdx);
    } else {
      inputOperands.push_back(addFloatOperand(model, 1.0f, nextOperand));
    }
    return true;
  }

  // ── Fully connected: needs fused activation code ──
  if (nnapiOp == ANEURALNETWORKS_FULLY_CONNECTED) {
    inputOperands.push_back(addScalarOperand(model, ANEURALNETWORKS_FUSED_NONE, nextOperand));
    return true;
  }

  // ── Concatenation: needs axis ──
  if (nnapiOp == ANEURALNETWORKS_CONCATENATION) {
    int axis = 0;
    if (slot.args.numIArgs > 0 && slot.args.iArgs) axis = static_cast<int>(slot.args.iArgs[0]);
    inputOperands.push_back(addScalarOperand(model, axis, nextOperand));
    return true;
  }

  // ── CONV_2D: input, filter, bias, padLeft, padRight, padTop, padBottom,
  //             strideW, strideH, activation
  // nd4j conv2d iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, ...]
  if (nnapiOp == ANEURALNETWORKS_CONV_2D) {
    int pH = 0, pW = 0, sH = 1, sW = 1;
    if (slot.args.numIArgs >= 6 && slot.args.iArgs) {
      sH = static_cast<int>(slot.args.iArgs[2]);
      sW = static_cast<int>(slot.args.iArgs[3]);
      pH = static_cast<int>(slot.args.iArgs[4]);
      pW = static_cast<int>(slot.args.iArgs[5]);
    }
    // NNAPI explicit padding: padLeft, padRight, padTop, padBottom
    inputOperands.push_back(addScalarOperand(model, pW, nextOperand));      // padLeft
    inputOperands.push_back(addScalarOperand(model, pW, nextOperand));      // padRight
    inputOperands.push_back(addScalarOperand(model, pH, nextOperand));      // padTop
    inputOperands.push_back(addScalarOperand(model, pH, nextOperand));      // padBottom
    inputOperands.push_back(addScalarOperand(model, sW, nextOperand));      // strideW
    inputOperands.push_back(addScalarOperand(model, sH, nextOperand));      // strideH
    inputOperands.push_back(addScalarOperand(model, ANEURALNETWORKS_FUSED_NONE, nextOperand));
    return true;
  }

  // ── DEPTHWISE_CONV_2D: same as CONV_2D + depth_multiplier ──
  if (nnapiOp == ANEURALNETWORKS_DEPTHWISE_CONV_2D) {
    int pH = 0, pW = 0, sH = 1, sW = 1, dm = 1;
    if (slot.args.numIArgs >= 6 && slot.args.iArgs) {
      sH = static_cast<int>(slot.args.iArgs[2]);
      sW = static_cast<int>(slot.args.iArgs[3]);
      pH = static_cast<int>(slot.args.iArgs[4]);
      pW = static_cast<int>(slot.args.iArgs[5]);
    }
    if (slot.args.numIArgs >= 9 && slot.args.iArgs) {
      dm = static_cast<int>(slot.args.iArgs[8]);
      if (dm < 1) dm = 1;
    }
    inputOperands.push_back(addScalarOperand(model, pW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, sW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, sH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, dm, nextOperand));
    inputOperands.push_back(addScalarOperand(model, ANEURALNETWORKS_FUSED_NONE, nextOperand));
    return true;
  }

  // ── TRANSPOSE_CONV_2D: padding, strideW, strideH, activation ──
  if (nnapiOp == ANEURALNETWORKS_TRANSPOSE_CONV_2D) {
    int pH = 0, pW = 0, sH = 1, sW = 1;
    if (slot.args.numIArgs >= 6 && slot.args.iArgs) {
      sH = static_cast<int>(slot.args.iArgs[2]);
      sW = static_cast<int>(slot.args.iArgs[3]);
      pH = static_cast<int>(slot.args.iArgs[4]);
      pW = static_cast<int>(slot.args.iArgs[5]);
    }
    inputOperands.push_back(addScalarOperand(model, pW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, sW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, sH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, ANEURALNETWORKS_FUSED_NONE, nextOperand));
    return true;
  }

  // ── AVERAGE_POOL_2D / MAX_POOL_2D:
  //    padLeft, padRight, padTop, padBottom, strideW, strideH,
  //    filterW, filterH, activation
  // nd4j pooling iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, ...]
  if (nnapiOp == ANEURALNETWORKS_AVERAGE_POOL_2D ||
      nnapiOp == ANEURALNETWORKS_MAX_POOL_2D) {
    int kH = 1, kW = 1, sH = 1, sW = 1, pH = 0, pW = 0;
    if (slot.args.numIArgs >= 6 && slot.args.iArgs) {
      kH = static_cast<int>(slot.args.iArgs[0]);
      kW = static_cast<int>(slot.args.iArgs[1]);
      sH = static_cast<int>(slot.args.iArgs[2]);
      sW = static_cast<int>(slot.args.iArgs[3]);
      pH = static_cast<int>(slot.args.iArgs[4]);
      pW = static_cast<int>(slot.args.iArgs[5]);
    }
    inputOperands.push_back(addScalarOperand(model, pW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, pH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, sW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, sH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, kW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, kH, nextOperand));
    inputOperands.push_back(addScalarOperand(model, ANEURALNETWORKS_FUSED_NONE, nextOperand));
    return true;
  }

  // ── RESHAPE: needs new shape as a 1D INT32 tensor ──
  if (nnapiOp == ANEURALNETWORKS_RESHAPE) {
    // The target shape comes from iArgs or from the output array's actual shape
    if (slot.args.numIArgs > 0 && slot.args.iArgs) {
      inputOperands.push_back(
          addIntVectorOperand(model, slot.args.iArgs, slot.args.numIArgs, nextOperand, vectorStorage));
    } else {
      // Infer from output shape
      int outIdx = slot.wiring.outputSlotIndices[0];
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]) {
        inputOperands.push_back(
            addShapeOperand(model, outputSlots[outIdx], nextOperand, vectorStorage));
      } else {
        return false;
      }
    }
    return true;
  }

  // ── TRANSPOSE: needs perm tensor ──
  if (nnapiOp == ANEURALNETWORKS_TRANSPOSE) {
    if (slot.args.numIArgs > 0 && slot.args.iArgs) {
      inputOperands.push_back(
          addIntVectorOperand(model, slot.args.iArgs, slot.args.numIArgs, nextOperand, vectorStorage));
    }
    // If no iArgs, NNAPI reverses dimensions by default (no perm operand needed)
    return true;
  }

  // ── GATHER: NNAPI order is [input, axis, indices]. SameDiff
  // exposes its data inputs as [input, indices], so the scalar axis must be
  // inserted between them rather than appended after the indices tensor.
  if (nnapiOp == ANEURALNETWORKS_GATHER) {
    if (inputOperands.size() != 2) return false;
    int axis = 0;
    if (slot.args.numIArgs > 0 && slot.args.iArgs) axis = static_cast<int>(slot.args.iArgs[0]);
    const uint32_t indicesOperand = inputOperands[1];
    inputOperands.resize(1);
    inputOperands.push_back(addScalarOperand(model, axis, nextOperand));
    inputOperands.push_back(indicesOperand);
    return true;
  }

  // ── SQUEEZE: needs axes as 1D tensor (optional) ──
  if (nnapiOp == ANEURALNETWORKS_SQUEEZE) {
    if (slot.args.numIArgs > 0 && slot.args.iArgs) {
      inputOperands.push_back(
          addIntVectorOperand(model, slot.args.iArgs, slot.args.numIArgs, nextOperand, vectorStorage));
    }
    // If no iArgs, NNAPI squeezes all size-1 dimensions
    return true;
  }

  // ── EXPAND_DIMS: needs axis scalar ──
  if (nnapiOp == ANEURALNETWORKS_EXPAND_DIMS) {
    int axis = 0;
    if (slot.args.numIArgs > 0 && slot.args.iArgs) axis = static_cast<int>(slot.args.iArgs[0]);
    inputOperands.push_back(addScalarOperand(model, axis, nextOperand));
    return true;
  }

  // ── MEAN / REDUCE_*: needs axes tensor + keepDims bool ──
  if (nnapiOp == ANEURALNETWORKS_MEAN ||
      nnapiOp == ANEURALNETWORKS_REDUCE_SUM ||
      nnapiOp == ANEURALNETWORKS_REDUCE_MAX ||
      nnapiOp == ANEURALNETWORKS_REDUCE_MIN ||
      nnapiOp == ANEURALNETWORKS_REDUCE_PROD ||
      nnapiOp == ANEURALNETWORKS_REDUCE_ANY ||
      nnapiOp == ANEURALNETWORKS_REDUCE_ALL) {
    // iArgs = axes to reduce along
    if (slot.args.numIArgs > 0 && slot.args.iArgs) {
      inputOperands.push_back(
          addIntVectorOperand(model, slot.args.iArgs, slot.args.numIArgs, nextOperand, vectorStorage));
    } else {
      // Reduce all — pass empty axes (NNAPI reduces all dims)
      vectorStorage.emplace_back();
      uint32_t aidx = nextOperand++;
      uint32_t dims[1] = {0};
      ANeuralNetworksOperandType type;
      type.type = ANEURALNETWORKS_TENSOR_INT32;
      type.dimensionCount = 1;
      type.dimensions = dims;
      type.scale = 0.0f;
      type.zeroPoint = 0;
      ANeuralNetworksModel_addOperand(model, &type);
      inputOperands.push_back(aidx);
    }
    // keep_dims: NNAPI MEAN requires an INT32 scalar, while REDUCE_* require
    // a BOOL scalar. Emitting BOOL for MEAN violates the model contract and
    // leaves acceptance up to driver leniency.
    bool keepDims = false;
    if (slot.args.numBArgs > 0 && slot.args.bArgs) keepDims = slot.args.bArgs[0];
    if (nnapiOp == ANEURALNETWORKS_MEAN) {
      inputOperands.push_back(addScalarOperand(model, keepDims ? 1 : 0, nextOperand));
    } else {
      inputOperands.push_back(addBoolOperand(model, keepDims, nextOperand));
    }
    return true;
  }

  // ── ARGMAX / ARGMIN: needs axis scalar ──
  if (nnapiOp == ANEURALNETWORKS_ARGMAX || nnapiOp == ANEURALNETWORKS_ARGMIN) {
    int axis = 0;
    if (slot.args.numIArgs > 0 && slot.args.iArgs) axis = static_cast<int>(slot.args.iArgs[0]);
    inputOperands.push_back(addScalarOperand(model, axis, nextOperand));
    return true;
  }

  // ── SPLIT: needs axis scalar, numSplits scalar ──
  if (nnapiOp == ANEURALNETWORKS_SPLIT) {
    int axis = 0;
    int numSplits = slot.wiring.numOutputs;
    if (slot.args.numIArgs > 0 && slot.args.iArgs) axis = static_cast<int>(slot.args.iArgs[0]);
    if (slot.args.numIArgs > 1 && slot.args.iArgs) numSplits = static_cast<int>(slot.args.iArgs[1]);
    inputOperands.push_back(addScalarOperand(model, axis, nextOperand));
    inputOperands.push_back(addScalarOperand(model, numSplits, nextOperand));
    return true;
  }

  // ── PAD: needs paddings 2D tensor ──
  if (nnapiOp == ANEURALNETWORKS_PAD) {
    // iArgs = [pad_before_d0, pad_after_d0, pad_before_d1, pad_after_d1, ...]
    if (slot.args.numIArgs >= 2 && slot.args.iArgs) {
      int rank = slot.args.numIArgs / 2;
      vectorStorage.emplace_back(rank * 2);
      auto& vec = vectorStorage.back();
      for (int i = 0; i < rank * 2; i++) {
        vec[i] = static_cast<int32_t>(slot.args.iArgs[i]);
      }

      uint32_t pidx = nextOperand++;
      uint32_t dims[2] = {static_cast<uint32_t>(rank), 2};
      ANeuralNetworksOperandType type;
      type.type = ANEURALNETWORKS_TENSOR_INT32;
      type.dimensionCount = 2;
      type.dimensions = dims;
      type.scale = 0.0f;
      type.zeroPoint = 0;
      ANeuralNetworksModel_addOperand(model, &type);
      ANeuralNetworksModel_setOperandValue(model, pidx, vec.data(),
                                            rank * 2 * sizeof(int32_t));
      inputOperands.push_back(pidx);
    } else {
      return false;
    }
    return true;
  }

  // ── TILE: needs multiples tensor ──
  if (nnapiOp == ANEURALNETWORKS_TILE) {
    if (slot.args.numIArgs > 0 && slot.args.iArgs) {
      inputOperands.push_back(
          addIntVectorOperand(model, slot.args.iArgs, slot.args.numIArgs, nextOperand, vectorStorage));
    } else {
      return false;
    }
    return true;
  }

  // ── STRIDED_SLICE: needs begin, end, strides tensors + masks ──
  if (nnapiOp == ANEURALNETWORKS_STRIDED_SLICE) {
    // nd4j strided_slice iArgs: [begin_d0..., end_d0..., strides_d0..., begin_mask, end_mask, ...]
    // We need to figure out the rank from the arrays
    int rank = 0;
    if (slot.args.numIArgs >= 3) {
      // Guess rank = numIArgs / 3 if no masks, or (numIArgs - extra) / 3
      // Safer: use iArgs count. Common layout: 3*rank values + optional masks
      // Try: if numIArgs >= 7 (rank=1: 3 + 4 masks, or rank=2: 6 + 1 mask)
      // The safest heuristic: subtract known scalar suffixes
      int numMasks = 0;
      if (slot.args.numIArgs > 3) {
        // StridedSlice has begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask
        // 5 mask scalars at the end
        if (slot.args.numIArgs > 5) {
          rank = (slot.args.numIArgs - 5) / 3;
          if (rank < 1) rank = 1;
          numMasks = 5;
        } else {
          rank = slot.args.numIArgs / 3;
        }
      } else {
        rank = 1;
      }

      if (rank > 0 && slot.args.numIArgs >= rank * 3) {
        inputOperands.push_back(
            addIntVectorOperand(model, slot.args.iArgs, rank, nextOperand, vectorStorage));
        inputOperands.push_back(
            addIntVectorOperand(model, slot.args.iArgs + rank, rank, nextOperand, vectorStorage));
        inputOperands.push_back(
            addIntVectorOperand(model, slot.args.iArgs + 2 * rank, rank, nextOperand, vectorStorage));

        int beginMask = 0, endMask = 0, shrinkAxisMask = 0;
        int base = 3 * rank;
        if (slot.args.numIArgs > base) beginMask = static_cast<int>(slot.args.iArgs[base]);
        if (slot.args.numIArgs > base + 1) endMask = static_cast<int>(slot.args.iArgs[base + 1]);
        if (slot.args.numIArgs > base + 4) shrinkAxisMask = static_cast<int>(slot.args.iArgs[base + 4]);

        inputOperands.push_back(addScalarOperand(model, beginMask, nextOperand));
        inputOperands.push_back(addScalarOperand(model, endMask, nextOperand));
        inputOperands.push_back(addScalarOperand(model, shrinkAxisMask, nextOperand));
        return true;
      }
    }
    return false;
  }

  // ── LOCAL_RESPONSE_NORMALIZATION: radius, bias, alpha, beta, axis ──
  if (nnapiOp == ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION) {
    int radius = 5;
    float bias = 1.0f, alpha = 1.0f, beta = 0.5f;
    if (slot.args.numIArgs >= 1 && slot.args.iArgs) radius = static_cast<int>(slot.args.iArgs[0]);
    if (slot.args.numTArgs >= 1 && slot.args.tArgs) bias = static_cast<float>(slot.args.tArgs[0]);
    if (slot.args.numTArgs >= 2 && slot.args.tArgs) alpha = static_cast<float>(slot.args.tArgs[1]);
    if (slot.args.numTArgs >= 3 && slot.args.tArgs) beta = static_cast<float>(slot.args.tArgs[2]);
    inputOperands.push_back(addScalarOperand(model, radius, nextOperand));
    inputOperands.push_back(addFloatOperand(model, bias, nextOperand));
    inputOperands.push_back(addFloatOperand(model, alpha, nextOperand));
    inputOperands.push_back(addFloatOperand(model, beta, nextOperand));
    return true;
  }

  // ── BATCH_MATMUL: adjX, adjY booleans ──
  if (nnapiOp == ANEURALNETWORKS_BATCH_MATMUL) {
    bool adjX = false, adjY = false;
    if (slot.args.numBArgs >= 1 && slot.args.bArgs) adjX = slot.args.bArgs[0];
    if (slot.args.numBArgs >= 2 && slot.args.bArgs) adjY = slot.args.bArgs[1];
    inputOperands.push_back(addBoolOperand(model, adjX, nextOperand));
    inputOperands.push_back(addBoolOperand(model, adjY, nextOperand));
    return true;
  }

  // ── RESIZE_BILINEAR / RESIZE_NEAREST_NEIGHBOR: output width, height ──
  if (nnapiOp == ANEURALNETWORKS_RESIZE_BILINEAR ||
      nnapiOp == ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR) {
    int newH = 0, newW = 0;
    if (slot.args.numIArgs >= 2 && slot.args.iArgs) {
      newH = static_cast<int>(slot.args.iArgs[0]);
      newW = static_cast<int>(slot.args.iArgs[1]);
    } else {
      // Infer from output shape (NHWC: [N, H, W, C])
      int outIdx = slot.wiring.outputSlotIndices[0];
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]) {
        auto* out = outputSlots[outIdx];
        if (out->rankOf() >= 3) {
          newH = static_cast<int>(out->shapeOf()[1]);
          newW = static_cast<int>(out->shapeOf()[2]);
        }
      }
    }
    if (newH <= 0 || newW <= 0) return false;
    inputOperands.push_back(addScalarOperand(model, newW, nextOperand));
    inputOperands.push_back(addScalarOperand(model, newH, nextOperand));
    return true;
  }

  // ── SPACE_TO_BATCH_ND / BATCH_TO_SPACE_ND: block_shape + paddings/crop ──
  if (nnapiOp == ANEURALNETWORKS_SPACE_TO_BATCH_ND ||
      nnapiOp == ANEURALNETWORKS_BATCH_TO_SPACE_ND) {
    // iArgs layout: [block_d0, block_d1, ..., pad/crop values...]
    // Typically rank-2 spatial: block_shape=[bH, bW], paddings=[[pH0,pH1],[pW0,pW1]]
    if (slot.args.numIArgs >= 2 && slot.args.iArgs) {
      int spatialRank = 2;
      inputOperands.push_back(
          addIntVectorOperand(model, slot.args.iArgs, spatialRank, nextOperand, vectorStorage));

      if (slot.args.numIArgs >= 2 + spatialRank * 2) {
        // Paddings/crops as 2D tensor
        vectorStorage.emplace_back(spatialRank * 2);
        auto& vec = vectorStorage.back();
        for (int i = 0; i < spatialRank * 2; i++) {
          vec[i] = static_cast<int32_t>(slot.args.iArgs[spatialRank + i]);
        }
        uint32_t pidx = nextOperand++;
        uint32_t dims[2] = {static_cast<uint32_t>(spatialRank), 2};
        ANeuralNetworksOperandType type;
        type.type = ANEURALNETWORKS_TENSOR_INT32;
        type.dimensionCount = 2;
        type.dimensions = dims;
        type.scale = 0.0f;
        type.zeroPoint = 0;
        ANeuralNetworksModel_addOperand(model, &type);
        ANeuralNetworksModel_setOperandValue(model, pidx, vec.data(),
                                              spatialRank * 2 * sizeof(int32_t));
        inputOperands.push_back(pidx);
      }
      return true;
    }
    return false;
  }

  // ── SPACE_TO_DEPTH / DEPTH_TO_SPACE: block_size ──
  if (nnapiOp == ANEURALNETWORKS_SPACE_TO_DEPTH ||
      nnapiOp == ANEURALNETWORKS_DEPTH_TO_SPACE) {
    int blockSize = 2;
    if (slot.args.numIArgs > 0 && slot.args.iArgs) blockSize = static_cast<int>(slot.args.iArgs[0]);
    inputOperands.push_back(addScalarOperand(model, blockSize, nextOperand));
    return true;
  }

  // ── CAST: needs output type operand ──
  if (nnapiOp == ANEURALNETWORKS_CAST) {
    // NNAPI CAST infers output type from the output operand type — no extra param needed
    return true;
  }

  // ── Ops that need no extra params ──
  // Unary: ABS, EXP, LOG, NEG, SQRT, RSQRT, SIN, FLOOR, RELU, RELU6, LOGISTIC, TANH
  // Binary: MAXIMUM, MINIMUM, POW
  // Comparison: LESS, LESS_EQUAL, GREATER, GREATER_EQUAL, EQUAL, NOT_EQUAL
  // Logical: LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT
  // Select: SELECT
  // Data: TILE (handled above), EXPAND_DIMS (handled above)
  return true;
}

// ─── Model building ─────────────────────────────────────────────────────────

bool NnapiGraphBackend::buildModel(ANeuralNetworksModel* model, CompiledModel& compiled,
                                    NativeSlot* slots, int startSlot, int endSlot,
                                    NDArray** externalInputs, int numExternalInputs,
                                    NDArray** outputSlots, int totalOutputSlots,
                                    int totalSlots,
                                    const int* requestedOutputSlotIndices,
                                    int numRequestedOutputs,
                                    std::vector<int>& operationSourceSlots) {
  uint32_t nextOperand = 0;
  operationSourceSlots.clear();

  // Persistent storage for vector operand data (must outlive ANeuralNetworksModel_finish)
  std::vector<std::vector<int32_t>> vectorStorage;
  // Persistent storage for contiguous copies of non-contiguous arrays
  std::vector<std::unique_ptr<NDArray>> contiguousCopies;

  // Track which source indices map to which NNAPI operands
  std::unordered_map<int, uint32_t> sourceToOperand;

  // Identify all outputs produced within this segment
  std::unordered_set<int> segmentOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      segmentOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  // Use the same backend-neutral dataflow contract as other graph backends.
  // NNAPI-specific operand construction begins only after visibility is resolved.
  const auto externalOutputSet = computeExternallyVisibleOutputSlots(
      slots, startSlot, endSlot, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);

  // Resolve a wiring source index to its NDArray (external input or output
  // slot). Shared by operand construction and shape-contract validation.
  auto resolveSourceArray = [&](int srcIdx) -> NDArray* {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs && externalInputs) return externalInputs[extIdx];
      return nullptr;
    }
    if (srcIdx < totalOutputSlots && outputSlots) return outputSlots[srcIdx];
    return nullptr;
  };

  auto bindingDataType = [&](DataType sourceType) -> DataType {
    if (isNnapiSupportedType(sourceType)) return sourceType;
    return DataTypeUtils::isR(sourceType) ? DataType::FLOAT32 : DataType::UNKNOWN;
  };

  auto addOperandDescriptor = [&](DataType dataType,
                                  const std::vector<LongType>& dimensions,
                                  uint32_t* outIdx) -> bool {
    const int32_t nnapiType = toNnapiOperandType(dataType);
    if (nnapiType < 0 || dimensions.empty() || outIdx == nullptr) return false;
    std::vector<uint32_t> nnapiDimensions(dimensions.size());
    for (size_t dimension = 0; dimension < dimensions.size(); ++dimension) {
      const LongType value = dimensions[dimension];
      if (value <= 0 ||
          static_cast<unsigned long long>(value) >
              static_cast<unsigned long long>(std::numeric_limits<uint32_t>::max())) {
        return false;
      }
      nnapiDimensions[dimension] = static_cast<uint32_t>(value);
    }
    ANeuralNetworksOperandType operandType;
    operandType.type = nnapiType;
    operandType.dimensionCount = static_cast<uint32_t>(nnapiDimensions.size());
    operandType.dimensions = nnapiDimensions.data();
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    const uint32_t operand = nextOperand++;
    const int result = ANeuralNetworksModel_addOperand(model, &operandType);
    if (result != ANEURALNETWORKS_NO_ERROR) return false;
    *outIdx = operand;
    return true;
  };

  std::unordered_map<int, EmbeddingLookupContract> embeddingContracts;
  std::unordered_set<int> embeddingLookupSources;
  for (int slotIndex = startSlot; slotIndex <= endSlot; ++slotIndex) {
    if (getNnapiOpCode(slots[slotIndex].ident.opName) !=
            ANEURALNETWORKS_GATHER ||
        slots[slotIndex].wiring.numInputs != 2 ||
        slots[slotIndex].wiring.numOutputs != 1) {
      continue;
    }
    const int axis =
        slots[slotIndex].args.numIArgs > 0 && slots[slotIndex].args.iArgs != nullptr
            ? static_cast<int>(slots[slotIndex].args.iArgs[0])
            : 0;
    NDArray* values = resolveSourceArray(
        slots[slotIndex].wiring.inputSourceIndices[0]);
    NDArray* lookups = resolveSourceArray(
        slots[slotIndex].wiring.inputSourceIndices[1]);
    NDArray* output = resolveSourceArray(
        slots[slotIndex].wiring.outputSlotIndices[0]);
    const auto contract =
        getEmbeddingLookupContract(values, lookups, output, axis);
    if (contract.valid) {
      embeddingContracts.emplace(slotIndex, contract);
      embeddingLookupSources.insert(
          slots[slotIndex].wiring.inputSourceIndices[1]);
    }
  }

  // Phase 1: Add input operands (external inputs + pre-segment intermediates)
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
      if (sourceToOperand.count(srcIdx)) continue;

      bool isExternal = (srcIdx < 0);
      bool isPreSegment = (!isExternal && !segmentOutputs.count(srcIdx));

      if (isExternal || isPreSegment) {
        NDArray* arr = resolveSourceArray(srcIdx);
        if (!arr) {
          DSP_DIAG(COMPILE, "NnapiGraphBackend: null input array for source %d at slot %d", srcIdx, i);
          return false;
        }

        const bool embeddingLookupInput = embeddingLookupSources.count(srcIdx) != 0;
        const DataType compiledDataType =
            embeddingLookupInput ? DataType::INT32 : bindingDataType(arr->dataType());
        uint32_t opIdx = 0;
        std::vector<LongType> dimensions(static_cast<size_t>(arr->rankOf()));
        for (int dimension = 0; dimension < arr->rankOf(); dimension++) {
          dimensions[static_cast<size_t>(dimension)] = arr->sizeAt(dimension);
        }
        const bool operandAdded = embeddingLookupInput
                                      ? addOperandDescriptor(compiledDataType, dimensions, &opIdx)
                                      : addOperand(model, arr, nextOperand,
                                                   contiguousCopies, &opIdx);
        if (!operandAdded) {
          DSP_DIAG(COMPILE,
                   "NnapiGraphBackend: unsupported input operand type for "
                   "source %d at slot %d",
                   srcIdx, i);
          return false;
        }
        sourceToOperand[srcIdx] = opIdx;
        compiled.inputMappings.push_back(
            {srcIdx, opIdx, false, arr->dataType(), compiledDataType,
             std::move(dimensions)});
      }
    }
  }

  // Phase 2: Process each slot — add output operands and NNAPI operations
  for (int i = startSlot; i <= endSlot; i++) {
    int nnapiOp = getNnapiOpCode(slots[i].ident.opName);
    if (nnapiOp < 0) {
      DSP_DIAG(FALLBACK, "NnapiGraphBackend: unmappable op '%s' at slot %d",
                slots[i].ident.opName.c_str(), i);
      return false;
    }

    std::string contractReason;
    if (!validateSlotContract(slots[i], nnapiOp, contractReason)) {
      DSP_DIAG(FALLBACK,
               "NnapiGraphBackend: refusing unverified contract for '%s' at slot %d: %s",
               slots[i].ident.opName.c_str(), i, contractReason.c_str());
      return false;
    }

    // Collect input operand indices for this op
    std::vector<uint32_t> inputOperands;
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
      auto it = sourceToOperand.find(srcIdx);
      if (it == sourceToOperand.end()) {
        DSP_DIAG(COMPILE, "NnapiGraphBackend: missing operand for source %d at slot %d input %d",
                  srcIdx, i, inp);
        return false;
      }
      inputOperands.push_back(it->second);
    }

    // Tensor G3 does not compile the general NNAPI GATHER operation. Qwen's
    // axis-0 embedding gather is equivalent to EMBEDDING_LOOKUP, but SDX supplies
    // batched INT64 token ids ([1, sequence]) while NNAPI requires one-dimensional
    // INT32 lookups. The verified contract below binds the ids as INT32 and, for
    // batched ids, emits reshape -> embedding lookup -> reshape.
    int operationOp = nnapiOp;
    const auto embeddingContractIt = embeddingContracts.find(i);
    const bool embeddingLookup = embeddingContractIt != embeddingContracts.end();
    bool batchedEmbeddingLookup = false;
    NDArray* gatherValues = nullptr;
    NDArray* gatherIndices = nullptr;
    NDArray* gatherOutput = nullptr;
    if (embeddingLookup) {
      gatherValues = resolveSourceArray(slots[i].wiring.inputSourceIndices[0]);
      gatherIndices = resolveSourceArray(slots[i].wiring.inputSourceIndices[1]);
      gatherOutput = resolveSourceArray(slots[i].wiring.outputSlotIndices[0]);
      operationOp = ANEURALNETWORKS_EMBEDDING_LOOKUP;
      std::swap(inputOperands[0], inputOperands[1]);
      const auto& contract = embeddingContractIt->second;
      batchedEmbeddingLookup = contract.flattenLookups;
      if (batchedEmbeddingLookup) {
        const LongType flattenedLookupShape[1] = {contract.lookupCount};
        const uint32_t shapeOperand = addIntVectorOperand(
            model, flattenedLookupShape, 1, nextOperand, vectorStorage);
        uint32_t flattenedLookupOperand = 0;
        if (!addOperandDescriptor(DataType::INT32, {contract.lookupCount},
                                  &flattenedLookupOperand)) {
          DSP_DIAG(COMPILE,
                   "NnapiGraphBackend: failed to declare flattened embedding "
                   "lookups at slot %d",
                   i);
          return false;
        }
        const uint32_t reshapeInputs[2] = {inputOperands[0], shapeOperand};
        const uint32_t reshapeOutputs[1] = {flattenedLookupOperand};
        const int reshapeResult = ANeuralNetworksModel_addOperation(
            model, ANEURALNETWORKS_RESHAPE, 2, reshapeInputs, 1,
            reshapeOutputs);
        if (reshapeResult != ANEURALNETWORKS_NO_ERROR) {
          DSP_DIAG(COMPILE,
                   "NnapiGraphBackend: failed to flatten embedding lookups at "
                   "slot %d: %d",
                   i, reshapeResult);
          return false;
        }
        operationSourceSlots.push_back(i);
        inputOperands[0] = flattenedLookupOperand;
      }
      DSP_DIAG(COMPILE,
               "NNAPI_GATHER_LOWERING slot=%d axis=0 values_rank=%d "
               "lookups_rank=%d lookups=%lld binding=INT32 flatten=%d "
               "lowered=EMBEDDING_LOOKUP",
               i, gatherValues->rankOf(), gatherIndices->rankOf(),
               static_cast<long long>(contract.lookupCount),
               batchedEmbeddingLookup ? 1 : 0);
    }

    // NNAPI FULLY_CONNECTED computes input @ weights^T with weights of shape
    // [num_units, input_size]. nd4j xw_plus_b computes x @ w + b with w of
    // shape [input, output]. Feeding w directly would silently compute the
    // transposed product, so insert an NNAPI TRANSPOSE on the weight operand
    // and hand the fully-connected op w^T instead.
    if (nnapiOp == ANEURALNETWORKS_FULLY_CONNECTED) {
      if (inputOperands.size() != 3) {
        DSP_DIAG(COMPILE,
                 "NnapiGraphBackend: fully-connected lowering requires 3 data inputs at slot %d",
                 i);
        return false;
      }
      NDArray* fcInput = resolveSourceArray(slots[i].wiring.inputSourceIndices[0]);
      NDArray* fcWeight = resolveSourceArray(slots[i].wiring.inputSourceIndices[1]);
      NDArray* fcBias = resolveSourceArray(slots[i].wiring.inputSourceIndices[2]);
      if (fcInput == nullptr || fcWeight == nullptr || fcBias == nullptr) {
        DSP_DIAG(COMPILE, "NnapiGraphBackend: null fully-connected input at slot %d", i);
        return false;
      }
      // NNAPI requires input rank >= 2, 2-D weights, and a 1-D bias. Anything
      // else must fall back rather than reach vendor compilation with
      // ambiguous shapes.
      if (fcInput->rankOf() < 2 || fcWeight->rankOf() != 2 ||
          fcBias->rankOf() != 1) {
        DSP_DIAG(COMPILE,
                 "NnapiGraphBackend: fully-connected shape contract violated "
                 "at slot %d (input rank %d, weight rank %d, bias rank %d)",
                 i, fcInput->rankOf(), fcWeight->rankOf(), fcBias->rankOf());
        return false;
      }

      const uint32_t weightOperand = inputOperands[1];
      // The weight operand was declared by addOperand, which promotes
      // unsupported floating types to FLOAT32.
      int32_t weightType = toNnapiOperandType(fcWeight->dataType());
      if (weightType < 0) weightType = ANEURALNETWORKS_TENSOR_FLOAT32;

      const uint32_t transposedOperand = nextOperand++;
      ANeuralNetworksOperandType transposedType;
      transposedType.type = weightType;
      transposedType.dimensionCount = 2;
      const uint32_t swappedDims[2] = {
          static_cast<uint32_t>(fcWeight->sizeAt(1)),
          static_cast<uint32_t>(fcWeight->sizeAt(0))};
      transposedType.dimensions = swappedDims;
      transposedType.scale = 0.0f;
      transposedType.zeroPoint = 0;
      const int transposedResult =
          ANeuralNetworksModel_addOperand(model, &transposedType);
      if (transposedResult != ANEURALNETWORKS_NO_ERROR) {
        DSP_DIAG(COMPILE,
                 "NnapiGraphBackend: failed to add transposed weight operand "
                 "at slot %d: %d",
                 i, transposedResult);
        return false;
      }

      static const LongType fcPerm[2] = {1, 0};
      const uint32_t permOperand =
          addIntVectorOperand(model, fcPerm, 2, nextOperand, vectorStorage);
      const uint32_t transposeInputs[2] = {weightOperand, permOperand};
      const uint32_t transposeOutputs[1] = {transposedOperand};
      const int transposeResult = ANeuralNetworksModel_addOperation(
          model, ANEURALNETWORKS_TRANSPOSE, 2, transposeInputs, 1,
          transposeOutputs);
      if (transposeResult != ANEURALNETWORKS_NO_ERROR) {
        DSP_DIAG(COMPILE,
                 "NnapiGraphBackend: failed to add weight transpose for "
                 "fully-connected op at slot %d: %d",
                 i, transposeResult);
        return false;
      }
      operationSourceSlots.push_back(i);
      inputOperands[1] = transposedOperand;
    }

    // Add implicit parameters (padding, stride, axis, activation codes, etc.)
    if (!addImplicitParams(model, slots[i], operationOp, inputOperands, nextOperand,
                           externalInputs, numExternalInputs,
                           outputSlots, totalOutputSlots, vectorStorage)) {
      DSP_DIAG(COMPILE, "NnapiGraphBackend: failed to add implicit params for '%s' at slot %d",
                slots[i].ident.opName.c_str(), i);
      return false;
    }

    // Add output operands for this op
    std::vector<uint32_t> outputOperands;
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      int outIdx = slots[i].wiring.outputSlotIndices[o];
      NDArray* arr = nullptr;
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots) {
        arr = outputSlots[outIdx];
      }
      if (!arr) {
        DSP_DIAG(COMPILE, "NnapiGraphBackend: null output array at slot %d output %d (outIdx=%d)",
                  i, o, outIdx);
        return false;
      }

      uint32_t opIdx = 0;
      if (!addOperand(model, arr, nextOperand, contiguousCopies, &opIdx)) {
        DSP_DIAG(COMPILE,
                 "NnapiGraphBackend: unsupported output operand type at slot "
                 "%d output %d",
                 i, o);
        return false;
      }
      sourceToOperand[outIdx] = opIdx;
      outputOperands.push_back(opIdx);

      if (externalOutputSet.count(outIdx)) {
        std::vector<LongType> dimensions(static_cast<size_t>(arr->rankOf()));
        for (int dimension = 0; dimension < arr->rankOf(); dimension++) {
          dimensions[static_cast<size_t>(dimension)] = arr->sizeAt(dimension);
        }
        DataType outputBindingDataType = bindingDataType(arr->dataType());
        compiled.outputMappings.push_back(
            {outIdx, opIdx, true, arr->dataType(), outputBindingDataType,
             std::move(dimensions)});
      }
    }

    if (batchedEmbeddingLookup) {
      const auto& contract = embeddingContractIt->second;
      const DataType intermediateDataType = bindingDataType(gatherOutput->dataType());
      uint32_t flattenedOutputOperand = 0;
      if (intermediateDataType == DataType::UNKNOWN ||
          !addOperandDescriptor(intermediateDataType,
                                contract.flattenedOutputDimensions,
                                &flattenedOutputOperand)) {
        DSP_DIAG(COMPILE,
                 "NnapiGraphBackend: failed to declare flattened embedding "
                 "output at slot %d",
                 i);
        return false;
      }
      const uint32_t embeddingOutputs[1] = {flattenedOutputOperand};
      int result = ANeuralNetworksModel_addOperation(
          model, operationOp, static_cast<uint32_t>(inputOperands.size()),
          inputOperands.data(), 1, embeddingOutputs);
      if (result != ANEURALNETWORKS_NO_ERROR) {
        DSP_DIAG(COMPILE,
                 "NnapiGraphBackend: failed to add flattened embedding lookup "
                 "at slot %d: %d",
                 i, result);
        return false;
      }
      operationSourceSlots.push_back(i);

      const uint32_t outputShapeOperand =
          addShapeOperand(model, gatherOutput, nextOperand, vectorStorage);
      const uint32_t restoreInputs[2] = {flattenedOutputOperand,
                                         outputShapeOperand};
      const uint32_t restoreOutputs[1] = {outputOperands[0]};
      result = ANeuralNetworksModel_addOperation(
          model, ANEURALNETWORKS_RESHAPE, 2, restoreInputs, 1,
          restoreOutputs);
      if (result != ANEURALNETWORKS_NO_ERROR) {
        DSP_DIAG(COMPILE,
                 "NnapiGraphBackend: failed to restore batched embedding "
                 "shape at slot %d: %d",
                 i, result);
        return false;
      }
      operationSourceSlots.push_back(i);
    } else {
      const int result = ANeuralNetworksModel_addOperation(
          model, operationOp,
          static_cast<uint32_t>(inputOperands.size()), inputOperands.data(),
          static_cast<uint32_t>(outputOperands.size()), outputOperands.data());

      if (result != ANEURALNETWORKS_NO_ERROR) {
        DSP_DIAG(COMPILE, "NnapiGraphBackend: failed to add op '%s' (NNAPI code %d) at slot %d, error=%d",
                  slots[i].ident.opName.c_str(), nnapiOp, i, result);
        return false;
      }
      operationSourceSlots.push_back(i);
    }
  }

  // Phase 3: Identify model inputs and outputs
  std::vector<uint32_t> modelInputs;
  for (auto& m : compiled.inputMappings) {
    modelInputs.push_back(m.operand);
  }

  std::vector<uint32_t> modelOutputs;
  for (auto& m : compiled.outputMappings) {
    modelOutputs.push_back(m.operand);
  }

  if (modelOutputs.empty()) {
    DSP_DIAG(COMPILE, "NnapiGraphBackend: no external outputs found for segment [%d-%d]",
              startSlot, endSlot);
    return false;
  }

  int result = ANeuralNetworksModel_identifyInputsAndOutputs(
      model,
      static_cast<uint32_t>(modelInputs.size()), modelInputs.data(),
      static_cast<uint32_t>(modelOutputs.size()), modelOutputs.data());

  if (result != ANEURALNETWORKS_NO_ERROR) {
    DSP_DIAG(COMPILE, "NnapiGraphBackend: identifyInputsAndOutputs failed: %d", result);
    return false;
  }

  // Do not relax FLOAT32 operands to FLOAT16. BF16 source tensors are
  // converted to FLOAT32 at NNAPI boundaries, and transformer reductions and
  // normalization require the declared FLOAT32 range and precision. Relaxing
  // the complete model here changes logits while still reporting a successful
  // accelerator execution.
  result = ANeuralNetworksModel_finish(model);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    DSP_DIAG(COMPILE, "NnapiGraphBackend: model finish failed: %d", result);
    return false;
  }

  return true;
}

// ─── Compile ────────────────────────────────────────────────────────────────

bool NnapiGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        LongType shapeKey,
                                        int totalSlots,
                                        int* requestedOutputSlotIndices,
                                        int numRequestedOutputs) {
  return compileSegmentImpl(nullptr, seg, slots, externalInputs, numExternalInputs,
                            outputSlots, totalOutputSlots, shapeKey, totalSlots,
                            requestedOutputSlotIndices, numRequestedOutputs);
}

bool NnapiGraphBackend::compileSegment(const GraphBackendRequest& request,
                                        GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        LongType shapeKey, int totalSlots,
                                        int* requestedOutputSlotIndices,
                                        int numRequestedOutputs) {
  return compileSegmentImpl(&request, seg, slots, externalInputs, numExternalInputs,
                            outputSlots, totalOutputSlots, shapeKey, totalSlots,
                            requestedOutputSlotIndices, numRequestedOutputs);
}

bool NnapiGraphBackend::compileSegmentImpl(const GraphBackendRequest* request,
                                            GraphSegment& seg, NativeSlot* slots,
                                            NDArray** externalInputs, int numExternalInputs,
                                            NDArray** outputSlots, int totalOutputSlots,
                                            LongType shapeKey,
                                            int totalSlots,
                                            int* requestedOutputSlotIndices,
                                            int numRequestedOutputs) {
  int startSlot = seg.def.startSlot;
  int endSlot = seg.def.endSlot;
  DSP_DIAG(COMPILE, "NNAPI_PHASE compile_admission_begin seg[%d-%d] ops=%d shapeKey=%lld",
            startSlot, endSlot, endSlot - startSlot + 1,
            static_cast<long long>(shapeKey));
  const int segmentOps = endSlot - startSlot + 1;
  if (segmentOps > kMaxNnapiSegmentOps) {
    DSP_DIAG(COMPILE,
             "NnapiGraphBackend: compile admission rejected seg[%d-%d]: %d ops exceeds bounded limit %d; explicit replay",
             startSlot, endSlot, segmentOps, kMaxNnapiSegmentOps);
    return false;
  }

  // Reuse is deliberately scoped to this exact GraphSegment. Slot ranges and
  // shape keys are not model identities and must never cross-hit another plan.
  if (seg.compiledGraphBackendArtifactOwner == this &&
      seg.compiledGraphBackendArtifactShapeKey == shapeKey &&
      seg.compiledGraphBackendArtifact) {
    auto existing = std::static_pointer_cast<CompiledModel>(
        seg.compiledGraphBackendArtifact);
    if (existing->valid && existing->startSlot == startSlot &&
        existing->endSlot == endSlot) {
      lastCompilationAudit_ = existing->compilationAudit;
      DSP_DIAG(COMPILE, "NNAPI_PHASE compile_segment_hit seg[%d-%d]", startSlot, endSlot);
      return true;
    }
  }

  // Create NNAPI model
  DSP_DIAG(COMPILE, "NNAPI_PHASE model_create_begin seg[%d-%d]", startSlot, endSlot);
  ANeuralNetworksModel* model = nullptr;
  int result = ANeuralNetworksModel_create(&model);
  if (result != ANEURALNETWORKS_NO_ERROR || !model) {
    DSP_DIAG(COMPILE, "NnapiGraphBackend: failed to create model: %d", result);
    return false;
  }

  auto compiled = std::make_shared<CompiledModel>();
  compiled->startSlot = startSlot;
  compiled->endSlot = endSlot;
  compiled->shapeKey = shapeKey;

  // Build the model graph
  DSP_DIAG(COMPILE, "NNAPI_PHASE model_build_begin seg[%d-%d]", startSlot, endSlot);
  std::vector<int> operationSourceSlots;
  if (!buildModel(model, *compiled, slots, startSlot, endSlot,
                  externalInputs, numExternalInputs,
                  outputSlots, totalOutputSlots, totalSlots,
                  requestedOutputSlotIndices, numRequestedOutputs,
                  operationSourceSlots)) {
    ANeuralNetworksModel_free(model);
    return false;
  }

  compiled->model = model;
  DSP_DIAG(COMPILE, "NNAPI_PHASE model_build_done seg[%d-%d] inputs=%d outputs=%d",
            startSlot, endSlot,
            static_cast<int>(compiled->inputMappings.size()),
            static_cast<int>(compiled->outputMappings.size()));

  // Compile the model. Accelerator-only builds must classify the finished
  // model against the required device and pin compilation to that exact device.
  // ANeuralNetworksCompilation_create() is intentionally forbidden here because
  // Android may otherwise partition or fall back to a CPU NNAPI implementation.
  DSP_DIAG(COMPILE, "NNAPI_PHASE compilation_create_begin seg[%d-%d]", startSlot, endSlot);
  ANeuralNetworksCompilation* compilation = nullptr;
#if defined(SD_NNAPI_ACCELERATOR_ONLY)
#if defined(__ANDROID_API__) && __ANDROID_API__ >= 29
  if (requiredDevice_ == nullptr) {
    DSP_DIAG(COMPILE,
             "NNAPI_DEVICE_COMPILE_REJECTED seg[%d-%d] required=%s reason=device_not_selected",
             startSlot, endSlot, requiredDeviceName_.c_str());
    return false;
  }

  const ANeuralNetworksDevice* devices[] = {requiredDevice_};
  const int operationCount = static_cast<int>(operationSourceSlots.size());
  if (operationCount <= 0) {
    DSP_DIAG(COMPILE,
             "NNAPI_DEVICE_CLASSIFICATION_FAILED seg[%d-%d] reason=no_emitted_operations",
             startSlot, endSlot);
    return false;
  }
  bool* supportedOperations = new bool[static_cast<size_t>(operationCount)]();
  result = ANeuralNetworksModel_getSupportedOperationsForDevices(
      model, devices, 1, supportedOperations);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    delete[] supportedOperations;
    DSP_DIAG(COMPILE,
             "NNAPI_DEVICE_CLASSIFICATION_FAILED seg[%d-%d] device=%s status=%d",
             startSlot, endSlot, selectedDeviceName_.c_str(), result);
    return false;
  }

  int supportedCount = 0;
  for (int operationIndex = 0; operationIndex < operationCount; ++operationIndex) {
    if (supportedOperations[operationIndex]) {
      ++supportedCount;
      continue;
    }
    const int slotIndex = operationSourceSlots[static_cast<size_t>(operationIndex)];
    DSP_DIAG(COMPILE,
             "NNAPI_DEVICE_UNSUPPORTED_OPERATION device=%s slot=%d op=%s",
             selectedDeviceName_.c_str(), slotIndex,
             slots[slotIndex].ident.opName.c_str());
  }
  DSP_DIAG(COMPILE,
           "NNAPI_DEVICE_CLASSIFICATION device=%s seg[%d-%d] supported=%d total=%d",
           selectedDeviceName_.c_str(), startSlot, endSlot,
           supportedCount, operationCount);
  delete[] supportedOperations;
  if (supportedCount != operationCount) {
    return false;
  }

  result = ANeuralNetworksCompilation_createForDevices(model, devices, 1, &compilation);
#else
  DSP_DIAG(COMPILE,
           "NNAPI_DEVICE_COMPILE_REJECTED seg[%d-%d] required=%s reason=api_below_29",
           startSlot, endSlot, requiredDeviceName_.c_str());
  return false;
#endif
#else
  result = ANeuralNetworksCompilation_create(model, &compilation);
#endif
  if (result != ANEURALNETWORKS_NO_ERROR || !compilation) {
    DSP_DIAG(COMPILE,
             "NnapiGraphBackend: failed to create compilation for device '%s': %d",
             selectedDeviceName_.empty() ? "nnapi-default" : selectedDeviceName_.c_str(),
             result);
    return false;
  }
  compiled->compilation = compilation;

  ANeuralNetworksCompilation_setPreference(compilation, preference_);

#if defined(__ANDROID_API__) && __ANDROID_API__ >= 29
  if (request != nullptr &&
      !request->deviceCompilationCacheDirectory.empty() &&
      !request->deviceCompilationCacheModelKey.empty()) {
    const auto cacheToken = makeNnapiCacheToken(
        request->deviceCompilationCacheModelKey, startSlot, endSlot, shapeKey);
    const int cacheStatus = ANeuralNetworksCompilation_setCaching(
        compilation, request->deviceCompilationCacheDirectory.c_str(),
        cacheToken.data());
    if (cacheStatus == ANEURALNETWORKS_NO_ERROR) {
      DSP_DIAG(COMPILE,
               "NNAPI_DEVICE_CACHE_CONFIGURED device=%s seg[%d-%d] directory=%s",
               selectedDeviceName_.empty() ? "nnapi-default" : selectedDeviceName_.c_str(),
               startSlot, endSlot,
               request->deviceCompilationCacheDirectory.c_str());
    } else {
      // Driver caching is an optimization. A vendor or filesystem rejection must
      // not make an otherwise valid accelerator compilation unusable.
      DSP_DIAG(COMPILE,
               "NNAPI_DEVICE_CACHE_REJECTED device=%s seg[%d-%d] status=%d",
               selectedDeviceName_.empty() ? "nnapi-default" : selectedDeviceName_.c_str(),
               startSlot, endSlot, cacheStatus);
    }
  }
#endif

  DSP_DIAG(COMPILE, "NNAPI_PHASE compilation_finish_begin seg[%d-%d]", startSlot, endSlot);
  result = ANeuralNetworksCompilation_finish(compilation);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    DSP_DIAG(COMPILE, "NnapiGraphBackend: compilation finish failed: %d", result);
    return false;
  }

  compiled->valid = true;
  DSP_DIAG(COMPILE,
           "NNAPI_DEVICE_COMPILATION_COMMITTED device=%s type=%d feature_level=%lld seg[%d-%d]",
           selectedDeviceName_.empty() ? "nnapi-default" : selectedDeviceName_.c_str(),
           selectedDeviceType_, static_cast<long long>(selectedDeviceFeatureLevel_),
           startSlot, endSlot);
  DSP_DIAG(COMPILE, "NNAPI_PHASE compilation_finish_done seg[%d-%d]", startSlot, endSlot);

  // Build compilation audit
  for (int i = startSlot; i <= endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].ident.opName;
    int minApi = getMinApiLevel(slots[i].ident.opName);
    entry.wasCompiled = (getNnapiOpCode(slots[i].ident.opName) >= 0 && apiLevel_ >= minApi);
    if (!entry.wasCompiled) {
      if (apiLevel_ < minApi) {
        entry.reason = "requires API " + std::to_string(minApi) + " (device is " +
                       std::to_string(apiLevel_) + ")";
      } else {
        entry.reason = "no NNAPI op mapping";
      }
    }
    compiled->compilationAudit.push_back(entry);
  }

  lastCompilationAudit_ = compiled->compilationAudit;

  DSP_DIAG(COMPILE, "NnapiGraphBackend: compiled segment [%d-%d] with %d inputs, %d outputs (%d ops) on API %d",
            startSlot, endSlot,
            static_cast<int>(compiled->inputMappings.size()),
            static_cast<int>(compiled->outputMappings.size()),
            endSlot - startSlot + 1,
            apiLevel_);

  // The segment is the sole strong owner. The singleton keeps weak references
  // only so an explicit global invalidation can still release driver resources.
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    compiledArtifacts_.erase(
        std::remove_if(
            compiledArtifacts_.begin(), compiledArtifacts_.end(),
            [](const std::weak_ptr<CompiledModel>& artifact) {
              return artifact.expired();
            }),
        compiledArtifacts_.end());
    compiledArtifacts_.push_back(compiled);
  }
  seg.setCompiledGraphBackendArtifact(this, shapeKey, compiled);

  return true;
}

// ─── Execute ────────────────────────────────────────────────────────────────

Status NnapiGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots,
                                          void* stream) {
  int startSlot = seg.def.startSlot;
  int endSlot = seg.def.endSlot;
  if (seg.compiledGraphBackendArtifactOwner != this ||
      !seg.compiledGraphBackendArtifact) {
    return Status::KERNEL_FAILURE;
  }
  auto compiledHandle = std::static_pointer_cast<CompiledModel>(
      seg.compiledGraphBackendArtifact);
  CompiledModel* compiled = compiledHandle.get();
  if (!compiled->valid || !compiled->compilation ||
      compiled->startSlot != startSlot || compiled->endSlot != endSlot ||
      compiled->shapeKey != seg.def.shapeKeyState.compiledShapeKey) {
    return Status::KERNEL_FAILURE;
  }

  auto matchesCompiledDescriptor = [](NDArray* array, DataType expectedDataType,
                                      const std::vector<LongType>& expectedDimensions) {
    if (!array || array->dataType() != expectedDataType ||
        static_cast<size_t>(array->rankOf()) != expectedDimensions.size()) {
      return false;
    }
    for (int dimension = 0; dimension < array->rankOf(); dimension++) {
      if (array->sizeAt(dimension) !=
          expectedDimensions[static_cast<size_t>(dimension)]) {
        return false;
      }
    }
    return true;
  };

  // Create execution
  ANeuralNetworksExecution* execution = nullptr;
  int result = ANeuralNetworksExecution_create(compiled->compilation, &execution);
  if (result != ANEURALNETWORKS_NO_ERROR || !execution) {
    DSP_DIAG(EXECUTE, "NnapiGraphBackend: failed to create execution: %d", result);
    return Status::KERNEL_FAILURE;
  }

#if defined(__ANDROID_API__) && __ANDROID_API__ >= 31
  // Feature level 5 lets the compilation describe the alignment and trailing
  // padding its selected driver expects. Opt in before binding any operands so
  // output lengths may include that padding. Tensor accelerators can otherwise
  // access a full transfer unit past the raw tensor payload.
  const bool usePreferredBufferLayout = apiLevel_ >= 31;
  if (usePreferredBufferLayout) {
    result = ANeuralNetworksExecution_enableInputAndOutputPadding(execution, true);
    if (result != ANEURALNETWORKS_NO_ERROR) {
      DSP_DIAG(EXECUTE,
               "NnapiGraphBackend: failed to enable NNAPI operand padding: %d",
               result);
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }
  }
#else
  const bool usePreferredBufferLayout = false;
#endif

  // Temporary contiguous copies for non-contiguous input arrays
  std::vector<std::unique_ptr<NDArray>> contiguousInputCopies;

  // Padded input staging buffers (API 31+). With input/output padding enabled
  // the driver may access input buffers in padded chunks; bind aligned,
  // padded staging instead of exact-size buffers so the driver never reads
  // past the logical tensor payload. Entries must outlive the execution.
  struct StagedInputBuffer {
    std::vector<uint8_t> storage;
    size_t dataOffset = 0;
    size_t boundBytes = 0;

    uint8_t* data() { return storage.data() + dataOffset; }
  };
  std::vector<StagedInputBuffer> stagedInputs;
  stagedInputs.reserve(compiled->inputMappings.size());

  // Set inputs — map source indices to actual NDArray buffers
  for (uint32_t idx = 0; idx < compiled->inputMappings.size(); idx++) {
    auto& mapping = compiled->inputMappings[idx];
    NDArray* arr = nullptr;

    if (mapping.sourceIndex < 0) {
      int extIdx = -(mapping.sourceIndex + 1);
      if (extIdx < numExternalInputs && externalInputs) arr = externalInputs[extIdx];
    } else {
      if (mapping.sourceIndex < totalOutputSlots && outputSlots)
        arr = outputSlots[mapping.sourceIndex];
    }

    if (!arr) {
      DSP_DIAG(EXECUTE, "NnapiGraphBackend: null input array for source %d", mapping.sourceIndex);
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }
    DataBuffer* inputDataBuffer = arr->dataBuffer();
    if (inputDataBuffer == nullptr || !inputDataBuffer->isValid()) {
      DSP_DIAG(EXECUTE,
               "NNAPI_INPUT_TARGET_INVALID seg[%d-%d] input=%u source_slot=%d "
               "arr=%p db=%p",
               startSlot, endSlot, idx, mapping.sourceIndex, (void*)arr,
               (void*)inputDataBuffer);
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }

    if (!matchesCompiledDescriptor(arr, mapping.sourceDataType,
                                   mapping.dimensions)) {
      DSP_DIAG(
          EXECUTE,
          "NNAPI_OPERAND_DESCRIPTOR_MISMATCH kind=input seg[%d-%d] input=%u "
          "source_slot=%d expected_dtype=%d actual_dtype=%d expected_rank=%zu "
          "actual_rank=%d",
          startSlot, endSlot, idx, mapping.sourceIndex,
          static_cast<int>(mapping.sourceDataType),
          static_cast<int>(arr->dataType()), mapping.dimensions.size(),
          arr->rankOf());
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }

    // Ensure data is on host and contiguous
    arr->syncToHost();

    NDArray* contiguous = arr;
    if (!isDenseCOrder(arr)) {
      auto copy = std::make_unique<NDArray>(arr->dup('c'));
      contiguous = copy.get();
      contiguousInputCopies.push_back(std::move(copy));
    }

    if (mapping.sourceDataType == DataType::INT64 &&
        mapping.bindingDataType == DataType::INT32) {
      const LongType* values = contiguous->bufferAsT<LongType>();
      for (LongType element = 0; element < contiguous->lengthOf(); ++element) {
        if (values[element] < std::numeric_limits<int32_t>::min() ||
            values[element] > std::numeric_limits<int32_t>::max()) {
          DSP_DIAG(EXECUTE,
                   "NNAPI_GATHER_INDEX_RANGE seg[%d-%d] input=%u "
                   "source_slot=%d element=%lld value=%lld",
                   startSlot, endSlot, idx, mapping.sourceIndex,
                   static_cast<long long>(element),
                   static_cast<long long>(values[element]));
          ANeuralNetworksExecution_free(execution);
          return Status::KERNEL_FAILURE;
        }
      }
    }

    // Bind the representation that addOperand() compiled into the model.
    if (contiguous->dataType() != mapping.bindingDataType) {
      auto cast = std::make_unique<NDArray>(contiguous->cast(mapping.bindingDataType));
      contiguous = cast.get();
      contiguousInputCopies.push_back(std::move(cast));
    }
    if (!matchesCompiledDescriptor(contiguous, mapping.bindingDataType,
                                   mapping.dimensions)) {
      DSP_DIAG(EXECUTE,
               "NNAPI_OPERAND_DESCRIPTOR_MISMATCH kind=bound_input "
               "seg[%d-%d] input=%u source_slot=%d expected_dtype=%d "
               "actual_dtype=%d",
               startSlot, endSlot, idx, mapping.sourceIndex,
               static_cast<int>(mapping.bindingDataType),
               static_cast<int>(contiguous->dataType()));
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }

    void* buffer = contiguous->buffer();
    size_t bufferSize = contiguous->lengthOf() * contiguous->sizeOfT();

#if defined(__ANDROID_API__) && __ANDROID_API__ >= 31
    if (usePreferredBufferLayout) {
      uint32_t preferredAlignment = 1;
      uint32_t preferredPadding = 1;
      int layoutStatus =
          ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput(
              compiled->compilation, idx, &preferredAlignment);
      if (layoutStatus == ANEURALNETWORKS_NO_ERROR) {
        layoutStatus =
            ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput(
                compiled->compilation, idx, &preferredPadding);
      }
      if (layoutStatus == ANEURALNETWORKS_NO_ERROR &&
          preferredAlignment != 0 &&
          (preferredAlignment & (preferredAlignment - 1)) == 0 &&
          preferredPadding != 0 &&
          (preferredPadding & (preferredPadding - 1)) == 0 &&
          (preferredAlignment > 1 || preferredPadding > 1)) {
        const size_t paddingMask = static_cast<size_t>(preferredPadding - 1);
        const size_t alignmentSlack = static_cast<size_t>(preferredAlignment - 1);
        if (bufferSize >
            std::numeric_limits<size_t>::max() - paddingMask - alignmentSlack) {
          ANeuralNetworksExecution_free(execution);
          return Status::KERNEL_FAILURE;
        }
        const size_t boundBytes = (bufferSize + paddingMask) & ~paddingMask;

        stagedInputs.emplace_back();
        auto& staging = stagedInputs.back();
        staging.storage.assign(alignmentSlack + boundBytes,
                               static_cast<uint8_t>(0));
        const uintptr_t storageAddress =
            reinterpret_cast<uintptr_t>(staging.storage.data());
        const uintptr_t alignedDataAddress =
            (storageAddress + alignmentSlack) &
            ~static_cast<uintptr_t>(alignmentSlack);
        staging.dataOffset =
            static_cast<size_t>(alignedDataAddress - storageAddress);
        staging.boundBytes = boundBytes;
        std::memcpy(staging.data(), contiguous->buffer(), bufferSize);
        std::fill(staging.data() + bufferSize,
                  staging.data() + boundBytes, static_cast<uint8_t>(0));

        DSP_DIAG(EXECUTE,
                 "NNAPI_INPUT_STAGING seg[%d-%d] input=%u source_slot=%d "
                 "raw_bytes=%zu bound_bytes=%zu alignment=%u padding=%u",
                 startSlot, endSlot, idx, mapping.sourceIndex, bufferSize,
                 boundBytes, preferredAlignment, preferredPadding);
        buffer = staging.data();
        bufferSize = boundBytes;
      }
    }
#endif

    result = ANeuralNetworksExecution_setInput(
        execution, idx, nullptr, buffer, bufferSize);
    if (result != ANEURALNETWORKS_NO_ERROR) {
      DSP_DIAG(EXECUTE, "NnapiGraphBackend: setInput failed for idx %d: %d", idx, result);
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }
  }

  // Bind every output to host-owned storage using the alignment and trailing
  // padding requested by the selected compilation. Keep guards outside the
  // driver-visible region and construct NDArray metadata only after execution.
  struct StagedOutputBuffer {
    std::vector<uint8_t> storage;
    size_t dataOffset = 0;
    size_t rawBytes = 0;
    size_t boundBytes = 0;
    uint32_t alignment = 1;
    uint32_t padding = 1;

    uint8_t* data() { return storage.data() + dataOffset; }
  };
  std::vector<StagedOutputBuffer> stagedOutputs;
  stagedOutputs.reserve(compiled->outputMappings.size());

  for (uint32_t idx = 0; idx < compiled->outputMappings.size(); idx++) {
    auto& mapping = compiled->outputMappings[idx];
    NDArray* arr = nullptr;

    if (mapping.sourceIndex >= 0 && mapping.sourceIndex < totalOutputSlots && outputSlots) {
      arr = outputSlots[mapping.sourceIndex];
    }

    if (!arr) {
      DSP_DIAG(EXECUTE, "NnapiGraphBackend: null output array for source %d", mapping.sourceIndex);
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }
    DataBuffer* outputDataBuffer = arr->dataBuffer();
    if (outputDataBuffer == nullptr || !outputDataBuffer->isValid()) {
      DSP_DIAG(EXECUTE,
               "NNAPI_OUTPUT_TARGET_INVALID seg[%d-%d] output=%u source_slot=%d "
               "stage=before_bind arr=%p db=%p",
               startSlot, endSlot, idx, mapping.sourceIndex, (void*)arr,
               (void*)outputDataBuffer);
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }
    if (!matchesCompiledDescriptor(arr, mapping.sourceDataType,
                                   mapping.dimensions)) {
      DSP_DIAG(
          EXECUTE,
          "NNAPI_OPERAND_DESCRIPTOR_MISMATCH kind=output seg[%d-%d] output=%u "
          "source_slot=%d expected_dtype=%d actual_dtype=%d expected_rank=%zu "
          "actual_rank=%d",
          startSlot, endSlot, idx, mapping.sourceIndex,
          static_cast<int>(mapping.sourceDataType),
          static_cast<int>(arr->dataType()), mapping.dimensions.size(),
          arr->rankOf());
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }

    uint32_t preferredAlignment = 1;
    uint32_t preferredPadding = 1;
#if defined(__ANDROID_API__) && __ANDROID_API__ >= 31
    if (usePreferredBufferLayout) {
      result = ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput(
          compiled->compilation, idx, &preferredAlignment);
      if (result == ANEURALNETWORKS_NO_ERROR) {
        result = ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput(
            compiled->compilation, idx, &preferredPadding);
      }
      if (result != ANEURALNETWORKS_NO_ERROR || preferredAlignment == 0 ||
          (preferredAlignment & (preferredAlignment - 1)) != 0 ||
          preferredPadding == 0 ||
          (preferredPadding & (preferredPadding - 1)) != 0) {
        DSP_DIAG(EXECUTE,
                 "NnapiGraphBackend: invalid preferred output layout for "
                 "output %u: status=%d alignment=%u padding=%u",
                 idx, result, preferredAlignment, preferredPadding);
        ANeuralNetworksExecution_free(execution);
        return Status::KERNEL_FAILURE;
      }
    }
#endif

    const LongType outputLength = arr->lengthOf();
    const size_t elementSize = DataTypeUtils::sizeOf(mapping.bindingDataType);
    const size_t maxBufferBytes = std::numeric_limits<size_t>::max();
    if (outputLength < 0 || elementSize == 0 ||
        static_cast<unsigned long long>(outputLength) >
            static_cast<unsigned long long>(maxBufferBytes / elementSize)) {
      DSP_DIAG(EXECUTE,
               "NnapiGraphBackend: invalid staging capacity for output %u "
               "source slot %d length=%lld element_bytes=%zu",
               idx, mapping.sourceIndex, static_cast<long long>(outputLength),
               elementSize);
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }

    const size_t rawBytes = static_cast<size_t>(outputLength) * elementSize;
    const size_t paddingMask = static_cast<size_t>(preferredPadding - 1);
    const size_t alignmentSlack = static_cast<size_t>(preferredAlignment - 1);
    if (rawBytes > maxBufferBytes - paddingMask) {
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }
    const size_t boundBytes = (rawBytes + paddingMask) & ~paddingMask;
    const size_t guardBytes = 2 * kNnapiOutputGuardBytes;
    if (alignmentSlack > maxBufferBytes - guardBytes ||
        boundBytes > maxBufferBytes - guardBytes - alignmentSlack) {
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }

    stagedOutputs.emplace_back();
    auto& staging = stagedOutputs.back();
    staging.storage.assign(guardBytes + alignmentSlack + boundBytes,
                           kNnapiOutputGuardValue);
    const uintptr_t storageAddress =
        reinterpret_cast<uintptr_t>(staging.storage.data());
    const uintptr_t minimumDataAddress =
        storageAddress + kNnapiOutputGuardBytes;
    const uintptr_t alignedDataAddress =
        (minimumDataAddress + alignmentSlack) & ~static_cast<uintptr_t>(alignmentSlack);
    staging.dataOffset = static_cast<size_t>(alignedDataAddress - storageAddress);
    staging.rawBytes = rawBytes;
    staging.boundBytes = boundBytes;
    staging.alignment = preferredAlignment;
    staging.padding = preferredPadding;
    std::fill(staging.data(), staging.data() + boundBytes,
              static_cast<uint8_t>(0));

    void* buffer = staging.data();
    DSP_DIAG(
        EXECUTE,
        "NNAPI_OUTPUT_STAGING seg[%d-%d] output=%u source_slot=%d "
        "source_dtype=%d binding_dtype=%d storage_bytes=%zu raw_bytes=%zu "
        "bound_bytes=%zu alignment=%u padding=%u",
        startSlot, endSlot, idx, mapping.sourceIndex,
        static_cast<int>(mapping.sourceDataType),
        static_cast<int>(mapping.bindingDataType),
        arr->lengthOf() * arr->sizeOfT(), rawBytes, boundBytes,
        preferredAlignment, preferredPadding);

    result = ANeuralNetworksExecution_setOutput(
        execution, idx, nullptr, buffer, boundBytes);
    if (result != ANEURALNETWORKS_NO_ERROR) {
      DSP_DIAG(EXECUTE, "NnapiGraphBackend: setOutput failed for idx %d: %d", idx, result);
      ANeuralNetworksExecution_free(execution);
      return Status::KERNEL_FAILURE;
    }
  }

  // Execute synchronously via startCompute + wait. In accelerator-only builds
  // this compilation was created exclusively for selectedDeviceName_.
  DSP_DIAG(EXECUTE,
           "NNAPI_DEVICE_EXECUTE_BEGIN device=%s seg[%d-%d]",
           selectedDeviceName_.empty() ? "nnapi-default" : selectedDeviceName_.c_str(),
           startSlot, endSlot);
  DSP_DIAG(EXECUTE, "NNAPI_PHASE start_compute_begin seg[%d-%d]", startSlot, endSlot);
  ANeuralNetworksEvent* event = nullptr;
  result = ANeuralNetworksExecution_startCompute(execution, &event);
  if (result != ANEURALNETWORKS_NO_ERROR) {
    DSP_DIAG(EXECUTE, "NnapiGraphBackend: startCompute failed: %d", result);
    ANeuralNetworksExecution_free(execution);
    return Status::KERNEL_FAILURE;
  }

  DSP_DIAG(EXECUTE, "NNAPI_PHASE event_wait_begin seg[%d-%d]", startSlot, endSlot);
  result = ANeuralNetworksEvent_wait(event);
  DSP_DIAG(EXECUTE, "NNAPI_PHASE event_wait_done seg[%d-%d] status=%d",
            startSlot, endSlot, result);
  ANeuralNetworksEvent_free(event);
  ANeuralNetworksExecution_free(execution);

  if (result != ANEURALNETWORKS_NO_ERROR) {
    DSP_DIAG(EXECUTE, "NnapiGraphBackend: execution failed for segment [%d-%d]: %d",
              startSlot, endSlot, result);
    return Status::KERNEL_FAILURE;
  }

  // Publish accelerator outputs only after the event completed. Staging data
  // is copied or converted into the DynamicShapePlan-owned target here.
  for (uint32_t idx = 0; idx < compiled->outputMappings.size(); idx++) {
    auto& mapping = compiled->outputMappings[idx];
    if (mapping.sourceIndex < 0 || mapping.sourceIndex >= totalOutputSlots || !outputSlots) {
      continue;
    }

    NDArray* arr = outputSlots[mapping.sourceIndex];
    if (!arr) {
      continue;
    }
    DataBuffer* outputDataBuffer = arr->dataBuffer();
    if (outputDataBuffer == nullptr || !outputDataBuffer->isValid()) {
      DSP_DIAG(EXECUTE,
               "NNAPI_OUTPUT_TARGET_INVALID seg[%d-%d] output=%u source_slot=%d "
               "stage=before_copyback arr=%p db=%p",
               startSlot, endSlot, idx, mapping.sourceIndex, (void*)arr,
               (void*)outputDataBuffer);
      return Status::KERNEL_FAILURE;
    }

    auto& staging = stagedOutputs[idx];
    const bool prefixGuardIntact = std::all_of(
        staging.storage.begin(), staging.storage.begin() + staging.dataOffset,
        [](uint8_t value) { return value == kNnapiOutputGuardValue; });
    const bool suffixGuardIntact = std::all_of(
        staging.storage.begin() + staging.dataOffset + staging.boundBytes,
        staging.storage.end(),
        [](uint8_t value) { return value == kNnapiOutputGuardValue; });
    if (!prefixGuardIntact || !suffixGuardIntact) {
      DSP_DIAG(EXECUTE,
               "NNAPI_OUTPUT_GUARD_CORRUPTION seg[%d-%d] output=%u "
               "source_slot=%d raw_bytes=%zu bound_bytes=%zu alignment=%u "
               "padding=%u prefix_intact=%d suffix_intact=%d",
               startSlot, endSlot, idx, mapping.sourceIndex,
               staging.rawBytes, staging.boundBytes, staging.alignment,
               staging.padding, prefixGuardIntact ? 1 : 0,
               suffixGuardIntact ? 1 : 0);
      return Status::KERNEL_FAILURE;
    }

    std::vector<LongType> stagingShape = mapping.dimensions;
    NDArray boundOutput(staging.data(), 'c', stagingShape,
                        mapping.bindingDataType, arr->getContext(), false);
    boundOutput.tickWriteHost();
    arr->assign(&boundOutput);
    if (!outputDataBuffer->isValid()) {
      DSP_DIAG(EXECUTE,
               "NNAPI_OUTPUT_TARGET_INVALID seg[%d-%d] output=%u source_slot=%d "
               "stage=after_copyback arr=%p db=%p",
               startSlot, endSlot, idx, mapping.sourceIndex, (void*)arr,
               (void*)outputDataBuffer);
      return Status::KERNEL_FAILURE;
    }
    DSP_DIAG(EXECUTE,
             "NNAPI_OUTPUT_COPYBACK seg[%d-%d] output=%u source_slot=%d "
             "source_dtype=%d binding_dtype=%d",
             startSlot, endSlot, idx, mapping.sourceIndex,
             static_cast<int>(mapping.sourceDataType),
             static_cast<int>(mapping.bindingDataType));
    arr->tickWriteHost();
  }

  DSP_DIAG(EXECUTE,
           "NNAPI_DEVICE_EXECUTE_DONE device=%s seg[%d-%d] status=0",
           selectedDeviceName_.empty() ? "nnapi-default" : selectedDeviceName_.c_str(),
           startSlot, endSlot);
  DSP_DIAG(EXECUTE, "NNAPI_PHASE execute_done seg[%d-%d]", startSlot, endSlot);
  return Status::OK;
}

// ─── Cache management ───────────────────────────────────────────────────────

void NnapiGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  for (auto& weakArtifact : compiledArtifacts_) {
    if (auto artifact = weakArtifact.lock()) artifact->invalidate();
  }
  compiledArtifacts_.clear();
}

std::vector<CompilationAuditEntry> NnapiGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_NNAPI
