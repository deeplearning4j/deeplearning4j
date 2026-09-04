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

/**
 * NativeDynamicShapePlan — Segment Management
 *
 * Contains computeSegmentShapeKey(),
 * executeSegmentWithGraphBackend(), and executeSegmentSlotBySlot().
 */

#include <graph/NativeDynamicShapePlan.h>
#include <graph/GraphBackendResolver.h>
#include <graph/gpu/SymbolicShapeRanges.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspPhaseUtils.h>
#include <graph/DspHashUtils.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/PlanExecutionContext.h>
#include <graph/DspThreadState.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/DspDeviceDispatch.h>
#include <execution/LaunchContext.h>
#if !defined(SD_VULKAN)
#include <graph/cpu/FunctionalReplayHandle.h>
#endif
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <system/Environment.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_set>

// GraphSegment static methods (moved from header to avoid Environment.h in NativeDynamicShapePlan.h)
int GraphSegment::maxOomRetries() { return sd::Environment::getInstance().dspCaptureOomMaxRetries(); }
int GraphSegment::retryInterval() { return sd::Environment::getInstance().dspCaptureOomRetryInterval(); }

// Include compiled graph backend implementations conditionally
#include <config.h>
#if !defined(SD_VULKAN)
#if HAVE_ONEDNN
#include <graph/cpu/OneDnnGraphBackend.h>
#endif
#if HAVE_ARMCOMPUTE
#include <graph/cpu/AclGraphBackend.h>
#endif
#if HAVE_MLIR
#include <graph/cpu/MlirCpuGraphBackend.h>
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
#include <graph/cpu/ArmHybridGraphBackend.h>
#endif
#endif
#if HAVE_NNAPI
#include <graph/cpu/NnapiGraphBackend.h>
#endif
#if HAVE_MLX
#include <graph/cpu/MlxGraphBackend.h>
#endif
#if HAVE_OPENVINO
#include <graph/cpu/OpenVinoGraphBackend.h>
#endif
#ifdef SD_TPU
#include <graph/tpu/TpuGraphBackend.h>
#endif
#ifdef HAVE_HEXAGON_MLIR
#include <graph/hexagon/HexagonGraphBackend.h>
#endif
#if defined(SD_HIP)
#include <graph/hip/HipGraphBackend.h>
#endif
#endif
namespace sd {
namespace graph {

static void scanAllSlotsForCorruption(
    NDArray** outputSlots, int totalOutputSlots,
    const char* checkpoint, int execCount) {
  for (int i = 0; i < totalOutputSlots; i++) {
    if (outputSlots[i] == nullptr) continue;
    auto* sib = outputSlots[i]->shapeInfoConstBuffer();
    if (sib == nullptr) continue;
    uintptr_t sibAddr = reinterpret_cast<uintptr_t>(sib);
    if (sibAddr % alignof(ConstantShapeBuffer) != 0) {
      DSP_DIAG(MEMORY,
               "CORRUPTION_SCAN_HIT: checkpoint=%s slot=%d arr=%p "
               "_shapeInfoBuffer=%p alignOffset=%zu execCount=%d",
               checkpoint, i, (void*)outputSlots[i], (void*)sib,
               static_cast<size_t>(sibAddr % alignof(ConstantShapeBuffer)),
               execCount);
      return;
    }
  }
}

namespace {
// OP_SANITY is a parity diagnostic, not a finiteness sampler. Fully hash typical
// activation/state tensors while retaining a cap for multi-hundred-megabyte
// parameter tables such as the tied embedding/output projection.
static constexpr LongType kOpSanityMaxScannedValues = 524288;

// Status enum string helper — delegates to shared dsp::dspStatusName in DspPhaseUtils.h.
const char* statusName_seg(Status status) {
  return dsp::dspStatusName(status);
}

void recordFailureDetailIfMissing(Status status, const std::string& detail) {
  auto* launchContext = LaunchContext::defaultContext();
  auto* errorReference =
      launchContext != nullptr ? launchContext->errorReference() : nullptr;
  if (errorReference == nullptr) return;
  const char* existing = errorReference->errorMessage();
  if (existing == nullptr || existing[0] == '\0') {
    errorReference->setErrorMessage(detail);
  }
  errorReference->setErrorCode(static_cast<int>(status));
}

struct OpSanitySummary {
  uint64_t valueHash = 0xcbf29ce484222325ULL;
  LongType scannedCount = 0;
  LongType finiteCount = 0;
  LongType nanCount = 0;
  LongType infCount = 0;
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();
  double sum = 0.0;
  double sumCompensation = 0.0;
  double l1 = 0.0;
  double l1Compensation = 0.0;
  double samples[4] = {};
  int sampleCount = 0;
};

static void addCompensated(double value, double* sum, double* compensation) {
  const double adjusted = value - *compensation;
  const double next = *sum + adjusted;
  *compensation = (next - *sum) - adjusted;
  *sum = next;
}

template <typename T>
static void hashSanityValue(T value, double numeric, OpSanitySummary* summary) {
  if constexpr (std::is_same<T, bool>::value) {
    summary->valueHash ^= static_cast<uint8_t>(value ? 1 : 0);
    summary->valueHash *= 0x100000001b3ULL;
  } else if constexpr (std::is_integral<T>::value) {
    using UnsignedT = typename std::make_unsigned<T>::type;
    const UnsignedT bits = static_cast<UnsignedT>(value);
    for (size_t shift = 0; shift < sizeof(UnsignedT) * 8; shift += 8) {
      summary->valueHash ^= static_cast<uint8_t>((bits >> shift) & static_cast<UnsignedT>(0xffU));
      summary->valueHash *= 0x100000001b3ULL;
    }
  } else {
    uint64_t canonicalBits = 0;
    std::memcpy(&canonicalBits, &numeric, sizeof(canonicalBits));
    for (int shift = 0; shift < 64; shift += 8) {
      summary->valueHash ^= static_cast<uint8_t>((canonicalBits >> shift) & 0xffU);
      summary->valueHash *= 0x100000001b3ULL;
    }
  }
}

static std::string formatSanitySamples(const double* values, int count) {
  std::string result = "[";
  for (int index = 0; index < count; ++index) {
    if (index > 0) result += ",";
    char value[32];
    snprintf(value, sizeof(value), "%.9g", values[index]);
    result += value;
  }
  result += "]";
  return result;
}

template <typename T>
void summarizeLogicalArray(NDArray* array, OpSanitySummary* summary) {
  if (array == nullptr || summary == nullptr || array->lengthOf() <= 0) return;
  const bool denseC = array->ordering() == 'c' &&
      shape::strideDescendingCAscendingF(array->shapeInfo());
  const T* buffer = reinterpret_cast<const T*>(array->buffer());
  const LongType length = array->lengthOf();
  const LongType scanCount = std::min(length, kOpSanityMaxScannedValues);
  summary->scannedCount = scanCount;
  LongType sampleOrdinals[4] = {};
  int samplePositionCount = 0;
  const LongType candidates[4] = {
      0, scanCount / 3, scanCount - 1 - scanCount / 3, scanCount - 1};
  for (int candidate = 0; candidate < 4; ++candidate) {
    if (candidate == 0 || candidates[candidate] != sampleOrdinals[samplePositionCount - 1]) {
      sampleOrdinals[samplePositionCount++] = candidates[candidate];
    }
  }
  const int rank = array->rankOf();
  const LongType scanSpan = scanCount - 1;
  const LongType logicalSpan = length - 1;

  for (LongType scanOrdinal = 0; scanOrdinal < scanCount; ++scanOrdinal) {
    LongType logicalIndex = scanOrdinal;
    if (scanCount < length && scanSpan > 0) {
      // Evenly cover the complete logical array without multiplying two large
      // dimensions. The remainder term is bounded by the configured scan cap.
      logicalIndex = (logicalSpan / scanSpan) * scanOrdinal +
          ((logicalSpan % scanSpan) * scanOrdinal) / scanSpan;
    }
    LongType offset = logicalIndex;
    if (!denseC) {
      LongType remaining = logicalIndex;
      offset = 0;
      for (int axis = rank - 1; axis >= 0; --axis) {
        const LongType dimension = array->sizeAt(axis);
        const LongType coordinate = dimension > 0 ? remaining % dimension : 0;
        remaining = dimension > 0 ? remaining / dimension : 0;
        offset += coordinate * array->strideAt(axis);
      }
    }
    const T value = buffer[offset];
    const double numeric = static_cast<double>(value);
    hashSanityValue(value, numeric, summary);

    if (std::isnan(numeric)) {
      summary->nanCount++;
    } else if (std::isinf(numeric)) {
      summary->infCount++;
    } else {
      summary->finiteCount++;
      summary->minimum = std::min(summary->minimum, numeric);
      summary->maximum = std::max(summary->maximum, numeric);
      addCompensated(numeric, &summary->sum, &summary->sumCompensation);
      addCompensated(std::abs(numeric), &summary->l1, &summary->l1Compensation);
    }
    for (int sample = 0; sample < samplePositionCount; ++sample) {
      if (scanOrdinal == sampleOrdinals[sample]) {
        summary->samples[sample] = numeric;
        summary->sampleCount = std::max(summary->sampleCount, sample + 1);
      }
    }
  }
}

static bool summarizeByDataType(NDArray* output, OpSanitySummary* summary) {
  switch (output->dataType()) {
    case DataType::BOOL: summarizeLogicalArray<bool>(output, summary); return true;
    case DataType::INT8: summarizeLogicalArray<SignedChar>(output, summary); return true;
    case DataType::INT16: summarizeLogicalArray<Int16Type>(output, summary); return true;
    case DataType::INT32: summarizeLogicalArray<Int32Type>(output, summary); return true;
    case DataType::INT64: summarizeLogicalArray<LongType>(output, summary); return true;
    case DataType::UINT8: summarizeLogicalArray<UnsignedChar>(output, summary); return true;
    case DataType::UINT16: summarizeLogicalArray<UInt16Type>(output, summary); return true;
    case DataType::UINT32: summarizeLogicalArray<UInt32Type>(output, summary); return true;
    case DataType::UINT64: summarizeLogicalArray<UInt64Type>(output, summary); return true;
    case DataType::QINT8: summarizeLogicalArray<SignedChar>(output, summary); return true;
    case DataType::QINT16: summarizeLogicalArray<Int16Type>(output, summary); return true;
    case DataType::HALF: summarizeLogicalArray<float16>(output, summary); return true;
    case DataType::BFLOAT16: summarizeLogicalArray<bfloat16>(output, summary); return true;
    case DataType::FLOAT32: summarizeLogicalArray<float>(output, summary); return true;
    case DataType::DOUBLE: summarizeLogicalArray<double>(output, summary); return true;
    case DataType::FLOAT8: summarizeLogicalArray<float8>(output, summary); return true;
    case DataType::FLOAT8_E5M2: summarizeLogicalArray<float8_e5m2>(output, summary); return true;
    default: return false;
  }
}

static bool opSanityEnabled(int execCount) {
  auto& diagnostics = DspDiagnostics::getInstance();
  return Environment::getInstance().dsp().diagnosticsNativeDump() &&
      diagnostics.isEnabled(DSP_DIAG_VERIFY) &&
      diagnostics.withinExecLimit(execCount);
}

static void recordOpSanity(const char* backend, int segmentId, int stepIdx,
                           int outputOrdinal, int outputSlot, const char* opName,
                           NDArray* output, int execCount, bool captureActive) {
  if (!opSanityEnabled(execCount)) return;
  auto& diagnostics = DspDiagnostics::getInstance();
  if (output == nullptr) {
    diagnostics.recordEvent(
        DSP_DIAG_VERIFY, stepIdx, segmentId, -1, opName, 0,
        "OP_SANITY backend=%s exec=%d output=%d slot=%d state=null",
        backend, execCount, outputOrdinal, outputSlot);
    return;
  }
  bool metadataOnly = captureActive;
#if defined(SD_CUDA) || defined(SD_VULKAN) || defined(SD_HIP)
  // Device-backed diagnostics stay asynchronous. CPU and Android host backends
  // provide the comparable value records used by this mode.
  metadataOnly = true;
#endif
  if (metadataOnly) {
    DataBuffer* dataBuffer = output->dataBuffer();
    const bool dataBufferValid = dataBuffer != nullptr && dataBuffer->isValid();
    diagnostics.recordEvent(
        DSP_DIAG_VERIFY, stepIdx, segmentId, -1, opName, 0,
        "OP_SANITY backend=%s exec=%d output=%d slot=%d state=capture-skip "
        "dtype=%s shape=%s len=%lld arr=%p db=%p primary=%p special=%p offset=%lld",
        backend, execCount, outputOrdinal, outputSlot,
        DataTypeUtils::asString(output->dataType()).c_str(),
        dspShapeStr(output).c_str(), static_cast<long long>(output->lengthOf()),
        (void*)output, (void*)dataBuffer,
        dataBufferValid ? dataBuffer->primary() : nullptr,
        dataBufferValid ? dataBuffer->special() : nullptr,
        static_cast<long long>(output->offset()));
    return;
  }

  if (output->lengthOf() > 0 &&
      (output->dataBuffer() == nullptr || output->buffer() == nullptr)) {
    diagnostics.recordEvent(
        DSP_DIAG_VERIFY, stepIdx, segmentId, -1, opName, 0,
        "OP_SANITY backend=%s exec=%d output=%d slot=%d state=invalid-buffer "
        "dtype=%s shape=%s len=%lld",
        backend, execCount, outputOrdinal, outputSlot,
        DataTypeUtils::asString(output->dataType()).c_str(),
        dspShapeStr(output).c_str(), static_cast<long long>(output->lengthOf()));
    return;
  }

  output->forceSyncToHost();
  OpSanitySummary summary;
  if (!summarizeByDataType(output, &summary)) {
    diagnostics.recordEvent(
        DSP_DIAG_VERIFY, stepIdx, segmentId, -1, opName, 0,
        "OP_SANITY backend=%s exec=%d output=%d slot=%d state=unsupported-dtype "
        "dtype=%s shape=%s len=%lld",
        backend, execCount, outputOrdinal, outputSlot,
        DataTypeUtils::asString(output->dataType()).c_str(),
        dspShapeStr(output).c_str(), static_cast<long long>(output->lengthOf()));
    return;
  }
  const double mean = summary.finiteCount > 0
      ? summary.sum / static_cast<double>(summary.finiteCount)
      : std::numeric_limits<double>::quiet_NaN();
  const bool completeScan = summary.scannedCount == output->lengthOf();
  const std::string samples = formatSanitySamples(summary.samples, summary.sampleCount);
  DataBuffer* dataBuffer = output->dataBuffer();
  const bool dataBufferValid = dataBuffer != nullptr && dataBuffer->isValid();
  diagnostics.recordEvent(
      DSP_DIAG_VERIFY, stepIdx, segmentId, -1, opName, 0,
      "OP_SANITY backend=%s exec=%d output=%d slot=%d dtype=%s shape=%s "
      "len=%lld scanned=%lld coverage=%s hash=0x%016llx "
      "finite=%lld nan=%lld inf=%lld "
      "min=%.9g max=%.9g mean=%.9g l1=%.9g samples=%s "
      "arr=%p db=%p primary=%p special=%p offset=%lld",
      backend, execCount, outputOrdinal, outputSlot,
      DataTypeUtils::asString(output->dataType()).c_str(),
      dspShapeStr(output).c_str(), static_cast<long long>(output->lengthOf()),
      static_cast<long long>(summary.scannedCount),
      completeScan ? "full" : "sampled",
      static_cast<unsigned long long>(summary.valueHash),
      static_cast<long long>(summary.finiteCount),
      static_cast<long long>(summary.nanCount),
      static_cast<long long>(summary.infCount),
      summary.minimum, summary.maximum, mean, summary.l1, samples.c_str(),
      (void*)output, (void*)dataBuffer,
      dataBufferValid ? dataBuffer->primary() : nullptr,
      dataBufferValid ? dataBuffer->special() : nullptr,
      static_cast<long long>(output->offset()));
}

static void recordSlotOpSanity(
    const char* backend, int segmentId, int stepIdx,
    const NativeSlot* slots, NDArray** outputSlots,
    int totalOutputSlots, int execCount, bool captureActive,
    bool outputsDead = false) {
  const auto& slot = slots[stepIdx];
  for (int output = 0; output < slot.wiring.numOutputs; ++output) {
    const int outputSlot = slot.wiring.outputSlotIndices[output];
    if (outputsDead) {
      if (opSanityEnabled(execCount)) {
        DspDiagnostics::getInstance().recordEvent(
            DSP_DIAG_VERIFY, stepIdx, segmentId, -1,
            slot.ident.opName.c_str(), 0,
            "OP_SANITY backend=%s exec=%d output=%d slot=%d state=dead",
            backend, execCount, output, outputSlot);
      }
      continue;
    }
    NDArray* value = outputSlot >= 0 && outputSlot < totalOutputSlots
        ? outputSlots[outputSlot]
        : nullptr;
    recordOpSanity(
        backend, segmentId, stepIdx, output, outputSlot,
        slot.ident.opName.c_str(), value, execCount, captureActive);
  }
}

static void recordGdrInputSanity(
    const char* backend, int segmentId, int stepIdx,
    const NativeSlot* slots, NDArray** outputSlots, int totalOutputSlots,
    NDArray** externalArrays, int numExternalArrays,
    const std::vector<std::string>& externalInputNames,
    int execCount, bool captureActive) {
  if (!opSanityEnabled(execCount) || captureActive || slots == nullptr ||
      slots[stepIdx].ident.opName != "gated_delta_rule") {
    return;
  }

  const auto& slot = slots[stepIdx];
  auto& diagnostics = DspDiagnostics::getInstance();
  for (int input = 0; input < slot.wiring.numInputs; ++input) {
    const int sourceIndex = slot.wiring.inputSourceIndices[input];
    NDArray* value = nullptr;
    std::string source;
    if (sourceIndex >= 0) {
      source = "slot:" + std::to_string(sourceIndex);
      if (outputSlots != nullptr && sourceIndex < totalOutputSlots) {
        value = outputSlots[sourceIndex];
      }
    } else {
      const int externalIndex = -(sourceIndex + 1);
      source = "ext:" + std::to_string(externalIndex);
      if (externalIndex >= 0 &&
          externalIndex < static_cast<int>(externalInputNames.size())) {
        source += ":" + externalInputNames[externalIndex];
      }
      if (externalArrays != nullptr && externalIndex >= 0 &&
          externalIndex < numExternalArrays) {
        value = externalArrays[externalIndex];
      }
    }

    if (value == nullptr) {
      diagnostics.recordEvent(
          DSP_DIAG_VERIFY, stepIdx, segmentId, -1, slot.ident.opName.c_str(), 0,
          "GDR_INPUT_SANITY backend=%s exec=%d input=%d source=%s state=null",
          backend, execCount, input, source.c_str());
      continue;
    }

#if defined(SD_CUDA) || defined(SD_VULKAN) || defined(SD_HIP)
    diagnostics.recordEvent(
        DSP_DIAG_VERIFY, stepIdx, segmentId, -1, slot.ident.opName.c_str(), 0,
        "GDR_INPUT_SANITY backend=%s exec=%d input=%d source=%s "
        "state=device-metadata dtype=%s shape=%s offset=%lld len=%lld",
        backend, execCount, input, source.c_str(),
        DataTypeUtils::asString(value->dataType()).c_str(),
        dspShapeStr(value).c_str(), static_cast<long long>(value->offset()),
        static_cast<long long>(value->lengthOf()));
    continue;
#endif

    if (value->lengthOf() > 0 &&
        (value->dataBuffer() == nullptr || value->buffer() == nullptr)) {
      diagnostics.recordEvent(
          DSP_DIAG_VERIFY, stepIdx, segmentId, -1, slot.ident.opName.c_str(), 0,
          "GDR_INPUT_SANITY backend=%s exec=%d input=%d source=%s "
          "state=invalid-buffer dtype=%s shape=%s len=%lld",
          backend, execCount, input, source.c_str(),
          DataTypeUtils::asString(value->dataType()).c_str(),
          dspShapeStr(value).c_str(), static_cast<long long>(value->lengthOf()));
      continue;
    }

    value->forceSyncToHost();
    OpSanitySummary summary;
    if (!summarizeByDataType(value, &summary)) {
      diagnostics.recordEvent(
          DSP_DIAG_VERIFY, stepIdx, segmentId, -1, slot.ident.opName.c_str(), 0,
          "GDR_INPUT_SANITY backend=%s exec=%d input=%d source=%s "
          "state=unsupported-dtype dtype=%s shape=%s",
          backend, execCount, input, source.c_str(),
          DataTypeUtils::asString(value->dataType()).c_str(),
          dspShapeStr(value).c_str());
      continue;
    }

    std::string strides = "[";
    for (int axis = 0; axis < value->rankOf(); ++axis) {
      if (axis > 0) strides += ",";
      strides += std::to_string(value->strideAt(axis));
    }
    strides += "]";
    const double mean = summary.finiteCount > 0
        ? summary.sum / static_cast<double>(summary.finiteCount)
        : std::numeric_limits<double>::quiet_NaN();
    const std::string samples = formatSanitySamples(summary.samples, summary.sampleCount);
    DataBuffer* dataBuffer = value->dataBuffer();
    const bool dataBufferValid = dataBuffer != nullptr && dataBuffer->isValid();
    diagnostics.recordEvent(
        DSP_DIAG_VERIFY, stepIdx, segmentId, -1, slot.ident.opName.c_str(), 0,
        "GDR_INPUT_SANITY backend=%s exec=%d input=%d source=%s dtype=%s "
        "shape=%s strides=%s offset=%lld len=%lld hash=0x%016llx "
        "min=%.9g max=%.9g mean=%.9g l1=%.9g samples=%s "
        "arr=%p db=%p primary=%p special=%p",
        backend, execCount, input, source.c_str(),
        DataTypeUtils::asString(value->dataType()).c_str(),
        dspShapeStr(value).c_str(), strides.c_str(),
        static_cast<long long>(value->offset()),
        static_cast<long long>(value->lengthOf()),
        static_cast<unsigned long long>(summary.valueHash),
        summary.minimum, summary.maximum, mean, summary.l1, samples.c_str(),
        (void*)value, (void*)dataBuffer,
        dataBufferValid ? dataBuffer->primary() : nullptr,
        dataBufferValid ? dataBuffer->special() : nullptr);
  }
}

static bool outputLeavesSegment(const NativeSlot* slots, int numSlots,
                                int segmentEnd, int outputSlot,
                                const int* requestedOutputs,
                                int numRequestedOutputs) {
  for (int requested = 0; requested < numRequestedOutputs; ++requested) {
    if (requestedOutputs != nullptr && requestedOutputs[requested] == outputSlot) return true;
  }
  for (int consumer = segmentEnd + 1; consumer < numSlots; ++consumer) {
    const auto& slot = slots[consumer];
    for (int input = 0; input < slot.wiring.numInputs; ++input) {
      if (slot.wiring.inputSourceIndices[input] == outputSlot) return true;
    }
  }
  return false;
}

static void recordSegmentBoundaryOpSanity(
    const char* backend, const GraphSegment& segment,
    const NativeSlot* slots, int numSlots,
    NDArray** outputSlots, int totalOutputSlots,
    const int* requestedOutputs, int numRequestedOutputs,
    int execCount) {
  if (!opSanityEnabled(execCount)) return;
  std::unordered_set<int> recordedOutputs;
  for (int step = segment.def.startSlot; step <= segment.def.endSlot; ++step) {
    const auto& slot = slots[step];
    for (int output = 0; output < slot.wiring.numOutputs; ++output) {
      const int outputSlot = slot.wiring.outputSlotIndices[output];
      const bool terminalOutput = step == segment.def.endSlot;
      if (!terminalOutput && !outputLeavesSegment(
              slots, numSlots, segment.def.endSlot, outputSlot,
              requestedOutputs, numRequestedOutputs)) {
        continue;
      }
      if (outputSlot < 0 || outputSlot >= totalOutputSlots ||
          !recordedOutputs.insert(outputSlot).second) {
        continue;
      }
      recordOpSanity(
          backend, segment.def.startSlot, step, output, outputSlot,
          slot.ident.opName.c_str(), outputSlots[outputSlot], execCount,
          dspGetGraphCaptureStream() != nullptr);
    }
  }
}

static void appendSlotInputExceptionContext(std::string& msg,
                                            const NativeSlot& slot,
                                            const NativeSlot* slots,
                                            int numSlots,
                                            NDArray** outputSlots,
                                            int totalOutputSlots,
                                            NDArray** externalArrays,
                                            int numExt) {
  msg += " | inputContext=[";
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    if (i > 0) msg += "; ";
    int srcIdx = slot.wiring.inputSourceIndices[i];
    NDArray* input = nullptr;
    const char* sourceKind = "unknown";
    int resolvedIdx = -1;
    int producerStep = -1;
    const char* producerOp = "?";

    if (srcIdx >= 0) {
      sourceKind = "slot";
      resolvedIdx = srcIdx;
      if (srcIdx < totalOutputSlots && outputSlots != nullptr) {
        input = outputSlots[srcIdx];
      }
      if (slots != nullptr) {
        for (int p = 0; p < numSlots; p++) {
          const auto& producer = slots[p];
          for (int o = 0; o < producer.wiring.numOutputs; o++) {
            if (producer.wiring.outputSlotIndices[o] == srcIdx) {
              producerStep = p;
              producerOp = producer.ident.opName.c_str();
              break;
            }
          }
          if (producerStep >= 0) break;
        }
      }
    } else {
      sourceKind = "ext";
      resolvedIdx = -(srcIdx + 1);
      if (resolvedIdx >= 0 && resolvedIdx < numExt && externalArrays != nullptr) {
        input = externalArrays[resolvedIdx];
      }
    }

    char buf[512];
    auto* shapeBuffer = input != nullptr ? input->shapeInfoConstBuffer() : nullptr;
    uintptr_t shapeBufferAddr = reinterpret_cast<uintptr_t>(shapeBuffer);
    size_t shapeBufferAlign = shapeBuffer != nullptr
        ? static_cast<size_t>(shapeBufferAddr % alignof(ConstantShapeBuffer))
        : 0;
    DataBuffer* db = nullptr;
    if (input != nullptr) {
      try {
        db = input->dataBuffer();
      } catch (...) {
        db = nullptr;
      }
    }
    snprintf(buf, sizeof(buf),
             "input[%d] src=%s[%d] producerStep=%d producerOp=%s arr=%p "
             "shapeBuf=%p shapeBufAlignOff=%zu db=%p",
             i, sourceKind, resolvedIdx, producerStep, producerOp,
             (void*)input, (void*)shapeBuffer, shapeBufferAlign, (void*)db);
    msg += buf;
  }
  msg += "]";
}

}  // namespace

// ─── Segment shape key computation ──────────────────────────────────────────

LongType NativeDynamicShapePlan::computeSegmentShapeKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {

  // ── Frozen fast path: reuse cached key if shapes can't change ──
  // This is the AUTHORITATIVE cache check — applies to ALL callers
  // (phaseCompile, executeSegmentWithGpuGraph, executeSegmentWithSpecificBackend, etc.)
  const bool directBackendNeedsLiveLayoutKey =
      seg.resolvedGraphBackendPolicy.artifactKind ==
          GraphBackendArtifactKind::DIRECT_COMPILED;
  if (!planLifecycle_.isSlotBySlot() && seg.exec.cachedShapeKey != 0 &&
      !seg.def.hasDynamicBoundaryInputs &&
      !directBackendNeedsLiveLayoutKey) {
    return seg.exec.cachedShapeKey;
  }

  // Requested outputs are part of compiled-kernel semantics: a value consumed
  // only inside a fused segment still has to be materialized when the caller
  // asks for it. Mix the sorted unique slot set into every shape-key variant so
  // process-wide backend caches cannot reuse a final-only kernel for a plan that
  // also requests a fused intermediate.
  auto mixRequestedOutputSet = [this](uint64_t& hash) {
    std::vector<int> requestedSlots;
    if (requestedOutputSlotIndices_ != nullptr && numRequestedOutputs_ > 0) {
      requestedSlots.reserve(static_cast<size_t>(numRequestedOutputs_));
      for (int i = 0; i < numRequestedOutputs_; i++) {
        int slot = requestedOutputSlotIndices_[i];
        if (slot >= 0) requestedSlots.push_back(slot);
      }
    }
    std::sort(requestedSlots.begin(), requestedSlots.end());
    requestedSlots.erase(std::unique(requestedSlots.begin(), requestedSlots.end()),
                         requestedSlots.end());
    dsp::fnv1aMixValue(hash, uint64_t{0x52514f5554505554ULL});  // "RQOUTPUT"
    dsp::fnv1aMixValue(hash, static_cast<uint64_t>(requestedSlots.size()));
    for (int slot : requestedSlots) {
      dsp::fnv1aMixValue(hash, static_cast<uint64_t>(slot));
    }
  };

  // ── Symbolic shape range path ──────────────────────────────────────────
  // When enabled, collect cross-segment inputs, feed them to the shape
  // profiler, and (after warmup) use range-based hashing that ignores
  // dynamic dimensions.
  if (!seg.def.hasDynamicBoundaryInputs &&
      seg.exec.symbolicShapeEnabled && seg.exec.symbolicRangeData != nullptr) {
    auto* profile = static_cast<SegmentShapeProfile*>(seg.exec.symbolicRangeData);

    // Collect cross-segment input arrays (same logic as standard path below)
    std::unordered_set<int> segOutputSlots;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        segOutputSlots.insert(slot.wiring.outputSlotIndices[i]);
      }
    }

    std::vector<NDArray*> crossInputs;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
            crossInputs.push_back(externalInputs[extIdx]);
          }
        } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
          if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
            crossInputs.push_back(outputSlots_[srcIdx]);
          }
        }
      }
    }

    // Record observations during warmup.
    // When planLifecycle_.isShapesFrozen(), shapes are constant — force warmup complete immediately
    // by recording one observation (sufficient for frozen shapes) and skip waiting
    // for the normal 2-observation warmup cycle.
    if (!isWarmupComplete(profile)) {
      recordObservedShapes(profile, crossInputs.data(),
                           static_cast<int>(crossInputs.size()));
      if (!planLifecycle_.isSlotBySlot() && !isWarmupComplete(profile)) {
        // Frozen shapes won't change, so one observation is enough.
        // Record again to satisfy the 2-step warmup requirement immediately.
        recordObservedShapes(profile, crossInputs.data(),
                             static_cast<int>(crossInputs.size()));
        DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] fast-completed warmup (shapesFrozen)",
                 seg.def.startSlot, seg.def.endSlot);
      } else {
        DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] observation %d/%d",
                 seg.def.startSlot, seg.def.endSlot,
                 getObservationCount(profile), getWarmupSteps(profile));
      }
    }

    // After warmup, use range-based key
    if (isWarmupComplete(profile)) {
      LongType rangeKey = computeRangeBasedShapeKey(
          profile, crossInputs.data(), static_cast<int>(crossInputs.size()),
          seg.def.startSlot, seg.def.endSlot);

      // Mix op names, iArgs, and tArgs into the range-based key so different
      // plans with the same input shapes but different ops produce unique keys
      // in singleton backend caches (OpenVINO, OneDNN Graph). v2-cache-fix.
      uint64_t rangeKeyU64 = static_cast<uint64_t>(rangeKey);
      auto mixRange = [&rangeKeyU64](LongType val) {
        dsp::fnv1aMixValue(rangeKeyU64, static_cast<uint64_t>(val));
      };
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        mixRange(slot.ident.opHash);
        mixRange(static_cast<LongType>(slot.args.numIArgs));
        for (int a = 0; a < slot.args.numIArgs; a++) {
          mixRange(static_cast<LongType>(slot.args.iArgs[a]));
        }
        mixRange(static_cast<LongType>(slot.args.numTArgs));
        for (int a = 0; a < slot.args.numTArgs; ++a) {
          LongType bits = 0;
          static_assert(sizeof(bits) == sizeof(double),
                        "LongType must preserve a double argument bit pattern");
          std::memcpy(&bits, &slot.args.tArgs[a], sizeof(bits));
          mixRange(bits);
        }
        mixRange(static_cast<LongType>(slot.args.numBArgs));
        for (int a = 0; a < slot.args.numBArgs; ++a) {
          mixRange(slot.args.bArgs[a] ? 1 : 0);
        }
        mixRange(static_cast<LongType>(slot.args.numDArgs));
        for (int a = 0; a < slot.args.numDArgs; ++a) {
          mixRange(static_cast<LongType>(slot.args.dArgs[a]));
        }
        mixRange(static_cast<LongType>(slot.args.numSArgs));
        for (int a = 0; a < slot.args.numSArgs; ++a) {
          for (char character : slot.args.sArgs[a]) {
            mixRange(static_cast<unsigned char>(character));
          }
          mixRange(0xFF);
        }
      }
      for (NDArray* input : crossInputs) {
        if (input == nullptr) continue;
        mixRange(static_cast<LongType>(input->dataType()));
        mixRange(static_cast<LongType>(input->rankOf()));
        mixRange(static_cast<LongType>(input->ordering()));
        for (int dimension = 0; dimension < input->rankOf(); ++dimension) {
          // A stride on a singleton dimension is not observable: that index is
          // always zero. Normalize it so canonical placeholder staging does not
          // invalidate a kernel compiled from an equivalent caller layout.
          mixRange(input->sizeAt(dimension) <= 1 ? 0 : input->strideAt(dimension));
        }
      }
      mixRequestedOutputSet(rangeKeyU64);

      DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] using range-based key=%lld (with-op-mix)",
               seg.def.startSlot, seg.def.endSlot, static_cast<long long>(rangeKeyU64));
      // NOTE: Do NOT set seg.exec.cachedShapeKey here. cachedShapeKey must only be
      // written after a successful compile+execute in executeSegmentWithSpecificBackend().
      // Writing it here causes the cascade to skip compilation for fallback backends
      // when the first backend fails (cachedShapeKey != 0 → needsCompile = false).
      return static_cast<LongType>(rangeKeyU64);
    }
    // Fall through to standard path during warmup
  }

  // ── Standard FNV-1a path ───────────────────────────────────────────────
  uint64_t key = dsp::FNV1A64_OFFSET_BASIS;
  auto mix = [&key](LongType val) {
    dsp::fnv1aMixValue(key, static_cast<uint64_t>(val));
  };

  // Hash the exact array layout contract used by compiled backends.
  auto mixArraySignature = [&](NDArray* arr) {
    if (arr == nullptr) return;

    const LongType* si = arr->shapeInfo();
    LongType rank = shape::rank(si);
    mix(rank);
    for (int d = 0; d < rank; d++) {
      mix(si[d + 1]);
      // Singleton-dimension strides are semantically irrelevant and can be
      // canonicalized by staging without changing the tensor layout.
      mix(si[d + 1] <= 1 ? 0 : arr->strideAt(d));
    }
    mix(static_cast<LongType>(arr->lengthOf()));
    mix(static_cast<LongType>(arr->dataType()));
    mix(static_cast<LongType>(arr->ordering()));
  };

  mix(seg.def.startSlot);
  mix(seg.def.endSlot);
  mixRequestedOutputSet(key);

  // Mix op names so different plans with same slot indices + shapes don't collide
  // in singleton backend caches (e.g. OpenVINO, OneDNN Graph)
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    mix(slot.ident.opHash);
    mix(static_cast<LongType>(slot.wiring.numInputs));
    mix(static_cast<LongType>(slot.wiring.numOutputs));
    mix(static_cast<LongType>(slot.args.numIArgs));
    // Mix actual iArg values (e.g. reshape target shape, axis indices)
    for (int a = 0; a < slot.args.numIArgs; a++) {
      mix(static_cast<LongType>(slot.args.iArgs[a]));
    }
    mix(static_cast<LongType>(slot.args.numTArgs));
    for (int a = 0; a < slot.args.numTArgs; ++a) {
      LongType bits = 0;
      static_assert(sizeof(bits) == sizeof(double),
                    "LongType must preserve a double argument bit pattern");
      std::memcpy(&bits, &slot.args.tArgs[a], sizeof(bits));
      mix(bits);
    }
    mix(static_cast<LongType>(slot.args.numBArgs));
    for (int a = 0; a < slot.args.numBArgs; ++a) {
      mix(slot.args.bArgs[a] ? 1 : 0);
    }
    mix(static_cast<LongType>(slot.args.numDArgs));
    for (int a = 0; a < slot.args.numDArgs; ++a) {
      mix(static_cast<LongType>(slot.args.dArgs[a]));
    }
    mix(static_cast<LongType>(slot.args.numSArgs));
    for (int a = 0; a < slot.args.numSArgs; ++a) {
      for (char character : slot.args.sArgs[a]) {
        mix(static_cast<unsigned char>(character));
      }
      mix(0xFF);
    }
  }

  std::unordered_set<int> segOutputSlots;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numOutputs; i++) {
      segOutputSlots.insert(slot.wiring.outputSlotIndices[i]);
    }
  }

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
          mixArraySignature(externalInputs[extIdx]);
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          mixArraySignature(outputSlots_[srcIdx]);
        }
      }
    }
  }

  // Segment identity is shape/layout based. Value-dependent slots validate their
  // bounded controls on the live slot path; hashing every small integral external
  // here would turn unrelated token/payload changes into segment recompiles.

  // NOTE: Do NOT set seg.exec.cachedShapeKey here. It must only be written after
  // a successful compile+execute in executeSegmentWithSpecificBackend() (line ~681).
  // Writing it here causes the cascade to skip compilation for fallback backends.
  return key;
}

// ─── Compiled graph-backend catalog and resolution ──────────────────────────

GraphBackendRequest NativeDynamicShapePlan::makeGraphBackendRequest() const {
  return GraphBackendRequest{
      graphExecutionMode_,
      runtimeCompilationAllowed_,
      runtimeArtifactDirectory_,
      deviceCompilationCacheDirectory_,
      deviceCompilationCacheModelKey_};
}

const std::vector<GraphBackend*>& NativeDynamicShapePlan::getGraphBackendCandidates() {
  if (graphBackendCandidatesBuilt_) return graphBackendCandidates_;
  graphBackendCandidatesBuilt_ = true;

  std::vector<GraphBackend*> catalog;
#if !defined(SD_VULKAN) && !defined(SD_CUDA) && !defined(SD_HIP)
#if HAVE_MLX
  catalog.push_back(&MlxGraphBackend::getInstance());
#endif
#if HAVE_OPENVINO
  catalog.push_back(&OpenVinoGraphBackend::getInstance());
#endif
#if HAVE_ONEDNN
  catalog.push_back(&OneDnnGraphBackend::getInstance());
#endif
#if HAVE_ARMCOMPUTE
  catalog.push_back(&AclGraphBackend::getInstance());
#endif
#if HAVE_NNAPI
  catalog.push_back(&NnapiGraphBackend::getInstance());
#endif
#if HAVE_MLIR
  // MLIR CPU backends are not an implicit third compiler tier. Production
  // x86 and ARM chains end in explicit functional replay after their two
  // target backends. Keep MLIR available only through an explicit priority
  // request until it has a dedicated execution mode.
  const auto explicitlyRequests = [this](const char* backendName) {
    return std::find(backendPriority_.begin(), backendPriority_.end(),
                     backendName) != backendPriority_.end();
  };
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
  if (explicitlyRequests("ARM MLIR CPU")) {
    catalog.push_back(&ArmHybridGraphBackend::getInstance());
  }
#endif
  if (explicitlyRequests("MLIR CPU JIT")) {
    catalog.push_back(&MlirCpuGraphBackend::getInstance());
  }
#endif
#endif
  catalog.push_back(dspGetTritonBackend());
  catalog.push_back(dspGetNvrtcBackend());
  catalog.push_back(dspGetPtxBackend());
#ifdef SD_TPU
  catalog.push_back(&TpuGraphBackend::getInstance());
#endif
#ifdef HAVE_HEXAGON_MLIR
  catalog.push_back(&HexagonGraphBackend::getInstance());
#endif
#if defined(SD_HIP)
  catalog.push_back(&HipGraphBackend::getInstance());
#endif

  const GraphBackendRequest request = makeGraphBackendRequest();
  graphBackendCandidates_ = GraphBackendResolver::resolve(request, catalog);

  // Apply the plan-local priority override after capability/mode resolution.
  // Backends named in the override come first in the requested order; all
  // remaining candidates retain the resolver's stable priority order.
  if (!backendPriority_.empty() && graphBackendCandidates_.size() > 1) {
    auto overrideRank = [this](const GraphBackend* backend) {
      const auto found = std::find(
          backendPriority_.begin(), backendPriority_.end(), backend->name());
      return found == backendPriority_.end()
                 ? backendPriority_.size()
                 : static_cast<size_t>(found - backendPriority_.begin());
    };
    std::stable_sort(
        graphBackendCandidates_.begin(), graphBackendCandidates_.end(),
        [&overrideRank](const GraphBackend* left, const GraphBackend* right) {
          return overrideRank(left) < overrideRank(right);
        });
  }

  DSP_DIAG(BACKEND,
           "graph backend resolver: mode=%d catalog=%d candidates=%d",
           static_cast<int>(graphExecutionMode_), static_cast<int>(catalog.size()),
           static_cast<int>(graphBackendCandidates_.size()));
  for (size_t i = 0; i < graphBackendCandidates_.size(); ++i) {
    GraphBackend* backend = graphBackendCandidates_[i];
    DSP_DIAG(BACKEND,
             "  candidate[%d]=%s priority=%d",
             static_cast<int>(i), backend->name(),
             backend->resolutionPriority(request));
  }

  return graphBackendCandidates_;
}

GraphBackendPlanningPolicy
NativeDynamicShapePlan::getResolvedGraphBackendPlanningPolicy() {
  const GraphBackendRequest request = makeGraphBackendRequest();
  const auto& candidates = getGraphBackendCandidates();
  return GraphBackendResolver::aggregatePlanningPolicy(request, candidates);
}

GraphBackendExecutionPolicy
NativeDynamicShapePlan::getResolvedGraphBackendExecutionPolicy() {
  const GraphBackendRequest request = makeGraphBackendRequest();
  const auto& candidates = getGraphBackendCandidates();
  return GraphBackendResolver::aggregateExecutionPolicy(request, candidates);
}

// ─── Segment execution: generic graph backend cascade ─────────────────────

DspExecutionResult NativeDynamicShapePlan::executeSegmentWithGraphBackend(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SLOT_BY_SLOT, "executeSegmentWithGraphBackend");

  // Resolution/admission failures have one backend-neutral contract:
  // return control to the platform dispatcher, which owns the replay fallback.
  if (seg.exec.compilationFailed || seg.exec.noFusibleOps) {
    DSP_DIAG(BACKEND,
             "graph backend cascade unavailable for seg[%d-%d]: failed=%d notFusible=%d",
             seg.def.startSlot, seg.def.endSlot,
             static_cast<int>(seg.exec.compilationFailed),
             static_cast<int>(seg.exec.noFusibleOps));
    return Status::KERNEL_FAILURE;
  }

  // Resolve the concrete segment once through the shared admission gate.
  // A sticky backend remains preferred, but fallback order stays resolver-owned.
  // Cascade through resolver-ordered candidates.
  const auto& chain = getGraphBackendCandidates();
  if (chain.empty()) {
    DSP_DIAG(BACKEND,
             "graph backend resolver produced no candidates for mode=%d seg[%d-%d]",
             static_cast<int>(graphExecutionMode_), seg.def.startSlot,
             seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // Admission is checked before warmup so a rejected large or unsupported
  // segment cannot trigger hidden slot-by-slot execution just to discover that
  // no backend can compile it.
  const GraphBackendRequest request = makeGraphBackendRequest();
  const auto admitted = GraphBackendResolver::resolveSegment(
      request, chain, slots_, seg.def.startSlot, seg.def.endSlot,
      seg.resolvedGraphBackend);
  if (admitted.empty()) {
    SegmentLifecycle::markNotFusible(seg.exec, "cascade_admission_rejected",
                                     seg.def.startSlot, seg.def.endSlot);
    DSP_DIAG(BACKEND,
             "cascade: no backend admitted seg[%d-%d]; requesting explicit replay",
             seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // Some compiled backends need values from one complete, internally consistent
  // functional execution before any segment is lowered. Do not compile and run
  // early segments while later calibration tensors are still being produced: that
  // both contaminates calibration and retains a throwaway model-sized artifact set
  // that must be rebuilt after freeze. The plan-level compile phase owns lowering.
  const auto cascadePlanningPolicy =
      GraphBackendResolver::aggregatePlanningPolicy(request, chain);
  // The plan-level predicate deliberately treats REPLAYING as frozen. Checking
  // PlanLifecycle::isShapesFrozen() directly would become false again after the
  // SHAPES_FROZEN -> REPLAYING transition and would incorrectly run provisional
  // host calibration forever instead of the committed graph backend.
  if (cascadePlanningPolicy.deferCompilationUntilPlanFreeze &&
      !isShapesFrozen()) {
    if (!cascadePlanningPolicy.requiresPrecommitFunctionalWarmup) {
      DSP_DIAG(COMPILE,
               "GRAPH_BACKEND_PRECOMMIT_REJECTED seg[%d-%d] deferred compilation "
               "has no explicit functional-warmup contract",
               seg.def.startSlot, seg.def.endSlot);
      return DspExecutionResult(Status::BAD_GRAPH, false, false);
    }
    ++seg.exec.precommitFunctionalWarmupCount;
    DSP_DIAG(BACKEND,
             "PRECOMMIT_FUNCTIONAL_WARMUP seg[%d-%d] count=%d "
             "owner=provisional backend=calibration",
             seg.def.startSlot, seg.def.endSlot,
             seg.exec.precommitFunctionalWarmupCount);
    auto warmupStatus =
        executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    DSP_DIAG(
        EXECUTE,
        "GRAPH_BACKEND_FUNCTIONAL_WARMUP seg[%d-%d] status=%s (%d) "
        "executionCount=%d compile=deferred_until_plan_freeze",
        seg.def.startSlot, seg.def.endSlot, statusName_seg(warmupStatus),
        static_cast<int>(warmupStatus),
        seg.exec.executionCount);
    if (warmupStatus == Status::OK && seg.exec.segPhase.needsWarmup()) {
      SegmentLifecycle::markWarmupDone(seg.exec);
    }
    return warmupStatus;
  }

  // Warmup must happen before any backend tries to compile (needs output shapes)
  bool warmedUpThisCall = false;
  if (seg.exec.executionCount == 0) {
    auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    DSP_DIAG(EXECUTE, "executeSegmentWithGraphBackend: warmup %s for seg[%d-%d], executionCount→%d",
             warmupStatus == Status::OK ? "OK" : "FAILED",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
    if (warmupStatus != Status::OK) {
      return warmupStatus;
    }
    if (seg.exec.segPhase.needsWarmup()) {
      SegmentLifecycle::markWarmupDone(seg.exec);
    }
    warmedUpThisCall = true;
  }

  // Try each admitted backend in shared resolver order.
  for (size_t i = 0; i < admitted.size(); i++) {
    GraphBackend* backend = admitted[i];
    const char* backendName = backend->name();
    const bool hadCommittedDirectArtifact =
        backend == seg.resolvedGraphBackend &&
        seg.resolvedGraphBackendPolicy.artifactKind ==
            GraphBackendArtifactKind::DIRECT_COMPILED &&
        seg.compiledGraphBackendArtifactOwner == backend &&
        seg.compiledGraphBackendArtifact != nullptr;
    const LongType attemptedArtifactShapeKey = hadCommittedDirectArtifact
        ? seg.compiledGraphBackendArtifactShapeKey : 0;

    // Attempt lower + validate + execute with this backend.
    DspExecutionResult result = executeSegmentWithSpecificBackend(
        seg, backend, externalArrays, numExt, stream);
    if (result.ok()) {
      // Cache backend identity and its exact lifecycle policy atomically.
      seg.setResolvedGraphBackend(backend, request);
      DSP_DIAG(BACKEND,
               "cascade: seg[%d-%d] resolved to backend=%s "
               "(admitted position %d/%d)",
               seg.def.startSlot, seg.def.endSlot, backendName,
               static_cast<int>(i) + 1, static_cast<int>(admitted.size()));
      return result;
    }

    // Once a direct artifact is committed, the resolver no longer owns a
    // candidate choice. A descriptor, calibration, or execution rejection is
    // terminal even when it occurs before the device API starts: compiling a
    // different backend during REPLAYING would violate ownership and could
    // silently change numerical semantics mid-session.
    const bool committedDirectArtifactAfterAttempt =
        backend == seg.resolvedGraphBackend &&
        seg.resolvedGraphBackendPolicy.artifactKind ==
            GraphBackendArtifactKind::DIRECT_COMPILED &&
        seg.compiledGraphBackendArtifactOwner == backend &&
        seg.compiledGraphBackendArtifact != nullptr &&
        seg.compiledGraphBackendArtifactShapeKey ==
            seg.def.shapeKeyState.lastComputedKey;
    const bool attemptedArtifactMatchesCurrentShape =
        attemptedArtifactShapeKey != 0 &&
        attemptedArtifactShapeKey == seg.def.shapeKeyState.lastComputedKey;
    if (attemptedArtifactMatchesCurrentShape ||
        committedDirectArtifactAfterAttempt) {
      DSP_DIAG(BACKEND,
               "COMMITTED_GRAPH_BACKEND_FAILURE seg[%d-%d] backend=%s "
               "status=%s (%d) terminal=1",
               seg.def.startSlot, seg.def.endSlot, backendName,
               statusName_seg(result.status),
               static_cast<int>(result.status));
      return DspExecutionResult(result.status, true, false);
    }

    if (hadCommittedDirectArtifact &&
        attemptedArtifactShapeKey != seg.def.shapeKeyState.lastComputedKey) {
      DSP_DIAG(BACKEND,
               "STALE_GRAPH_BACKEND_ARTIFACT_REJECTED seg[%d-%d] backend=%s "
               "artifactKey=%lld currentKey=%lld terminal=0",
               seg.def.startSlot, seg.def.endSlot, backendName,
               static_cast<long long>(attemptedArtifactShapeKey),
               static_cast<long long>(seg.def.shapeKeyState.lastComputedKey));
    }

    if (!result.preExecutionRejection()) {
      DSP_DIAG(BACKEND,
               "cascade: backend=%s execution failed for seg[%d-%d] "
               "(status=%s (%d)); post-start failover is forbidden",
               backendName, seg.def.startSlot, seg.def.endSlot,
               statusName_seg(result.status),
               static_cast<int>(result.status));
      return result;
    }

    DSP_DIAG(BACKEND,
             "cascade: backend=%s rejected lowering for seg[%d-%d], "
             "status=%s (%d); trying next admitted backend",
             backendName, seg.def.startSlot, seg.def.endSlot,
             statusName_seg(result.status),
             static_cast<int>(result.status));
    seg.resetGraphBackend();
    // compilationFailed is managed by lifecycle — no raw reset needed here.
    // The markFailed() call below handles the terminal case; individual backend
    // failures don't set compilationFailed since the cascade continues.
  }

  // At least one candidate admitted the metadata but every concrete lowering
  // rejected its dtype/layout/argument contract. AUTO/ARM hybrid own an explicit
  // third tier, selected here before any backend execution starts.
  if (ModeContract::forMode(graphExecutionMode_).allowsFallback) {
    seg.resetGraphBackend();
    seg.def.selectedBackend = SelectedBackend::EMULATED_REPLAY;
    SegmentLifecycle::prepareFunctionalReplayHandoff(
        seg.exec, seg.def.startSlot, seg.def.endSlot);
    DSP_DIAG(BACKEND,
             "cascade: concrete lowering rejected seg[%d-%d]; ownership -> explicit replay",
             seg.def.startSlot, seg.def.endSlot);
    if (warmedUpThisCall) {
      return DspExecutionResult(Status::OK, false, true);
    }
    return executeSegmentEmulatedReplay(
        seg, externalArrays, numExt, stream);
  }

  SegmentLifecycle::markFailed(seg.exec, "cascade_all_backends_failed",
                               seg.def.startSlot, seg.def.endSlot);
  DSP_DIAG(COMPILE,
           "graph backend cascade: all %d candidates failed lowering seg[%d-%d]",
           static_cast<int>(admitted.size()), seg.def.startSlot,
           seg.def.endSlot);
  recordFailureDetailIfMissing(
      Status::KERNEL_FAILURE,
      "graph backend lowering failed for segment [" +
          std::to_string(seg.def.startSlot) + "-" +
          std::to_string(seg.def.endSlot) +
          "]: every admitted backend rejected the concrete dtype/layout/argument contract");
  return Status::KERNEL_FAILURE;
}

// ─── Execute segment with a specific backend (shared logic) ─────────────────

DspExecutionResult NativeDynamicShapePlan::executeSegmentWithSpecificBackend(
    GraphSegment& seg, GraphBackend* backend, NDArray** externalArrays, int numExt,
    void* stream) {
  DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SLOT_BY_SLOT, "executeSegmentWithSpecificBackend");

  const char* backendName = backend->name();
  const GraphBackendRequest request = makeGraphBackendRequest();

  // ── Shape key: detect if segment needs recompilation ──
  // Frozen + cached key: reuse (shapes can't change). Otherwise: compute and cache.
  // NOTE: cachedShapeKey is only set AFTER successful compilation (below), not here.
  // Setting it before compile would cause the cascade to skip compilation for the
  // next backend when the first backend fails (the key would be non-zero but no
  // compiled segment exists in the next backend's cache).
  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
  seg.def.shapeKeyState.recordComputed(segShapeKey);
  const bool backendIdentityChanged = seg.resolvedGraphBackend != backend;
  bool needsCompile = seg.exec.segPhase.needsCompile() ||
                      seg.def.shapeKeyState.hasDrifted() ||
                      backendIdentityChanged;
  const bool isRecompileDueToShapeChange =
      seg.def.shapeKeyState.hasDrifted();
  bool invocationSatisfiedByWarmup = false;

  // ── Phase guard: compilation must not happen during REPLAYING ────────────
  if (needsCompile && planLifecycle_.isReplaying() && !isRecompileDueToShapeChange) {
    DSP_DIAG(COMPILE,
             "ERROR: CPU backend compilation triggered during REPLAYING phase for seg[%d-%d] "
             "(executionCount=%d, planPhase=%s). Demoting plan phase.",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
             planLifecycle_.displayName());
    REQUIRE_TRUE(false, 0,
                 "DSP phase contract violation: CPU compilation during REPLAYING phase "
                 "for seg[%d-%d].", seg.def.startSlot, seg.def.endSlot);
    demotePlanPhase(PlanPhase::SHAPES_FROZEN,
                    "CPU compilation triggered during REPLAYING phase");
  }

  if (isRecompileDueToShapeChange) {
    DSP_SEG_EVENT(seg, RECOMPILE_TRIGGERED,
                  "dynamic boundary shape changed; refreshing segment before backend compile");
    SegmentLifecycle::invalidateSegmentCaptures(this, seg, "cpu_shape_change");
    platformResetGapCaches();
    platformResetBatchD2D();
    ShapeChangeWarmupGuard warmupGuard(*this, seg.def.startSlot, seg.def.endSlot);
    Status warmupStatus = executeSegmentSlotBySlot(
        seg, externalArrays, numExt, stream);
    if (warmupStatus != Status::OK) return warmupStatus;
    SegmentLifecycle::markWarmupDone(seg.exec);
    invocationSatisfiedByWarmup = true;
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    seg.def.shapeKeyState.recordComputed(segShapeKey);
    needsCompile = true;
  }

  if (needsCompile) {
    DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                 "seg[%d-%d] needs compile: %s (execCount=%d shapeKey=%lld->%lld backend=%s)",
                 seg.def.startSlot, seg.def.endSlot,
                 backendIdentityChanged ? "backend-identity-missing-or-changed" :
                 (seg.exec.executionCount == 1 ? "first-compile" : "shape-key-changed"),
                 seg.exec.executionCount,
                 static_cast<long long>(seg.def.shapeKeyState.compiledShapeKey),
                 static_cast<long long>(segShapeKey),
                 backendName);
  }
  if (needsCompile) {
    const auto lowering = GraphBackendResolver::lowerSegment(
        request, std::vector<GraphBackend*>{backend}, backend, seg, slots_,
        seg.def.startSlot, seg.def.endSlot, externalArrays, numExt,
        outputSlots_, totalOutputSlots_, segShapeKey, numSlots_,
        requestedOutputSlotIndices_, numRequestedOutputs_);
    if (!lowering.succeeded()) {
      if (!lowering.attempts.empty()) {
        lastCompilationAudit_ = lowering.attempts.back().audit;
      }
      if (lowering.prerequisiteBlocked()) {
        const std::string message =
            std::string("graph backend compilation prerequisite missing for ") +
            lowering.prerequisiteBlockedBackend->name() + " seg[" +
            std::to_string(seg.def.startSlot) + "-" +
            std::to_string(seg.def.endSlot) + "]: " +
            lowering.prerequisiteFailureReason;
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(
            message.c_str());
        DSP_DIAG(COMPILE,
                 "GRAPH_BACKEND_PREREQUISITE_BLOCKED backend=%s seg[%d-%d] "
                 "reason=%s terminal=1",
                 lowering.prerequisiteBlockedBackend->name(),
                 seg.def.startSlot, seg.def.endSlot,
                 lowering.prerequisiteFailureReason.c_str());
        return Status::KERNEL_FAILURE;
      }
      DSP_DIAG(
          COMPILE,
          "executeSegmentWithSpecificBackend: backend=%s lowering failed "
          "for seg[%d-%d]",
          backendName, seg.def.startSlot, seg.def.endSlot);
      return Status::BAD_GRAPH;
    }
    backend = lowering.backend;
    backendName = backend->name();
    seg.setResolvedGraphBackend(backend, request);
    seg.compilationAudit = lowering.attempts.back().audit;
    lastCompilationAudit_ = seg.compilationAudit;
  }

  if (needsCompile) {
    const auto& audit = seg.compilationAudit;
    lastCompilationAudit_ = audit;
    const int expectedAuditEntries = seg.def.endSlot - seg.def.startSlot + 1;
    bool allCovered = static_cast<int>(audit.size()) == expectedAuditEntries;
    int compiledCount = 0, nativeHandledCount = 0, uncoveredCount = 0;
    std::vector<bool> auditedSlots(static_cast<size_t>(expectedAuditEntries), false);
    for (const auto& entry : audit) {
      if (entry.slotIndex < seg.def.startSlot || entry.slotIndex > seg.def.endSlot) {
        allCovered = false;
        uncoveredCount++;
        DSP_DIAG(COMPILE,
                 "%s VALIDATION: audit entry references out-of-range slot %d for seg[%d-%d]",
                 backendName, entry.slotIndex, seg.def.startSlot, seg.def.endSlot);
        continue;
      }
      const size_t auditIndex =
          static_cast<size_t>(entry.slotIndex - seg.def.startSlot);
      if (auditedSlots[auditIndex]) {
        allCovered = false;
        uncoveredCount++;
        DSP_DIAG(COMPILE, "%s VALIDATION: duplicate audit entry for slot %d",
                 backendName, entry.slotIndex);
        continue;
      }
      auditedSlots[auditIndex] = true;
      if (entry.wasCompiled) {
        compiledCount++;
      } else if (entry.isNativeHandled) {
        // Backend owns native execution for this op (e.g., mixed-segment interleaving).
        // This is NOT an error — the backend guarantees correct execution.
        nativeHandledCount++;
        DSP_DIAG(COMPILE, "%s VALIDATION: slot %d (%s) natively handled by backend: %s",
                  backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      } else {
        allCovered = false;
        uncoveredCount++;
        DSP_DIAG(COMPILE, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                  backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    for (int offset = 0; offset < expectedAuditEntries; ++offset) {
      if (!auditedSlots[static_cast<size_t>(offset)]) {
        allCovered = false;
        uncoveredCount++;
        DSP_DIAG(COMPILE, "%s VALIDATION: missing audit entry for slot %d",
                 backendName, seg.def.startSlot + offset);
      }
    }
    if (!allCovered) {
      SegmentLifecycle::markFailed(seg.exec, "validation_ops_not_covered",
                                   seg.def.startSlot, seg.def.endSlot);
      DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                    "%s VALIDATION FAILURE: segment [%d-%d] has %d ops not covered by backend. "
                    "Fix the backend to compile or natively handle all ops — silent fallback is not permitted.",
                    backendName, seg.def.startSlot, seg.def.endSlot, uncoveredCount);
    } else {
      DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                   "%s VALIDATION OK: seg[%d-%d] all %d ops covered "
                   "(compiled=%d nativeHandled=%d)",
                   backendName, seg.def.startSlot, seg.def.endSlot, (int)audit.size(),
                   compiledCount, nativeHandledCount);
    }
  }

  // Only update compiledShapeKey when compilation actually occurred.
  // Without this guard, a no-compile reuse call overwrites compiledShapeKey with
  // the current segShapeKey even though the backend's compiled artifacts were
  // produced for a DIFFERENT key. In the CPU cascade (OneDNN → OpenVINO), the
  // second backend reads compiledShapeKey via neverCompiled() / hasDrifted() —
  // an unconditional write here masks compile failures from the first backend.
  if (needsCompile) {
    seg.def.shapeKeyState.markCompiled(segShapeKey);
  }
  if (seg.resolvedGraphBackendPolicy.artifactKind ==
          GraphBackendArtifactKind::DIRECT_COMPILED &&
      (seg.compiledGraphBackendArtifactOwner != backend ||
       seg.compiledGraphBackendArtifactShapeKey != segShapeKey ||
       !seg.compiledGraphBackendArtifact)) {
    DSP_DIAG(COMPILE,
             "executeSegmentWithSpecificBackend: backend=%s accepted seg[%d-%d] "
             "without publishing its segment-owned direct artifact",
             backendName, seg.def.startSlot, seg.def.endSlot);
    return Status::BAD_GRAPH;
  }

  // A shape-drift warmup already produced this invocation's stateful output.
  // The replacement artifact is compiled and audited above, but executing it
  // now would apply in-place/scatter/accumulation effects twice. Publish the
  // compiled lifecycle state and defer execution/capture to the next call.
  if (invocationSatisfiedByWarmup) {
    if (!seg.exec.segPhase.isSealed()) {
      if (seg.resolvedGraphBackendPolicy.artifactKind ==
              GraphBackendArtifactKind::DIRECT_COMPILED &&
          seg.compiledGraphBackendArtifactOwner == backend &&
          seg.compiledGraphBackendArtifactShapeKey == segShapeKey &&
          seg.compiledGraphBackendArtifact) {
        SegmentLifecycle::markDirectGraphBackendCompiled(
            seg.exec, backendName, segShapeKey,
            seg.def.startSlot, seg.def.endSlot);
      } else if (seg.resolvedGraphBackendPolicy.artifactKind ==
                     GraphBackendArtifactKind::BACKEND_REPLAY_HANDLE &&
                 seg.exec.replayHandle && seg.exec.replayHandle->isReady()) {
        SegmentLifecycle::markBackendReplayHandleSealed(
            seg.exec, backendName, segShapeKey,
            seg.def.startSlot, seg.def.endSlot);
      } else if (seg.exec.segPhase.needsCompile()) {
        SegmentLifecycle::markCompiled(seg.exec, backendName, segShapeKey);
      }
    }
    DSP_DIAG(EXECUTE,
             "SHAPE_CHANGE_WARMUP_OUTPUT_COMMITTED: seg[%d-%d] key=%lld — "
             "compiled execution deferred to next invocation",
             seg.def.startSlot, seg.def.endSlot,
             static_cast<long long>(segShapeKey));
    return DspExecutionResult(Status::OK, false, true);
  }

  // tl_graphExecutionActive must NOT be set here — it is a CUDA-graph-capture
  // guard that suppresses frees and skips syncs. This function drives non-capture
  // paths (CPU backends, Triton warmup). Capture manages the flag internally.
  DSP_DIAG(EXECUTE, "PRE-EXECUTE: seg[%d-%d] backend=%s shapeKey=%lld",
           seg.def.startSlot, seg.def.endSlot, backendName, (long long)segShapeKey);

#if !defined(SD_VULKAN)
  bool nativeExecutionStarted = false;
  // NativeSlotExecutor callback — shared by OneDNN and OpenVINO backends.
  // Uses persistent GraphSegments per native range so FunctionalReplayHandle
  // accumulates and CPU_FROZEN_REPLAY fires after the first call.
  auto nativeSlotCallback = [this, &externalArrays, numExt, &stream, &backendName,
                             &nativeExecutionStarted]
      (int nativeStart, int nativeEnd) -> Status {
    auto key = nativeRangeKey(nativeStart, nativeEnd);
    auto& nativeSeg = nativeRangeSegments_[key];
    if (nativeSeg.exec.executionCount == 0) {
      nativeSeg.def.startSlot = nativeStart;
      nativeSeg.def.endSlot   = nativeEnd;
      bool capturable = true;
      for (int s = nativeStart; s <= nativeEnd && capturable; s++) {
        if (!slots_[s].isCapturable()) capturable = false;
      }
      nativeSeg.def.isCapturable = capturable;
      nativeSeg.def.isFunctionalReplayEligible = capturable && !hasControlFlow_;
      nativeSeg.exec.executionCount = 1;
    }

    FunctionalReplayHandle* functionalHandle = nullptr;
    if (nativeSeg.def.isCapturable && !hasControlFlow_) {
      functionalHandle =
          dynamic_cast<FunctionalReplayHandle*>(nativeSeg.exec.replayHandle.get());
      if (functionalHandle == nullptr) {
        nativeSeg.exec.replayHandle = GraphReplayFactory::createFunctional();
        functionalHandle =
            dynamic_cast<FunctionalReplayHandle*>(nativeSeg.exec.replayHandle.get());
      }
      if (functionalHandle == nullptr) {
        DSP_DIAG(EXECUTE,
                 "%s NativeSlotExecutor: functional recorder unavailable for range [%d-%d]",
                 backendName, nativeStart, nativeEnd);
        nativeSeg.exec.replayHandle.reset();
        return Status::BAD_GRAPH;
      }
      if (!functionalHandle->hasReplayProgram()) {
        if (functionalHandle->getState() != ReplayState::EMPTY) {
          functionalHandle->abortCapture();
        }
        if (!functionalHandle->beginCapture(nullptr)) {
          DSP_DIAG(EXECUTE,
                   "%s NativeSlotExecutor: failed to begin functional capture for range [%d-%d]",
                   backendName, nativeStart, nativeEnd);
          nativeSeg.exec.replayHandle.reset();
          return Status::BAD_GRAPH;
        }
        DSP_DIAG(EXECUTE,
                 "%s NativeSlotExecutor: began functional capture for range [%d-%d]",
                 backendName, nativeStart, nativeEnd);
      }
    }

    Status status = Status::OK;
    try {
      nativeExecutionStarted = true;
      status = executeSegmentSlotBySlot(
          nativeSeg, externalArrays, numExt, stream);
    } catch (...) {
      if (functionalHandle != nullptr &&
          functionalHandle->getState() == ReplayState::CAPTURING) {
        functionalHandle->abortCapture();
      }
      nativeSeg.exec.replayHandle.reset();
      throw;
    }

    if (status != Status::OK) {
      if (functionalHandle != nullptr &&
          functionalHandle->getState() == ReplayState::CAPTURING) {
        functionalHandle->abortCapture();
      }
      nativeSeg.exec.replayHandle.reset();
      return status;
    }

    if (functionalHandle != nullptr &&
        functionalHandle->getState() == ReplayState::CAPTURING) {
      if (!functionalHandle->endCapture(nullptr) ||
          !functionalHandle->finalize()) {
        functionalHandle->abortCapture();
        nativeSeg.exec.replayHandle.reset();
        DSP_DIAG(EXECUTE,
                 "%s NativeSlotExecutor: failed to finalize functional capture for range [%d-%d]",
                 backendName, nativeStart, nativeEnd);
        return Status::BAD_GRAPH;
      }
      DSP_DIAG(EXECUTE,
               "%s NativeSlotExecutor: functional capture finalized for range [%d-%d] commands=%d",
               backendName, nativeStart, nativeEnd,
               functionalHandle->getRecordedOpCount());
    }
    return status;
  };

  // Mixed lowered/native execution is a GraphBackend capability. The plan
  // offers the callback through the common interface and never inspects the
  // concrete implementation.
  backend->setNativeSlotExecutor(nativeSlotCallback);
#endif

  auto status = backend->executeSegment(
      request, seg, slots_, externalArrays, numExt, outputSlots_,
      totalOutputSlots_, stream);
  // GraphBackend contract: BAD_GRAPH is a pre-execution rejection. Any status
  // after work starts must be reported as a non-BAD_GRAPH failure so the
  // resolver never retries after partial execution.
  const bool executionStarted = nativeExecutionStarted || status != Status::BAD_GRAPH;

#if !defined(SD_VULKAN)
  backend->clearNativeSlotExecutor();
#endif

  DSP_DIAG(EXECUTE, "POST-EXECUTE: seg[%d-%d] backend=%s status=%s (%d)",
           seg.def.startSlot, seg.def.endSlot, backendName,
           statusName_seg(status), static_cast<int>(status));

  DSP_DIAG(EXECUTE, "executeSegmentWithSpecificBackend: exec%d seg[%d-%d]: backend=%s status=%s (%d)",
            seg.exec.executionCount, seg.def.startSlot, seg.def.endSlot, backendName,
            statusName_seg(status), static_cast<int>(status));

  if (status != Status::OK && status != Status::BAD_GRAPH) {
    recordFailureDetailIfMissing(
        status,
        std::string("graph backend ") + backendName + " returned " +
            statusName_seg(status) + " (" +
            std::to_string(static_cast<int>(status)) + ") after starting segment [" +
            std::to_string(seg.def.startSlot) + "-" +
            std::to_string(seg.def.endSlot) + "] in phase " +
            seg.exec.displayPhaseName() +
            "; the backend did not publish a lower-level failure detail");
  }

  if (status == Status::OK) {
    // Publish backend identity on every successful path, including a sticky
    // backend selected by the outer cascade. A direct graph backend's compiled
    // artifact is its steady-state executable; it reaches SEALED without being
    // mislabeled as, or wrapped in, a platform replay handle.
    seg.setResolvedGraphBackend(backend, request);
    if (seg.resolvedGraphBackendPolicy.artifactKind ==
            GraphBackendArtifactKind::DIRECT_COMPILED &&
        seg.compiledGraphBackendArtifactOwner == backend &&
        seg.compiledGraphBackendArtifactShapeKey == segShapeKey &&
        seg.compiledGraphBackendArtifact && !seg.exec.segPhase.isSealed()) {
      SegmentLifecycle::markDirectGraphBackendCompiled(
          seg.exec, backendName, segShapeKey,
          seg.def.startSlot, seg.def.endSlot);
    } else if (seg.resolvedGraphBackendPolicy.artifactKind ==
                   GraphBackendArtifactKind::BACKEND_REPLAY_HANDLE &&
               seg.exec.replayHandle && seg.exec.replayHandle->isReady() &&
               !seg.exec.segPhase.isSealed()) {
      SegmentLifecycle::markBackendReplayHandleSealed(
          seg.exec, backendName, segShapeKey,
          seg.def.startSlot, seg.def.endSlot);
    }
    // Cache the shape key only after successful compile+execute so the cascade
    // doesn't skip compilation for the next backend when the current one fails.
    if (!planLifecycle_.isSlotBySlot()) {
      seg.exec.cachedShapeKey = segShapeKey;
    }
    seg.exec.executionCount++;
    if (seg.exec.outcome == SegmentExecOutcome::GRAPH_REPLAY) {
      totalGraphReplays_++;
    }

    // ── Segment boundary validation (warmup only, backend path) ──────────
    if (executeCount_ < 4 &&
        seg.resolvedGraphBackendPolicy.materializesAllFrameworkSlots) {
      char segErr[512] = {};
      int segInvalid = validateSlotRange(
          slots_, numSlots_, seg.def.startSlot, seg.def.endSlot,
          outputSlots_, totalOutputSlots_,
          executeCount_, planLifecycle_.toLegacyCode(),
          segErr, sizeof(segErr));
      if (segInvalid > 0) {
        DSP_THROW(MEMORY,
                 "SEGMENT_BOUNDARY_INVALID: %d invalid array(s) at end of seg[%d-%d] "
                 "(backend=%s, segExecCount=%d): %s",
                 segInvalid, seg.def.startSlot, seg.def.endSlot,
                 backendName, seg.exec.executionCount, segErr);
      }
    } else if (executeCount_ < 4) {
      DSP_DIAG(VERIFY,
               "SEGMENT_BOUNDARY_VALIDATION: backend=%s seg[%d-%d] owns "
               "validation of materialized boundary tensors",
               backendName, seg.def.startSlot, seg.def.endSlot);
    }
    recordSegmentBoundaryOpSanity(
        backendName, seg, slots_, numSlots_, outputSlots_, totalOutputSlots_,
        requestedOutputSlotIndices_, numRequestedOutputs_, diagnosticExecuteCount());
  }

  return DspExecutionResult(status, executionStarted, false);
}

// ─── Native range segment invalidation ───────────────────────────────────────
//
// When an outer segment is invalidated via invalidateForRebuild, any
// nativeRangeSegments_ entries whose slot range falls within that segment must
// also be cleared.  Otherwise the FunctionalReplayHandle captured by the
// NativeSlotExecutor lambda holds executable slot indices recorded against the
// OLD slot array state and will be replayed with stale data on the next token.
//
// This function is called from DspSegmentLifecycle::invalidateForRebuild.

void NativeDynamicShapePlan::clearNativeRangeSegmentsForSlotRange(int startSlot, int endSlot) {
  if (nativeRangeSegments_.empty()) return;
  // Collect keys to erase (can't modify map while iterating it).
  std::vector<uint64_t> toErase;
  for (auto& kv : nativeRangeSegments_) {
    int nStart = static_cast<int>(kv.first >> 32);
    int nEnd   = static_cast<int>(kv.first & 0xFFFFFFFFu);
    // Overlap: native range is inside or spans the outer segment range.
    if (nStart <= endSlot && nEnd >= startSlot) {
      // Release the replay handle before erasing so resources are freed.
      platformCleanupSegmentForRebuild(kv.second);
      toErase.push_back(kv.first);
    }
  }
  for (auto key : toErase) {
    nativeRangeSegments_.erase(key);
  }
  if (!toErase.empty()) {
    DSP_DIAG(EXECUTE,
             "clearNativeRangeSegmentsForSlotRange: cleared %d native range entry(s) "
             "for seg[%d-%d]",
             static_cast<int>(toErase.size()), startSlot, endSlot);
  }
}

// ─── Segment execution: slot-by-slot ─────────────────────────────────────────

// ─── Control flow helpers ────────────────────────────────────────────────────

namespace {

// Resolve an input for a control flow slot
inline NDArray* resolveCfInput(NativeSlot& slot, int inputIdx,
                               NDArray** outputSlots, int totalOutputSlots,
                               NDArray** externalInputs, int numExt) {
  if (inputIdx < 0 || inputIdx >= slot.wiring.numInputs) return nullptr;
  int srcIdx = slot.wiring.inputSourceIndices[inputIdx];
  if (srcIdx >= 0) {
    return (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
  } else {
    int extIdx = -(srcIdx + 1);
    return (extIdx < numExt) ? externalInputs[extIdx] : nullptr;
  }
}

// Check if any input from an output slot is dead.
// In TF-style control flow, if ANY input comes from a dead Switch branch,
// the op is on that dead branch and must be skipped entirely.
// External inputs (srcIdx < 0) don't participate in dead propagation.
inline bool anyInputDead(NativeSlot& slot, bool* slotIsDead, int slotIsDeadSize) {
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    int srcIdx = slot.wiring.inputSourceIndices[i];
    if (srcIdx >= 0 && srcIdx < slotIsDeadSize && slotIsDead[srcIdx]) {
      return true;
    }
  }
  return false;
}

// Mark all outputs of a slot as dead
inline void markOutputsDead(NativeSlot& slot, bool* slotIsDead, int slotIsDeadSize) {
  for (int i = 0; i < slot.wiring.numOutputs; i++) {
    int si = slot.wiring.outputSlotIndices[i];
    if (si >= 0 && si < slotIsDeadSize) slotIsDead[si] = true;
  }
}

// Forward input[0] to all outputs (identity operation for Enter/Exit/LoopCond/NextIteration)
inline void forwardInput(NativeDynamicShapePlan* plan, NativeSlot& slot, NDArray** outputSlots,
                         int totalOutputSlots, NDArray** externalInputs, int numExt,
                         const char* tag) {
  NDArray* input = resolveCfInput(slot, 0, outputSlots, totalOutputSlots, externalInputs, numExt);
  for (int i = 0; i < slot.wiring.numOutputs; i++) {
    int si = slot.wiring.outputSlotIndices[i];
    if (si >= 0 && si < totalOutputSlots) {
      plan->writeOutputSlot(si, input, tag);
    }
  }
}

// Verify helper: log control flow slot output mutations
inline void verifyCfSlotWrite(int stepIdx, const char* cfType, const char* opName,
                               NDArray** outputSlots, int* outputSlotIndices,
                               int numOutputs, int totalOutputSlots) {
  if (!Environment::getInstance().tritonVerifyKernels()) return;
  for (int i = 0; i < numOutputs; i++) {
    int si = outputSlotIndices[i];
    if (si < 0 || si >= totalOutputSlots) continue;
    NDArray* out = outputSlots[si];
    if (out == nullptr) {
      DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=CF_FORWARD cf=%s op=%s (nullptr/dead)", si, cfType, opName);
    } else {
      DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=CF_FORWARD cf=%s op=%s dtype=%s len=%lld addr=%p",
                si, cfType, opName,
                DataTypeUtils::asString(out->dataType()).c_str(),
                (long long)out->lengthOf(), dspBuffer(out));
    }
  }
}

}  // namespace

Status NativeDynamicShapePlan::executeSegmentSlotBySlot(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SLOT_BY_SLOT, "executeSegmentSlotBySlot");
  DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
               "executeSegmentSlotBySlot: ENTER seg[%d-%d] size=%d execCount=%d capturable=%d compilationFailed=%d",
               seg.def.startSlot, seg.def.endSlot, seg.def.endSlot - seg.def.startSlot + 1,
               seg.exec.executionCount, seg.def.isCapturable ? 1 : 0, seg.exec.compilationFailed ? 1 : 0);

  // Corruption scan at segment entry — detect pre-existing corruption from
  // prior segment, writeOutputSlot, or refreshStaleViewWrappers.
  if (sd::Environment::getInstance().isDebug()) {
    char segLabel[256];
    snprintf(segLabel, sizeof(segLabel),
             "SEGMENT_ENTRY_seg%d-%d", seg.def.startSlot, seg.def.endSlot);
    scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
        segLabel, executeCount_);
  }

  // Slot-by-slot execution zeros each slot immediately before that slot runs.
  // Segment-wide prezero is unsafe after freeze-time segment merging: a merged
  // segment can contain hundreds of ops, and zeroing every future output at
  // segment entry can clobber buffers still visible through earlier views.

  // Reset per-segment allocation counters
  tl_dspAllocBytes = 0; tl_dspFreeBytes = 0;
  tl_dspAllocCount = 0; tl_dspFreeCount = 0; tl_dspFreeSkipCount = 0;
  // Resolve the live plan execution stream as a STREAM-VALUE. The `stream` PARAMETER is a
  // STREAM-POINTER (cudaStream_t*); dspStreamIsCapturing and DspThreadState both consume a
  // STREAM-VALUE (see the convention block in DspCudaDispatch.h) — passing the raw pointer
  // to them queries/uses a host address as a stream handle (CUDA 201 / SIGSEGV). Prefer the
  // live value in tl_dspExecutionStream, which platformBeginExecution pins to the plan-owned
  // stream (its heap DspStreamGuard outlives this call). On the FIRST execute the caller
  // stream points at a stale ContextBuffers stream whose context may have been replaced;
  // installing it as tl_dspGapStream made warmup ops routed through getCudaStream()'s
  // gap-stream branch (slot-0 output alloc + setToZeroBuffers) fail with CUDA 201. Fall back
  // to dereferencing the pointer parameter into a value.
  void* segmentStream = dspGetExecutionStream();
  if (segmentStream == nullptr) segmentStream = dspStreamPtrToValue(stream);
  bool streamIsCapturing = dspStreamIsCapturing(segmentStream);
  DspThreadState segmentThreadState(
      segmentStream,
      segmentStream,
      tl_graphExecutionActive,
      tl_dspReplayActive);

#ifdef SD_CUDA
  // BUF_FP_RING external-input fingerprints: async XOR fingerprint of each
  // external's device buffer at segment entry, on the LC stream (the same
  // stream family the H2D prepare and SBS op kernels ride) — records what
  // this exec's ops will actually read. Tracks [0..numExt-1], labels "e<i>".
  if (fpRingEnabled_ && seg.def.startSlot == 0) {
    auto* fpLc = LaunchContext::defaultContext();
    auto* fpStreamPtr = fpLc != nullptr ? fpLc->getCudaStream() : nullptr;
    if (fpStreamPtr != nullptr) {
      auto* fpExecCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
      int fpStep = fpExecCtx != nullptr ? fpExecCtx->fpStep : executeCount_;
      int fpMaxExt = numExt < BUF_FP_MAX_STAGING ? numExt : BUF_FP_MAX_STAGING;
      for (int fpEi = 0; fpEi < fpMaxExt; fpEi++) {
        NDArray* fpExt = externalArrays[fpEi];
        if (fpExt != nullptr && !fpExt->isEmpty() && fpExt->specialBuffer() != nullptr) {
          recordBufFingerprintPublic(*fpStreamPtr, fpStep, fpEi,
                                     fpExt->specialBuffer(), dspSafeByteCount(fpExt));
          if (fpLabels_[fpEi].tag[0] == '\0') {
            snprintf(fpLabels_[fpEi].tag, sizeof(fpLabels_[fpEi].tag), "e%d", fpEi);
            fpLabels_[fpEi].extIdx = fpEi;
            fpLabels_[fpEi].groupIdx = -1;
            fpLabels_[fpEi].whichAB = -1;
          }
        }
      }
    }
  }
#endif

  // Dead-slot flags are reset once per plan execution (in the main execute loop),
  // NOT per segment — dead flags from Switch in seg N must persist to affect
  // ops in seg N+1.

  int stepIdx = seg.def.startSlot;
  int loopIterations = 0;
  bool functionalReplayCompleted = false;

#if !defined(SD_VULKAN)
  auto* functionalHandle =
      dynamic_cast<FunctionalReplayHandle*>(seg.exec.replayHandle.get());
  auto recordFunctionalCommand =
      [&](FunctionalReplayCommandType type, int slotIndex, int argument = -1) -> bool {
    if (functionalHandle == nullptr ||
        functionalHandle->getState() != ReplayState::CAPTURING) {
      return true;
    }
    return functionalHandle->recordCommand(
        type, slots_[slotIndex].ident.op, slotIndex, argument);
  };

  // ── Functional Replay Fast Path ────────────────────────────────────────
  // A finalized program records semantic commands, not just slot numbers.
  // Replay validates the borrowed op identity and resolves current invocation
  // inputs before executing each command.
  if (!planLifecycle_.isSlotBySlot() && executeCount_ >= 2 && !hasControlFlow_ &&
      functionalHandle != nullptr && functionalHandle->hasReplayProgram()) {
    struct FunctionalReplayInvocation {
      NativeDynamicShapePlan* plan;
      GraphSegment* segment;
      NDArray** externalArrays;
      int numExternalArrays;
      void* streamPointer;
      void* streamValue;
      bool captureActive;
    } invocation{
        this, &seg, externalArrays, numExt, stream, segmentStream,
        streamIsCapturing};

    FunctionalReplayExecutionContext replayContext;
    replayContext.userData = &invocation;
    replayContext.execute =
        [](void* userData, const FunctionalReplayCommand& command) -> Status {
      auto* call = static_cast<FunctionalReplayInvocation*>(userData);
      auto* plan = call->plan;
      if (plan == nullptr || call->segment == nullptr) return Status::BAD_INPUT;
      if (command.slotIndex < call->segment->def.startSlot ||
          command.slotIndex > call->segment->def.endSlot ||
          command.slotIndex < 0 || command.slotIndex >= plan->numSlots_) {
        return Status::BAD_GRAPH;
      }

      NativeSlot& replaySlot = plan->slots_[command.slotIndex];
      if (command.op == nullptr || replaySlot.ident.op != command.op) {
        return Status::BAD_GRAPH;
      }

      switch (command.type) {
        case FunctionalReplayCommandType::EXECUTE_SLOT: {
          Status status = plan->executeSlot(
              command.slotIndex, call->externalArrays,
              call->numExternalArrays, call->streamPointer);
          if (status == Status::OK) {
            recordSlotOpSanity(
                "FUNCTIONAL_REPLAY", call->segment->def.startSlot,
                command.slotIndex, plan->slots_, plan->outputSlots_,
                plan->totalOutputSlots_, plan->diagnosticExecuteCount(),
                call->captureActive);
          }
          return status;
        }

        case FunctionalReplayCommandType::FORWARD_IDENTITY: {
          if (!replaySlot.isIdentityOp() ||
              replaySlot.wiring.numInputs != 1 ||
              replaySlot.wiring.numOutputs < 1) {
            return Status::BAD_GRAPH;
          }

          int sourceIndex = replaySlot.wiring.inputSourceIndices[0];
          NDArray* input = nullptr;
          if (sourceIndex >= 0) {
            if (sourceIndex >= plan->totalOutputSlots_) {
              return Status::BAD_GRAPH;
            }
            input = plan->outputSlots_[sourceIndex];
          } else {
            int externalIndex = -(sourceIndex + 1);
            if (externalIndex < 0 ||
                externalIndex >= call->numExternalArrays ||
                call->externalArrays == nullptr) {
              return Status::BAD_INPUT;
            }
            input = call->externalArrays[externalIndex];
          }
          if (input == nullptr) return Status::BAD_INPUT;

          for (int output = 0; output < replaySlot.wiring.numOutputs; output++) {
            int outputIndex = replaySlot.wiring.outputSlotIndices[output];
            if (outputIndex < 0 || outputIndex >= plan->totalOutputSlots_) {
              return Status::BAD_GRAPH;
            }
          }
          for (int output = 0; output < replaySlot.wiring.numOutputs; output++) {
            plan->writeOutputSlot(
                replaySlot.wiring.outputSlotIndices[output],
                input, "functional-replay-identity");
          }
          replaySlot.bumpGeneration();
          recordSlotOpSanity(
              "FUNCTIONAL_REPLAY", call->segment->def.startSlot,
              command.slotIndex, plan->slots_, plan->outputSlots_,
              plan->totalOutputSlots_, plan->diagnosticExecuteCount(),
              call->captureActive);
          return Status::OK;
        }

        case FunctionalReplayCommandType::EXECUTE_BATCHED_GEMM: {
          int groupIndex = command.argument;
          if (groupIndex < 0 ||
              groupIndex >= static_cast<int>(plan->batchedGemmGroups_.size()) ||
              command.slotIndex >=
                  static_cast<int>(plan->slotToBatchedGemmGroup_.size())) {
            return Status::BAD_GRAPH;
          }
          auto& group = plan->batchedGemmGroups_[groupIndex];
          if (group.triggerSlot != command.slotIndex ||
              plan->slotToBatchedGemmGroup_[command.slotIndex] != groupIndex) {
            return Status::BAD_GRAPH;
          }
          Status status = plan->executeBatchedGemmGroup(
              groupIndex, call->externalArrays,
              call->numExternalArrays, call->streamValue);
          if (status == Status::OK) {
            for (int groupSlot : group.slotIndices) {
              recordSlotOpSanity(
                  "FUNCTIONAL_REPLAY_BATCH", call->segment->def.startSlot,
                  groupSlot, plan->slots_, plan->outputSlots_,
                  plan->totalOutputSlots_, plan->diagnosticExecuteCount(),
                  call->captureActive);
            }
          }
          return status;
        }
      }
      return Status::BAD_GRAPH;
    };

    int commandCount = functionalHandle->getProgram().size();
    DSP_DIAG(EXECUTE,
             "FUNCTIONAL_REPLAY: seg[%d-%d] executing %d semantic commands",
             seg.def.startSlot, seg.def.endSlot, commandCount);
    Status replayStatus = functionalHandle->replayWithContext(replayContext);
    if (replayStatus != Status::OK) {
      DSP_DIAG(EXECUTE,
               "FUNCTIONAL_REPLAY: seg[%d-%d] failed status=%s (%d)",
               seg.def.startSlot, seg.def.endSlot,
               statusName_seg(replayStatus),
               static_cast<int>(replayStatus));
      recordFailureDetailIfMissing(
          replayStatus,
          "functional replay returned " +
              std::string(statusName_seg(replayStatus)) + " (" +
              std::to_string(static_cast<int>(replayStatus)) +
              ") for segment [" + std::to_string(seg.def.startSlot) + "-" +
              std::to_string(seg.def.endSlot) +
              "] without command-level failure detail");
      return replayStatus;
    }
    functionalReplayCompleted = true;
  }
#endif

  while (!functionalReplayCompleted && stepIdx <= seg.def.endSlot) {
    NativeSlot& slot = slots_[stepIdx];

    // STEP_ENTER trace: only during warmup to avoid per-slot string formatting
    // overhead (std::to_string heap alloc) in frozen steady-state decode.
    if (executeCount_ < 2) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
                    "STEP_ENTER step=%d op=%s cf=%d numIn=%d numOut=%d outSlots=[%d%s%s] "
                    "identity=%d fusedHead=%d fusedTail=%d fusedLen=%d frozenConst=%d",
                    stepIdx, slot.ident.opName.c_str(),
                    (int)slot.cf.controlFlowType,
                    slot.wiring.numInputs, slot.wiring.numOutputs,
                    slot.wiring.numOutputs > 0 ? slot.wiring.outputSlotIndices[0] : -1,
                    slot.wiring.numOutputs > 1 ? "," : "",
                    slot.wiring.numOutputs > 1 ? std::to_string(slot.wiring.outputSlotIndices[1]).c_str() : "",
                    slot.isIdentityOp() ? 1 : 0,
                    slot.fusedChain.isFusedChainHead ? 1 : 0,
                    slot.fusedChain.isFusedChainTail ? 1 : 0,
                    slot.fusedChain.fusedChainLength,
                    slot.frozenConstantSlot() ? 1 : 0);
    }

    // ── Control flow dispatch ────────────────────────────────────────
    if (slot.cf.controlFlowType != CF_NONE) {
      // Dead propagation: if all inputs are dead and this is not a Merge, propagate dead
      if (slot.cf.controlFlowType != CF_MERGE && hasControlFlow_ && slotIsDead_ != nullptr) {
        if (anyInputDead(slot, slotIsDead_, slotIsDeadSize_)) {
          DSP_DIAG_SLOT(EXECUTE, stepIdx,
                        "slot %d (%s) DEAD: propagated from dead input (cf=%d)",
                        stepIdx, slot.ident.opName.c_str(), (int)slot.cf.controlFlowType);
          markOutputsDead(slot, slotIsDead_, slotIsDeadSize_);
          recordSlotOpSanity(
              "SLOT_BY_SLOT_CF_DEAD", seg.def.startSlot, stepIdx,
              slots_, outputSlots_, totalOutputSlots_, diagnosticExecuteCount(),
              streamIsCapturing, true);
          stepIdx++;
          continue;
        }
      }

      switch (slot.cf.controlFlowType) {
        case CF_SWITCH: {
          // Switch: input[0] = data, input[1] = predicate
          // If predicate is true: output[1] = data, output[0] is dead
          // If predicate is false: output[0] = data, output[1] is dead
          NDArray* data = resolveCfInput(slot, 0, outputSlots_, totalOutputSlots_, externalArrays, numExt);
          NDArray* pred = resolveCfInput(slot, 1, outputSlots_, totalOutputSlots_, externalArrays, numExt);
          bool predValue = false;
          if (pred != nullptr && !pred->isEmpty()) {
            predValue = pred->e<bool>(0);
          }
          int liveIdx = predValue ? 1 : 0;
          int deadIdx = predValue ? 0 : 1;
          for (int i = 0; i < slot.wiring.numOutputs; i++) {
            int si = slot.wiring.outputSlotIndices[i];
            if (si >= 0 && si < totalOutputSlots_) {
              if (i == liveIdx) {
                writeOutputSlot(si, data, "cf-switch-live");
                if (slotIsDead_) slotIsDead_[si] = false;
              } else {
                writeOutputSlot(si, nullptr, "cf-switch-dead");
                if (slotIsDead_) slotIsDead_[si] = true;
              }
            }
          }
          verifyCfSlotWrite(stepIdx, "SWITCH", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
          break;
        }

        case CF_MERGE: {
          // Merge: select first non-dead, non-null input
          NDArray* selected = nullptr;
          for (int i = 0; i < slot.wiring.numInputs; i++) {
            int srcIdx = slot.wiring.inputSourceIndices[i];
            bool isDead = (srcIdx >= 0 && srcIdx < slotIsDeadSize_ && slotIsDead_ && slotIsDead_[srcIdx]);
            if (!isDead) {
              NDArray* inp = resolveCfInput(slot, i, outputSlots_, totalOutputSlots_, externalArrays, numExt);
              if (inp != nullptr) {
                selected = inp;
                break;
              }
            }
          }
          for (int i = 0; i < slot.wiring.numOutputs; i++) {
            int si = slot.wiring.outputSlotIndices[i];
            if (si >= 0 && si < totalOutputSlots_) {
              writeOutputSlot(si, selected, "cf-merge");
              if (slotIsDead_) slotIsDead_[si] = (selected == nullptr);
            }
          }
          verifyCfSlotWrite(stepIdx, "MERGE", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
          break;
        }

        case CF_ENTER:
        case CF_EXIT:
        case CF_LOOP_COND:
          // Identity: forward input[0] to output[0]
          forwardInput(this, slot, outputSlots_, totalOutputSlots_, externalArrays, numExt,
                       slot.cf.controlFlowType == CF_ENTER ? "cf-enter"
                       : slot.cf.controlFlowType == CF_EXIT ? "cf-exit"
                       : "cf-loop-cond");
          {
            const char* cfName = (slot.cf.controlFlowType == CF_ENTER) ? "ENTER" :
                                  (slot.cf.controlFlowType == CF_EXIT) ? "EXIT" : "LOOP_COND";
            verifyCfSlotWrite(stepIdx, cfName, slot.ident.opName.c_str(),
                              outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
          }
          break;

        case CF_NEXT_ITERATION: {
          // Forward input[0] to output[0]
          forwardInput(this, slot, outputSlots_, totalOutputSlots_, externalArrays, numExt,
                       "cf-next-iter");
          verifyCfSlotWrite(stepIdx, "NEXT_ITER", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);

          // Loop-back is handled at the phaseReplay level (across segments),
          // not here, because NextIteration and its target Merge are typically
          // in different segments. Signal phaseReplay by recording the target.
          if (slot.cf.loopBackTarget >= 0) {
            if (cfLoopBackStep_ < 0 || slot.cf.loopBackTarget < cfLoopBackStep_) {
              cfLoopBackStep_ = slot.cf.loopBackTarget;
            }
          }
          break;
        }

        default:
          break;
      }

      // Release schedule removed: arrays persist (one array per slot, never nullified)

      recordSlotOpSanity(
          "SLOT_BY_SLOT_CF", seg.def.startSlot, stepIdx,
          slots_, outputSlots_, totalOutputSlots_, diagnosticExecuteCount(),
          streamIsCapturing);

      stepIdx++;
      continue;
    }

    // ── Dead propagation for regular ops in CF graphs ────────────────
    if (hasControlFlow_ && slotIsDead_ != nullptr) {
      if (anyInputDead(slot, slotIsDead_, slotIsDeadSize_)) {
        markOutputsDead(slot, slotIsDead_, slotIsDeadSize_);
        recordSlotOpSanity(
            "SLOT_BY_SLOT_DEAD", seg.def.startSlot, stepIdx,
            slots_, outputSlots_, totalOutputSlots_, diagnosticExecuteCount(),
            streamIsCapturing, true);
        stepIdx++;
        continue;
      }
    }

    // ── Batched GEMM dispatch ─────────────────────────────────────────
    // Strategy: the FIRST member in each group is the trigger.
    // When reached, it executes the entire batch and populates outputs for
    // ALL members. Non-first members are skipped (output already computed).
    // This ensures downstream ops between members see valid outputs.
    // batchedGemmGroups_ is always empty on CPU builds — the inner body never runs.
    if (sd::graph::dspIsCudaBuild() &&
        !batchedGemmGroups_.empty() && stepIdx < (int)slotToBatchedGemmGroup_.size()) {
      int bgIdx = slotToBatchedGemmGroup_[stepIdx];
      if (bgIdx >= 0 && bgIdx < (int)batchedGemmGroups_.size()) {
        auto& bgGroup = batchedGemmGroups_[bgIdx];
        if (stepIdx == bgGroup.triggerSlot) {
          // This is the trigger (FIRST slot in group) — execute entire batch.
          // All members' inputs are guaranteed available (checked at detection time).
          // executeBatchedGemmGroup reinterpret_casts its void* param to cudaStream_t,
          // so it needs the stream VALUE — pass segmentStream (the live segment stream),
          // NOT the raw pointer param `stream` (which yields a garbage cuBLAS stream → EXECUTION_FAILED).
          Status batchStatus = executeBatchedGemmGroup(bgIdx, externalArrays, numExt, segmentStream);

          if (batchStatus == Status::OK) {
#if !defined(SD_VULKAN)
            if (!recordFunctionalCommand(
                    FunctionalReplayCommandType::EXECUTE_BATCHED_GEMM,
                    stepIdx, bgIdx)) {
              DSP_DIAG(EXECUTE,
                       "FUNCTIONAL_CAPTURE: failed to record batched GEMM group %d at slot %d",
                       bgIdx, stepIdx);
              return Status::BAD_GRAPH;
            }
#endif
            for (int groupSlot : bgGroup.slotIndices) {
              recordSlotOpSanity(
                  "SLOT_BY_SLOT_BATCH", seg.def.startSlot, groupSlot,
                  slots_, outputSlots_, totalOutputSlots_, diagnosticExecuteCount(),
                  streamIsCapturing);
            }
            // Release schedule removed: arrays persist (one array per slot)
            stepIdx++;
            continue;
          }
          // Batched GEMM failure is a hard error — do not silently fall back to individual execution
          DSP_THROW(EXECUTE,
                    "batched GEMM group %d failed with %s (%d) at slot %d (%s). "
                    "Fix the batched GEMM execution — silent fallback to individual execution is not permitted.",
                    bgIdx, statusName_seg(batchStatus), static_cast<int>(batchStatus),
                    stepIdx, slots_[stepIdx].ident.opName.c_str());
        } else {
          // Non-first member: output already computed by the trigger's batch call.
          // Release schedule removed: arrays persist (one array per slot)
          stepIdx++;
          continue;
        }
      }
    }

    // ── Outer-level fast skips (frozen steady-state) ─────────────────
    // When shapes are frozen and past warmup, skip trivially no-op slots
    // at the loop level, avoiding executeSlot() function call overhead.
    // This mirrors the GPU compositeReplay gap loop optimization.
    if (!planLifecycle_.isSlotBySlot() && executeCount_ >= 2) {
      // Frozen constant: output never changes, skip entirely
      if (slot.frozenConstantSlot()) {
        bool allPopulated = true;
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          int si = slot.wiring.outputSlotIndices[o];
          if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] == nullptr) {
            allPopulated = false;
            break;
          }
        }
        if (allPopulated) {
          recordSlotOpSanity(
              "SLOT_BY_SLOT_FROZEN", seg.def.startSlot, stepIdx,
              slots_, outputSlots_, totalOutputSlots_, diagnosticExecuteCount(),
              streamIsCapturing);
          stepIdx++;
          continue;
        }
      }

      // Fused chain tail: head already executed the entire chain
      if (slot.fusedChain.isFusedChainTail) {
        recordSlotOpSanity(
            "SLOT_BY_SLOT_FUSED_TAIL", seg.def.startSlot, stepIdx,
            slots_, outputSlots_, totalOutputSlots_, diagnosticExecuteCount(),
            streamIsCapturing);
        stepIdx++;
        continue;
      }

      // Identity op: just alias input → output
      if (slot.isIdentityOp() && slot.wiring.numInputs == 1 && slot.wiring.numOutputs >= 1) {
        int srcIdx = slot.wiring.inputSourceIndices[0];
        NDArray* input = nullptr;
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          input = outputSlots_[srcIdx];
        } else if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt) input = externalArrays[extIdx];
        }
        if (input != nullptr) {
          for (int o = 0; o < slot.wiring.numOutputs; o++) {
            int si = slot.wiring.outputSlotIndices[o];
            if (si >= 0 && si < totalOutputSlots_) {
              writeOutputSlot(si, input, "outer-identity");
            }
          }
          slot.bumpGeneration();
#if !defined(SD_VULKAN)
          if (!recordFunctionalCommand(
                  FunctionalReplayCommandType::FORWARD_IDENTITY,
                  stepIdx)) {
            DSP_DIAG(EXECUTE,
                     "FUNCTIONAL_CAPTURE: failed to record identity at slot %d",
                     stepIdx);
            return Status::BAD_GRAPH;
          }
#endif
          recordSlotOpSanity(
              "SLOT_BY_SLOT_IDENTITY", seg.def.startSlot, stepIdx,
              slots_, outputSlots_, totalOutputSlots_, diagnosticExecuteCount(),
              streamIsCapturing);
          stepIdx++;
          continue;
        }
      }
    }

    // ── Normal op execution ──────────────────────────────────────────
    Status status;
    bool retriedAfterTrim = false;
    bool shouldRetry = false;
    recordGdrInputSanity(
        "SLOT_BY_SLOT", seg.def.startSlot, stepIdx,
        slots_, outputSlots_, totalOutputSlots_, externalArrays, numExt,
        externalInputNames_, diagnosticExecuteCount(), streamIsCapturing);
    do {
      shouldRetry = false;
      try {
        status = executeSlot(stepIdx, externalArrays, numExt, stream);
      } catch (const std::exception& e) {
        std::string msg = e.what();
        if (!streamIsCapturing &&
            !retriedAfterTrim && (msg.find("cannot allocate") != std::string::npos ||
                                   msg.find("out of memory") != std::string::npos ||
                                   msg.find("Error code: [2]") != std::string::npos)) {
          retriedAfterTrim = true;
          shouldRetry = true;
          DSP_DIAG_SLOT(MEMORY, stepIdx, "slot %d (%s) OOM, trimming pool and retrying...",
                    stepIdx, slots_[stepIdx].ident.opName.c_str());
          dspClearLastCudaError();
          {
            int dev = dspGetCurrentDevice();
            if (dev >= 0) {
              if (dspMemPoolTrim(dev, 0)) {
                DSP_DIAG(MEMORY, "trimmed memory pool on device %d", dev);
              }
            }
          }
          continue;  // retry the slot execution after trimming
        }
        std::string detail = e.what();
        appendSlotInputExceptionContext(detail, slots_[stepIdx],
                                        slots_, numSlots_,
                                        outputSlots_, totalOutputSlots_,
                                        externalArrays, numExt);
        DSP_THROW(EXECUTE, "slot %d (%s) threw exception: %s",
                  stepIdx, slots_[stepIdx].ident.opName.c_str(), detail.c_str());
      } catch (...) {
        DSP_THROW(EXECUTE, "slot %d (%s) threw unknown exception",
                  stepIdx, slots_[stepIdx].ident.opName.c_str());
      }
    } while (shouldRetry);

    // ── DIAG: bisect per-slot output freshness across decode steps ──────
    // In SLOT_BY_SLOT phase every slot re-executes each step. If a slot's
    // output is byte-identical across consecutive decode steps despite fresh
    // external inputs, that slot is the freeze origin. Dump the first output
    // of the first 64 slots (embedding + first layer) so consecutive steps
    // can be diffed. EXECUTE-gated so it costs nothing when disabled.
    if (status == Status::OK && stepIdx < 64 && planLifecycle_.isSlotBySlot()
        && DSP_DIAG_ENABLED(EXECUTE)) {
      NativeSlot& dslot = slots_[stepIdx];
      if (dslot.wiring.numOutputs >= 1) {
        int si = dslot.wiring.outputSlotIndices[0];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr
            && outputSlots_[si]->specialBuffer() != nullptr) {
          DSP_DIAG(EXECUTE, "SLOTOUT_BISECT slot=%d op=%s len=%lld devvals=%s",
                   stepIdx, dslot.ident.opName.c_str(),
                   (long long)outputSlots_[si]->lengthOf(),
                   dspDumpSlotValues(outputSlots_[si]->specialBuffer(),
                                     outputSlots_[si]->dataType(),
                                     outputSlots_[si]->lengthOf(), 4).c_str());
        }
      }
    }

    // ── Post-slot output validation ─────────────────────────────────────
    // After each slot executes, verify its outputs are valid before the next
    // slot reads them as inputs. Catches the exact slot that produces an
    // invalid array, rather than discovering it N slots later.
    // Skip in frozen steady-state (executeCount >= 3): all slot outputs have
    // been validated on prior executions. This eliminates ~1883 validateSlotOutputs
    // calls per decode step.
    if (status == Status::OK && executeCount_ < 3) {
      auto& doneSlot = slots_[stepIdx];
      char postSlotErr[512] = {};
      int badOutputs = validateSlotOutputs(
          stepIdx, doneSlot.ident.opName.c_str(),
          outputSlots_, totalOutputSlots_,
          doneSlot.wiring.outputSlotIndices, doneSlot.wiring.numOutputs,
          executeCount_, planLifecycle_.toLegacyCode(),
          postSlotErr, sizeof(postSlotErr));
      if (badOutputs > 0) {
        DSP_THROW(MEMORY,
                 "SLOT_OUTPUT_INVALID: %d invalid output(s) detected AFTER slot %d (%s) "
                 "execution: %s",
                 badOutputs, stepIdx, doneSlot.ident.opName.c_str(), postSlotErr);
      }
    }

    if (status == Status::OK && opSanityEnabled(diagnosticExecuteCount())) {
      recordSlotOpSanity(
          "SLOT_BY_SLOT", seg.def.startSlot, stepIdx,
          slots_, outputSlots_, totalOutputSlots_, diagnosticExecuteCount(),
          streamIsCapturing);
    }

    // ── Diagnostic: per-slot CUDA error check on warmup execution ──────────
    // Synchronize the active execution stream before inspecting the error. A
    // non-blocking peek only reports launch/setup failures; deferred illegal
    // accesses otherwise surface at segment cleanup, several slots later.
    if (DSP_DIAG_ENABLED(EXECUTE) && status == Status::OK && seg.exec.executionCount == 0
        && !streamIsCapturing && dspGetGraphCaptureStream() == nullptr) {
      dspClearLastCudaError();
      dspSyncDefaultStream();
      int launchErr = dspPeekLastCudaError();
      if (launchErr != 0) {
        // Log all inputs to the failing slot
        auto& faultSlot = slots_[stepIdx];
        for (int i = 0; i < faultSlot.wiring.numInputs; i++) {
          int srcIdx = faultSlot.wiring.inputSourceIndices[i];
          NDArray* inp = nullptr;
          if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
            inp = outputSlots_[srcIdx];
          } else if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx >= 0 && extIdx < numExt) inp = externalArrays[extIdx];
          }
          if (inp != nullptr && inp->dataBuffer() != nullptr) {
            DSP_DIAG(EXECUTE,
                "  FAULT INPUT[%d] srcIdx=%d: shape=%s special=%p primary=%p "
                "len=%lld db=%p closed=%d devId=%d",
                i, srcIdx, ShapeUtils::shapeAsString(inp).c_str(),
                inp->dataBuffer()->special(), inp->dataBuffer()->primary(),
                (long long)inp->lengthOf(), (void*)inp->dataBuffer(),
                inp->dataBuffer()->isClosed() ? 1 : 0,
                inp->dataBuffer()->deviceId());
          } else {
            DSP_DIAG(EXECUTE, "  FAULT INPUT[%d] srcIdx=%d: %s",
                     i, srcIdx, inp ? "db=null" : "null");
          }
        }
        // Log outputs of the failing slot
        for (int i = 0; i < faultSlot.wiring.numOutputs; i++) {
          int si = faultSlot.wiring.outputSlotIndices[i];
          NDArray* out = (si >= 0 && si < totalOutputSlots_) ? outputSlots_[si] : nullptr;
          if (out != nullptr && out->dataBuffer() != nullptr) {
            DSP_DIAG(EXECUTE,
                "  FAULT OUTPUT[%d] slotIdx=%d: shape=%s special=%p len=%lld",
                i, si, ShapeUtils::shapeAsString(out).c_str(),
                out->dataBuffer()->special(), (long long)out->lengthOf());
          } else {
            DSP_DIAG(EXECUTE, "  FAULT OUTPUT[%d] slotIdx=%d: %s",
                     i, si, out ? "db=null" : "null");
          }
        }
        DSP_THROW_CUDA(EXECUTE, static_cast<cudaError_t>(launchErr),
                       "CUDA LAUNCH DIAGNOSTIC: dspPeekLastCudaError after slot %d (%s) "
                       "returned error %d. "
                       "seg=[%d-%d] execCount=%d shapesFrozen=%d",
                       stepIdx, slots_[stepIdx].ident.opName.c_str(),
                       static_cast<int>(launchErr),
                       seg.def.startSlot, seg.def.endSlot, executeCount_, static_cast<int>(planLifecycle_.isShapesFrozen()));
      }
    }

    if (status != Status::OK) {
      // Record the failure in the execution flow log FIRST so it's
      // available even if the exception below is caught by a caller.
      auto* execCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
      if (execCtx != nullptr) {
        execCtx->recordFlow(PlanExecutionContext::FlowEventType::SLOT_EXEC_FAIL,
                             stepIdx, static_cast<int>(status));
        // Dump entire flow log on failure so we can see exactly what
        // happened this execution (auto-seal, frozen constants, etc.)
        execCtx->dumpFlowLog(executeCount_);
      }

      char buf[1024];
      auto* errorReference =
          sd::LaunchContext::defaultContext()->errorReference();
      const char* existingMsg = errorReference->errorMessage();
      if (existingMsg != nullptr && existingMsg[0] != '\0') {
        // Keep the originating exception first so the fixed ErrorReference buffer
        // cannot truncate it behind repeated status-wrapper prefixes.
        snprintf(buf, sizeof(buf), "%s [slot %d (%s) returned %s (%d)]",
                 existingMsg, stepIdx, slots_[stepIdx].ident.opName.c_str(),
                 statusName_seg(status), static_cast<int>(status));
      } else {
        snprintf(buf, sizeof(buf),
                 "slot %d (%s) returned %s (%d) without native failure detail; "
                 "the op/platform implementation returned an error status without "
                 "setting LaunchContext::errorReference",
                 stepIdx, slots_[stepIdx].ident.opName.c_str(),
                 statusName_seg(status), static_cast<int>(status));
      }
      errorReference->setErrorCode(static_cast<int>(status));
      errorReference->setErrorMessage(buf);

      // Log full input details for the failing slot
      auto& failedSlot = slots_[stepIdx];
      for (int i = 0; i < failedSlot.wiring.numInputs; i++) {
        int srcIdx = failedSlot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0) {
          NDArray* inp = (srcIdx < totalOutputSlots_ ? outputSlots_[srcIdx] : nullptr);
          if (inp != nullptr) {
            // Protect rankOf() call — if shapeInfo is null, rankOf() would throw
            // and propagate out of this catch handler, causing cascading failures.
            try {
              // Check if the source slot is frozen — if so, that's critical diagnostic info
              bool srcIsFrozen = false;
              for (int fs = 0; fs < numSlots_; fs++) {
                for (int fo = 0; fo < slots_[fs].wiring.numOutputs; fo++) {
                  if (slots_[fs].wiring.outputSlotIndices[fo] == srcIdx) {
                    srcIsFrozen = slots_[fs].frozenConstantSlot();
                    break;
                  }
                }
                if (srcIsFrozen) break;
              }
              DSP_DIAG(EXECUTE, "  input[%d] from outputSlot[%d]: rank=%lld shape=%s "
                        "shapeInfo=%p db=%p frozenSrc=%d",
                        i, srcIdx, (long long)inp->rankOf(),
                        ShapeUtils::shapeAsString(inp).c_str(),
                        (void*)inp->shapeInfo(), (void*)inp->dataBuffer(),
                        srcIsFrozen ? 1 : 0);
            } catch (...) {
              DSP_DIAG(EXECUTE, "  input[%d] from outputSlot[%d]: ptr=%p (shapeInfo INVALID)",
                        i, srcIdx, (void*)inp);
            }
          } else {
            DSP_DIAG(EXECUTE, "  input[%d] from outputSlot[%d]: null", i, srcIdx);
          }
        } else {
          DSP_DIAG(EXECUTE, "  input[%d] from external[%d]", i, -(srcIdx + 1));
        }
      }

      // Log output slot info for the failing slot
      for (int o = 0; o < failedSlot.wiring.numOutputs; o++) {
        int outIdx = failedSlot.wiring.outputSlotIndices[o];
        NDArray* out = (outIdx >= 0 && outIdx < totalOutputSlots_) ? outputSlots_[outIdx] : nullptr;
        if (out != nullptr) {
          try {
            DSP_DIAG(EXECUTE, "  output[%d] outputSlot[%d]: rank=%lld shape=%s db=%p",
                      o, outIdx, (long long)out->rankOf(),
                      ShapeUtils::shapeAsString(out).c_str(),
                      (void*)out->dataBuffer());
          } catch (...) {
            DSP_DIAG(EXECUTE, "  output[%d] outputSlot[%d]: ptr=%p (shapeInfo INVALID)",
                      o, outIdx, (void*)out);
          }
        } else {
          DSP_DIAG(EXECUTE, "  output[%d] outputSlot[%d]: %s", o, outIdx,
                    outIdx >= 0 ? "null" : "invalid-idx");
        }
      }

      dspClearLastCudaError();
      DSP_THROW(EXECUTE, "%s", buf);
    }

    // Classify ownership for all outputs produced by this slot.
    // Skip in frozen steady-state: ownership is stable after warmup.
    if (slotOwnership_ != nullptr && executeCount_ < 2) {
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        int si = slot.wiring.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
          classifyAndUpdateOwnership(
              slotOwnership_[si], outputSlots_[si], si,
              externalArrays, numExt,
              outputSlots_, totalOutputSlots_,
              slotOwnership_);
        }
      }
    }

#if !defined(SD_VULKAN)
    // Functional replay is a host backend. Vulkan records complete hardware
    // segments through VulkanSegmentRecorder and never enters this slot path.
    bool skipFunctionalRecord =
        slot.frozenConstantSlot() || slot.fusedChain.isFusedChainTail;
    if (!skipFunctionalRecord &&
        !recordFunctionalCommand(
            FunctionalReplayCommandType::EXECUTE_SLOT, stepIdx)) {
      DSP_DIAG(EXECUTE,
               "FUNCTIONAL_CAPTURE: failed to record executable slot %d (%s)",
               stepIdx, slot.ident.opName.c_str());
      return Status::BAD_GRAPH;
    }
#endif

    // Release schedule removed: arrays persist (one array per slot, never nullified).
    // Same plan = same shapes. Arrays allocated on first execution, reused forever.

    stepIdx++;
  }

  if (!viewProducerDetectionDone_) {
    viewProducerDetectionDone_ = true;
    int viewCount = 0;
    // slots_ has exactly numSlots_ elements (NativePlanCompiler: new NativeSlot[numSteps]).
    // totalOutputSlots_ counts output-slot INDEX entries and is >= numSlots_ whenever any op
    // has 2+ outputs, so using it here read slots_[numSlots_..totalOutputSlots_-1] OUT OF
    // BOUNDS — the SLOT_BY_SLOT SIGSEGV (SEGV_ACCERR) on the 2419-slot prefill. isViewProducer
    // is a per-SLOT (per-op) property, so the scan must be bounded by numSlots_.
    for (int i = 0; i < numSlots_; i++) {
      if (slots_[i].slotPhase.isViewProducer) viewCount++;
    }
    DSP_DIAG(SHAPE, "view producer detection done: %d/%d slots are view producers",
              viewCount, numSlots_);
  }

  // Log per-segment allocation/free summary
  DSP_DIAG(MEMORY, "SEG-MEM: seg[%d-%d] exec=%d alloc=%lldMB(%d) free=%lldMB(%d) freeSkip=%d net=%lldMB",
           seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
           tl_dspAllocBytes / (1024*1024), tl_dspAllocCount,
           tl_dspFreeBytes / (1024*1024), tl_dspFreeCount, tl_dspFreeSkipCount,
           (tl_dspAllocBytes - tl_dspFreeBytes) / (1024*1024));

  seg.exec.executionCount++;

  // ── Segment boundary validation (warmup only) ──────────────────────────
  // After a slot-by-slot segment completes, verify all outputs in the segment
  // range are valid before downstream segments read them as inputs.
  // Gated to first 4 executions only — in frozen steady-state, shapes and
  // buffers are stable and this O(totalSlots) scan is pure overhead.
  if (executeCount_ < 4) {
    char segErr[512] = {};
    int segInvalid = validateSlotRange(
        slots_, numSlots_, seg.def.startSlot, seg.def.endSlot,
        outputSlots_, totalOutputSlots_,
        executeCount_, planLifecycle_.toLegacyCode(),
        segErr, sizeof(segErr));
    if (segInvalid > 0) {
      DSP_THROW(MEMORY,
               "SEGMENT_BOUNDARY_INVALID: %d invalid array(s) at end of seg[%d-%d] "
               "(slot-by-slot, segExecCount=%d): %s",
               segInvalid, seg.def.startSlot, seg.def.endSlot,
               seg.exec.executionCount, segErr);
    }
  }

  // ── Pin externally-owned buffers this segment re-reads on later frozen re-execs ─────────
  // A NOT_FUSIBLE / slot-by-slot segment never reaches the CUDA-graph seal sites, so its
  // cached device addresses were previously UNPROTECTED: a frozen segment caches a VIEW over a
  // weight (e.g. a reshape-over-constant — outputSlots_[slot] shares the weight's device buffer)
  // and re-reads it every exec, but a user close()/rebind of that weight freed the buffer →
  // err700 illegal access on the next re-exec. Pin views + SOURCE_VARIABLE weights here once
  // shapes are frozen (addresses stable); pinOwnedOutputs=false leaves recomputed intermediates
  // freeable. Idempotent + frozen-gated, so steady-state cost is a bounded dedup scan.
  if (planLifecycle_.isShapesFrozen()) {
    pinSegmentGraphBakedSlots(seg, externalArrays, numExt, /*pinOwnedOutputs=*/false);
  }

  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Emulated Graph Replay
// ═══════════════════════════════════════════════════════════════════════════════
//
// Executes ops slot-by-slot but emulates the full graph replay lifecycle:
//   executionCount == 0 (WARMUP): Record baseline shape key + address snapshot
//   executionCount == 1 (EMULATED_CAPTURE): Verify shape/address stability
//   executionCount >= 2 (EMULATED_REPLAY): Steady-state with stability tracking
//
// Emits DSP_DIAG_EMULATED_REPLAY diagnostics at every phase, reporting what a
// real CUDA graph replay backend would see. This lets users diagnose graph
// replay failures without needing actual CUDA graph capture.
// ═══════════════════════════════════════════════════════════════════════════════

#if !defined(SD_VULKAN)
namespace {

FunctionalReplayPointerBinding snapshotFunctionalReplayPointer(
    FunctionalReplayPointerRole role, int index, int sourceType,
    bool requiredAtEntry, NDArray* array) {
  FunctionalReplayPointerBinding binding;
  binding.role = role;
  binding.index = index;
  binding.sourceType = sourceType;
  binding.requiredAtEntry = requiredAtEntry;
  binding.array = array;
  if (array == nullptr) return binding;

  binding.shapeInfo = array->shapeInfo();
  if (binding.shapeInfo == nullptr) return binding;

  binding.offset = array->offset();
  binding.length = array->lengthOf();
  binding.dataType = static_cast<int>(array->dataType());
  binding.empty = array->isEmpty();

  auto* dataBuffer = array->dataBuffer();
  binding.dataBuffer = dataBuffer;
  if (dataBuffer == nullptr) {
    binding.live = binding.empty;
    return binding;
  }
  if (!dataBuffer->isValid()) return binding;

  // Use raw DataBuffer identities. NDArray::specialBuffer() can allocate or
  // migrate; pointer observation must never change residency.
  binding.primaryBuffer = dataBuffer->primary();
  binding.specialBuffer = dataBuffer->special();
  binding.live = binding.empty || binding.primaryBuffer != nullptr ||
                 binding.specialBuffer != nullptr;
  return binding;
}

Status buildFunctionalReplayPointerSnapshot(
    const GraphSegment& seg, NativeSlot* slots, int numSlots,
    NDArray** externalArrays, int numExt,
    NDArray** outputSlots, int totalOutputSlots,
    std::vector<FunctionalReplayPointerBinding>* bindings) {
  if (bindings == nullptr || slots == nullptr || numSlots < 0 ||
      numExt < 0 || totalOutputSlots < 0) {
    return Status::BAD_INPUT;
  }

  bindings->clear();
  std::vector<int> externalIndices;
  std::vector<int> externalTypes(static_cast<size_t>(numExt), -1);
  std::vector<int> producedOutputIndices;
  std::vector<int> crossSegmentIndices;

  for (int slotIndex = seg.def.startSlot;
       slotIndex <= seg.def.endSlot; slotIndex++) {
    if (slotIndex < 0 || slotIndex >= numSlots) return Status::BAD_GRAPH;
    const NativeSlot& slot = slots[slotIndex];

    for (int output = 0; output < slot.wiring.numOutputs; output++) {
      int outputIndex = slot.wiring.outputSlotIndices[output];
      if (outputIndex < 0 || outputIndex >= totalOutputSlots) {
        return Status::BAD_GRAPH;
      }
      producedOutputIndices.push_back(outputIndex);
    }
  }

  std::sort(producedOutputIndices.begin(), producedOutputIndices.end());
  producedOutputIndices.erase(
      std::unique(producedOutputIndices.begin(), producedOutputIndices.end()),
      producedOutputIndices.end());

  for (int slotIndex = seg.def.startSlot;
       slotIndex <= seg.def.endSlot; slotIndex++) {
    const NativeSlot& slot = slots[slotIndex];
    for (int input = 0; input < slot.wiring.numInputs; input++) {
      int sourceIndex = slot.wiring.inputSourceIndices[input];
      if (sourceIndex < 0) {
        int externalIndex = -(sourceIndex + 1);
        if (externalIndex < 0 || externalIndex >= numExt ||
            externalArrays == nullptr ||
            slot.wiring.inputSourceTypes == nullptr) {
          return Status::BAD_INPUT;
        }
        int sourceType = static_cast<int>(
            slot.wiring.inputSourceTypes[input]);
        if (externalTypes[externalIndex] >= 0 &&
            externalTypes[externalIndex] != sourceType) {
          return Status::BAD_GRAPH;
        }
        if (externalTypes[externalIndex] < 0) {
          externalTypes[externalIndex] = sourceType;
          externalIndices.push_back(externalIndex);
        }
        continue;
      }

      if (sourceIndex >= totalOutputSlots) return Status::BAD_GRAPH;
      if (!std::binary_search(producedOutputIndices.begin(),
                              producedOutputIndices.end(), sourceIndex)) {
        crossSegmentIndices.push_back(sourceIndex);
      }
    }
  }

  std::sort(externalIndices.begin(), externalIndices.end());
  std::sort(crossSegmentIndices.begin(), crossSegmentIndices.end());
  crossSegmentIndices.erase(
      std::unique(crossSegmentIndices.begin(), crossSegmentIndices.end()),
      crossSegmentIndices.end());

  bindings->reserve(externalIndices.size() + crossSegmentIndices.size() +
                    producedOutputIndices.size());
  for (int externalIndex : externalIndices) {
    bindings->push_back(snapshotFunctionalReplayPointer(
        FunctionalReplayPointerRole::EXTERNAL_INPUT, externalIndex,
        externalTypes[externalIndex], true,
        externalArrays[externalIndex]));
  }
  for (int outputIndex : crossSegmentIndices) {
    NDArray* array = outputSlots == nullptr ? nullptr : outputSlots[outputIndex];
    bindings->push_back(snapshotFunctionalReplayPointer(
        FunctionalReplayPointerRole::CROSS_SEGMENT_INPUT, outputIndex,
        SOURCE_OP_OUTPUT, true, array));
  }
  for (int outputIndex : producedOutputIndices) {
    NDArray* array = outputSlots == nullptr ? nullptr : outputSlots[outputIndex];
    bindings->push_back(snapshotFunctionalReplayPointer(
        FunctionalReplayPointerRole::SEGMENT_OUTPUT, outputIndex,
        SOURCE_OP_OUTPUT, false, array));
  }
  return Status::OK;
}

}  // namespace
#endif

LongType NativeDynamicShapePlan::computeSegmentInputAddrKeyPortable(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  // FNV-1a hash of buffer addresses for all CROSS-SEGMENT inputs plus all
  // non-PLACEHOLDER external inputs (weights / constants). On CUDA, uses
  // specialBuffer(); on CPU, uses primaryBuffer(). Address changes between
  // executions indicate the graph would have stale pointers.
  //
  // PLACEHOLDER externals (position_ids, attention_mask, …) are excluded —
  // Java allocates fresh arrays for them every decode step, so their
  // pointers always change; hashing them would pin the plan below
  // REPLAYING forever. Weights / constants are device-authoritative
  // and supposed to be stable; a user-visible close+associateArrayWithVariable
  // rebind on one of them should invalidate the cached replay rather than
  // silently replay against freed device memory.
  uint64_t hash = dsp::FNV1A64_OFFSET_BASIS;
  auto mix = [&hash](uintptr_t val) {
    dsp::fnv1aMixValue(hash, static_cast<uint64_t>(val));
  };

  // externalInputIsVariable_ is populated at plan-load time, not at
  // shape-freeze time — in explicit graphExecutionMode (CUDA_GRAPHS,
  // TRITON, NVRTC_JIT, PTX_JIT) the plan never goes through AUTO_SEAL so
  // planLifecycle_.isShapesFrozen() stays false even once graph replay is active. Gate on
  // the vector being non-empty only.
  const bool canClassifyExternals = !externalInputIsVariable_.empty();

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        if (!canClassifyExternals) continue;
        int extIdx = -(srcIdx + 1);
        if (extIdx < 0 || extIdx >= numExt) continue;
        if (extIdx >= static_cast<int>(externalInputIsVariable_.size())) continue;
        // Skip ALL variable inputs — their addresses churn every step as Java
        // recreates NDArray wrappers, preventing addr key stabilization.
        // Variable data sync is handled by performPreReplaySync + staging buffers.
        if (externalInputIsVariable_[extIdx]) continue;
        NDArray* extArr = externalInputs[extIdx];
        if (extArr == nullptr) continue;
        mix(reinterpret_cast<uintptr_t>(dspBuffer(extArr)));
        continue;
      }
      NDArray* arr = nullptr;
      if (srcIdx < totalOutputSlots_) {
        arr = outputSlots_[srcIdx];
      }
      if (arr != nullptr) {
        // Guard against stale view NDArrays whose DataBuffer has been freed
        // between calls. In EMULATED_REPLAY (and unfrozen slot-by-slot),
        // placeholder arrays are closed by Java after each execution. View
        // chains (permute→reshape→...) that wrap a placeholder's DataBuffer
        // become dangling. Calling specialBuffer() on such an array triggers
        // syncToDevice → migrate() on the freed DataBuffer, reading corrupted
        // _lenInBytes/_deviceId fields → SIGSEGV in Workspace::allocateBytes.
        auto* db = arr->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          mix(0);
        } else {
          mix(reinterpret_cast<uintptr_t>(dspBuffer(arr)));
        }
      } else {
        mix(0);  // cross-segment nullptr sentinel
      }
    }
  }
  return hash;
}

Status NativeDynamicShapePlan::executeSegmentEmulatedReplay(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SLOT_BY_SLOT, "executeSegmentEmulatedReplay");
  if (seg.def.selectedBackend != SelectedBackend::EMULATED_REPLAY) {
    DSP_DIAG(FALLBACK,
             "FUNCTIONAL_POINTER_TRACKER_SCOPE_VIOLATION: seg[%d-%d] backend=%d",
             seg.def.startSlot, seg.def.endSlot,
             static_cast<int>(seg.def.selectedBackend));
    return Status::BAD_GRAPH;
  }

  int segSize = seg.def.endSlot - seg.def.startSlot + 1;
  int execCount = seg.exec.executionCount;

  // ── Phase determination ─────────────────────────────────────────────────
  const char* phaseName;
  if (execCount == 0) {
    DSP_SET_SEG_PHASE(seg, ExecutionPhase::WARMUP, "emulated_replay_exec0");
    phaseName = "WARMUP";
  } else if (execCount == 1) {
    DSP_SET_SEG_PHASE(seg, ExecutionPhase::COMPILING, "emulated_replay_capture");  // "capture" equivalent
    phaseName = "EMULATED_CAPTURE";
  } else {
    DSP_SET_SEG_PHASE(seg, ExecutionPhase::REPLAYING, "emulated_replay_steady");
    phaseName = "EMULATED_REPLAY";
  }

  // phaseWarmup() executes this segment slot-by-slot before the first call into
  // the emulated replay backend and leaves executionCount at 1. Treat that
  // centralized warmup as complete so this call can record the functional
  // replay program instead of remaining in BUILDING:WARMUP indefinitely.
  if (execCount >= 1 && seg.exec.segPhase.needsWarmup()) {
    SegmentLifecycle::skipToCapturing(
        seg.exec, "emulated_replay",
        seg.def.startSlot, seg.def.endSlot);
  }

  DSP_DIAG(EMULATED_REPLAY,
           "EMULATED seg[%d-%d] phase=%s execCount=%d slots=%d capturable=%d frozen=%d",
           seg.def.startSlot, seg.def.endSlot, phaseName, execCount, segSize,
           seg.def.isCapturable ? 1 : 0, planLifecycle_.isShapesFrozen() ? 1 : 0);

  // ── External input dtype validation ────────────────────────────────────
  // Fresh placeholder arrays passed from Java must have valid dtypes (FLOAT32,
  // HALF, etc.). An UNKNOWN dtype on a freshly created NDArray means its
  // shapeInfo extras word lost its dtype flags — typically from a
  // ConstantShapeHelper trie entry being overwritten by a corrected shape
  // from a prior plan execution, or from a freed DataBuffer whose shapeInfo
  // pointer was recycled.  Detect and repair here before the UNKNOWN poisons
  // downstream view chains and causes "Unknown data type requested" throws.
  for (int ei = 0; ei < numExt; ei++) {
    if (externalArrays[ei] == nullptr) continue;
    auto extDt = externalArrays[ei]->dataType();
    if (extDt == DataType::UNKNOWN || extDt == DataType::INHERIT) {
      // Repair: the DataBuffer knows its true type even if shapeInfo is corrupt.
      auto* db = externalArrays[ei]->dataBuffer();
      if (db != nullptr && db->isValid()) {
        DataType dbDt = db->getDataType();
        if (dbDt != DataType::UNKNOWN && dbDt != DataType::INHERIT) {
          const LongType* curShape = externalArrays[ei]->shapeInfo();
          int rank = shape::rank(curShape);
          const LongType* fixedShape = ConstantShapeHelper::getInstance().createShapeInfo(
              dbDt, shape::order(curShape), rank, shape::shapeOf(curShape));
          externalArrays[ei]->setShapeInfo(const_cast<LongType*>(fixedShape));
          DSP_DIAG(SHAPE,
              "EXT_INPUT_DTYPE_REPAIR: ext[%d] had UNKNOWN dtype, repaired to %s "
              "from DataBuffer (execCount=%d phase=%s)",
              ei, DataTypeUtils::asString(dbDt).c_str(), execCount, phaseName);
        } else {
          DSP_THROW(SHAPE,
              "EXT_INPUT_UNRECOVERABLE: ext[%d] has UNKNOWN dtype AND DataBuffer "
              "dtype is also UNKNOWN/INHERIT. Cannot execute. execCount=%d phase=%s "
              "db=%p valid=%d closed=%d",
              ei, execCount, phaseName, (void*)db, db->isValid() ? 1 : 0,
              db->isClosed() ? 1 : 0);
        }
      }
    }
  }

  // ── View wrapper refresh ───────────────────────────────────────────────
  // View-producer slots (permute, reshape, etc.) wrap their input's DataBuffer.
  // When the placeholder is replaced between calls and the old one is closed,
  // the view in outputSlots_ holds a dangling DataBuffer pointer. Refresh
  // stale view wrappers BEFORE any key computation or slot execution to
  // prevent downstream slots from reading UNKNOWN dtype from freed memory.
  //
  // The guard resolves the first produced output slot rather than treating
  // the operation start index as a flat output index. This matters when an
  // earlier operation has multiple outputs and the segment's first output is
  // not numerically equal to its first operation. The segment executionCount
  // can also be reset by platformReleaseSegmentGpuResources while output
  // arrays from a previous execution persist, so populated outputs still need
  // refresh in that case.
  const int firstOutputSlot = dsp::firstSegmentOutputSlot(
      slots_, numSlots_, seg.def.startSlot, seg.def.endSlot,
      totalOutputSlots_);
  bool hasPopulatedSlots = (outputSlots_ != nullptr && firstOutputSlot >= 0 &&
      outputSlots_[firstOutputSlot] != nullptr);
  if (hasPopulatedSlots) {
    // refreshStaleViewWrappersInSegment self-marks args stale (markArgsStale) when
    // it refreshes or demotes any view wrapper — no manual bump+reset needed here.
    refreshStaleViewWrappersInSegment(seg, externalArrays, numExt);
  }

  // ── Gap 1: Fast path — skip key recomputation when stable ──────────────
  // When the generation counter shows no refresh needed (both shape and addr
  // keys matched on previous execution), skip expensive hash computations and
  // go straight to slot-by-slot execution. This eliminates shape key overhead
  // (~5-10us per segment) that real graph replay also avoids.
  bool fastPath = false;
  // Never fast-path when variable external inputs exist: their data may have
  // been mutated in-place (e.g. putScalar during gradient checking) without
  // changing the pointer, so the generation counter won't detect the change.
  bool hasVariableInputs = !cachedVariableExtIndices_.empty();
  if (execCount >= 2 && !seg.exec.needsArgRefresh() && !hasVariableInputs) {
    fastPath = true;
    // Ext-input address-key guard (task #54): the generation counter only
    // advances when NATIVE code observes a change. When the Java fast-path
    // resolver replaces a placeholder (new INDArray + new device buffer),
    // nothing bumps the generation — the fast path then dispatches with
    // capture/staging buffers still referencing the PREVIOUS (freed) device
    // address, and the kernel reads recycled memory. Re-use the canonical
    // staleness detector: recompute the (cheap, pointer-hash) input addr key
    // and demote to the full path on baseline mismatch. lastExternalInputAddrs_
    // is useless here — it is re-recorded at every execute ENTRY, so it always
    // matches the current arrays by construction.
    LongType fastPathAddrKey =
        computeSegmentInputAddrKeyPortable(seg, externalArrays, numExt);
    if (fastPathAddrKey != seg.exec.capturedInputAddrKey) {
      DSP_DIAG(FALLBACK,
               "FAST_PATH_EXT_ADDR_MISMATCH: input addr key 0x%llx != captured "
               "0x%llx with gen %llu unbumped (Java-side placeholder "
               "re-resolution). Demoting to full key path — staging/arg refresh "
               "will run.",
               (long long)fastPathAddrKey,
               (long long)seg.exec.capturedInputAddrKey,
               (unsigned long long)seg.exec.argTableGeneration);
      fastPath = false;
    }
  }
  if (fastPath) {
    DSP_DIAG(EMULATED_REPLAY,
             "  FAST PATH: args current (gen %llu), skipping key recomputation",
             (unsigned long long)seg.exec.argTableGeneration);
  }

  LongType currentShapeKey = 0;
  LongType currentAddrKey = 0;

  if (!fastPath) {
    // ── Compute shape key ──────────────────────────────────────────────────
    currentShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    currentAddrKey = computeSegmentInputAddrKeyPortable(seg, externalArrays, numExt);
  }

  if (execCount == 0) {
    // ══════════════════════════════════════════════════════════════════════
    // WARMUP: baseline keys + fusion analysis + capture buffer sizing + DOT
    // ══════════════════════════════════════════════════════════════════════
    seg.exec.recordReplayBaselineKeys(currentShapeKey, currentAddrKey,
                                      "emulated_replay_warmup");
    seg.exec.markArgsStale();

    DSP_DIAG(EMULATED_REPLAY,
             "  WARMUP baseline: shapeKey=0x%llx addrKey=0x%llx",
             (long long)currentShapeKey, (long long)currentAddrKey);

    // ── Gap 3: Capture buffer sizing (byte-level) ────────────────────────
    int numPlaceholders = 0, numConstants = 0, numVariables = 0;
    size_t placeholderBytes = 0, constantBytes = 0, variableBytes = 0;
    // Track unique external indices to avoid double-counting
    std::unordered_set<int> seenExt;

    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (seenExt.count(extIdx)) continue;
          seenExt.insert(extIdx);

          int8_t srcType = slot.wiring.inputSourceTypes[i];
          size_t bytes = 0;
          if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
            bytes = externalArrays[extIdx]->lengthOf() * externalArrays[extIdx]->sizeOfT();
          }

          if (srcType == SOURCE_PLACEHOLDER) {
            numPlaceholders++;
            placeholderBytes += bytes;
          } else if (srcType == SOURCE_CONSTANT) {
            numConstants++;
            constantBytes += bytes;
          } else if (srcType == SOURCE_VARIABLE) {
            numVariables++;
            variableBytes += bytes;
          }
        }
      }
    }

    DSP_DIAG(EMULATED_REPLAY,
             "  capture buffers: %d placeholders (%zuKB staging needed), "
             "%d constants (%zuKB direct ref), %d variables (%zuKB direct ref if frozen)",
             numPlaceholders, placeholderBytes / 1024,
             numConstants, constantBytes / 1024,
             numVariables, variableBytes / 1024);

    // Per-placeholder detail for large inputs
    if (DSP_DIAG_ENABLED(EMULATED_REPLAY)) {
      seenExt.clear();
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (seenExt.count(extIdx)) continue;
            seenExt.insert(extIdx);
            if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER &&
                extIdx < numExt && externalArrays[extIdx] != nullptr) {
              auto* arr = externalArrays[extIdx];
              size_t bytes = arr->lengthOf() * arr->sizeOfT();
              DSP_DIAG(EMULATED_REPLAY,
                       "    ext[%d] PLACEHOLDER shape=[%s] dtype=%d bytes=%zu",
                       extIdx, ShapeUtils::shapeAsString(arr).c_str(),
                       (int)arr->dataType(), bytes);
            }
          }
        }
      }
    }

    // ── Gap 2: Kernel fusion analysis ────────────────────────────────────
    int numIdentity = 0, numViewOps = 0, numFusedChains = 0, numFusedTails = 0;
    int numInPlaceFused = 0, numDataDependent = 0, numControlFlow = 0;
    int numMatmul = 0, numElementwise = 0, numOther = 0;
    int totalFusedChainOps = 0;

    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      if (slot.isIdentityOp())     numIdentity++;
      if (slot.isViewCapableOp())  numViewOps++;
      if (slot.fusedChain.isFusedChainHead) { numFusedChains++; totalFusedChainOps += slot.fusedChain.fusedChainLength; }
      if (slot.fusedChain.isFusedChainTail) numFusedTails++;
      if (slot.isInPlaceFused())        numInPlaceFused++;
      if (slot.isDataDependent())  numDataDependent++;
      if (slot.cf.controlFlowType != CF_NONE) numControlFlow++;

      // Classify by op name heuristic
      const auto& name = slot.ident.opName;
      if (name.find("matmul") != std::string::npos || name.find("mmul") != std::string::npos ||
          name.find("gemm") != std::string::npos || name.find("batched_gemm") != std::string::npos) {
        numMatmul++;
      } else if (slot.isIdentityOp() || slot.isViewCapableOp() || slot.fusedChain.isFusedChainTail) {
        // Already counted above — these are "free" ops
      } else {
        // Heuristic: ops with no iArgs, 1-2 inputs, and no data dependency are likely elementwise
        if (!slot.isDataDependent() && slot.wiring.numInputs <= 2 && slot.wiring.numOutputs == 1) {
          numElementwise++;
        } else {
          numOther++;
        }
      }
    }

    int eliminatedOps = numIdentity + numFusedTails;
    int effectiveOps = segSize - eliminatedOps;

    DSP_DIAG(EMULATED_REPLAY,
             "  fusion analysis: %d total ops, %d effective (-%d identity, -%d fused tails)",
             segSize, effectiveOps, numIdentity, numFusedTails);
    DSP_DIAG(EMULATED_REPLAY,
             "    matmul=%d elementwise=%d view=%d inPlaceFused=%d dataDep=%d controlFlow=%d other=%d",
             numMatmul, numElementwise, numViewOps, numInPlaceFused,
             numDataDependent, numControlFlow, numOther);
    if (numFusedChains > 0) {
      DSP_DIAG(EMULATED_REPLAY,
               "    fused chains: %d chains covering %d ops (avg %.1f ops/chain)",
               numFusedChains, totalFusedChainOps,
               numFusedChains > 0 ? (float)totalFusedChainOps / numFusedChains : 0.0f);
    }

    // Segment pattern classification
    const char* pattern = "MIXED";
    if (numDataDependent > 0)           pattern = "DATA_DEPENDENT (non-capturable)";
    else if (numMatmul > 0 && numElementwise > 0) pattern = "MATMUL_EPILOGUE (best for graph capture)";
    else if (numMatmul > 0)             pattern = "PURE_MATMUL (cuBLAS graph capture)";
    else if (numElementwise == effectiveOps) pattern = "PURE_ELEMENTWISE (best for kernel fusion)";
    else if (numViewOps == segSize)      pattern = "PURE_VIEW (zero compute, identity graph)";

    DSP_DIAG(EMULATED_REPLAY, "    segment pattern: %s", pattern);

    // ── Gap 4: DOT graph topology ────────────────────────────────────────
    if (DSP_DIAG_ENABLED(EMULATED_REPLAY)) {
      DSP_DIAG(EMULATED_REPLAY, "  DOT_BEGIN seg[%d-%d]", seg.def.startSlot, seg.def.endSlot);
      DSP_DIAG(EMULATED_REPLAY, "  digraph segment_%d_%d {", seg.def.startSlot, seg.def.endSlot);
      DSP_DIAG(EMULATED_REPLAY, "    rankdir=TB;");
      DSP_DIAG(EMULATED_REPLAY, "    node [shape=box, fontsize=10];");

      // External input nodes
      std::unordered_set<int> emittedExt;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (emittedExt.count(extIdx)) continue;
            emittedExt.insert(extIdx);
            const char* srcLabel = "EXT";
            if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) srcLabel = "PH";
            else if (slot.wiring.inputSourceTypes[i] == SOURCE_CONSTANT) srcLabel = "CONST";
            else if (slot.wiring.inputSourceTypes[i] == SOURCE_VARIABLE) srcLabel = "VAR";
            DSP_DIAG(EMULATED_REPLAY,
                     "    ext_%d [label=\"%s[%d]\", shape=ellipse, style=filled, fillcolor=lightblue];",
                     extIdx, srcLabel, extIdx);
          }
        }
      }

      // Op nodes
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        const char* color = "white";
        if (slot.isIdentityOp())         color = "gray90";
        else if (slot.fusedChain.isFusedChainHead) color = "lightyellow";
        else if (slot.fusedChain.isFusedChainTail) color = "lightyellow";
        else if (slot.isViewCapableOp())  color = "honeydew";
        else if (slot.isDataDependent())  color = "mistyrose";

        DSP_DIAG(EMULATED_REPLAY,
                 "    slot_%d [label=\"[%d] %s\", style=filled, fillcolor=%s];",
                 s, s, slot.ident.opName.empty() ? "?" : slot.ident.opName.c_str(), color);
      }

      // Edges
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            DSP_DIAG(EMULATED_REPLAY, "    ext_%d -> slot_%d;", extIdx, s);
          } else if (srcIdx >= 0) {
            // Find which slot produces this output
            bool foundProducer = false;
            for (int ps = seg.def.startSlot; ps < s && !foundProducer; ps++) {
              NativeSlot& pslot = slots_[ps];
              for (int o = 0; o < pslot.wiring.numOutputs; o++) {
                if (pslot.wiring.outputSlotIndices[o] == srcIdx) {
                  DSP_DIAG(EMULATED_REPLAY, "    slot_%d -> slot_%d;", ps, s);
                  foundProducer = true;
                  break;
                }
              }
            }
            if (!foundProducer) {
              // Cross-segment input
              DSP_DIAG(EMULATED_REPLAY, "    cross_%d [label=\"slot[%d]\", shape=diamond];", srcIdx, srcIdx);
              DSP_DIAG(EMULATED_REPLAY, "    cross_%d -> slot_%d;", srcIdx, s);
            }
          }
        }
      }

      DSP_DIAG(EMULATED_REPLAY, "  }");
      DSP_DIAG(EMULATED_REPLAY, "  DOT_END seg[%d-%d]", seg.def.startSlot, seg.def.endSlot);
    }

  } else if (!fastPath) {
    // ══════════════════════════════════════════════════════════════════════
    // POST-WARMUP: stability checks (not on fast path)
    // ══════════════════════════════════════════════════════════════════════
    bool shapeStable = (currentShapeKey == seg.exec.cachedShapeKey);
    bool addrStable = (currentAddrKey == seg.exec.capturedInputAddrKey);

    const char* shapeVerdict = shapeStable ? "STABLE" : "CHANGED";
    const char* addrVerdict = addrStable ? "STABLE" : "CHANGED";

    DSP_DIAG(EMULATED_REPLAY,
             "  stability: shape=%s (0x%llx vs cached 0x%llx) addr=%s (0x%llx vs cached 0x%llx)",
             shapeVerdict,
             (long long)currentShapeKey, (long long)seg.exec.cachedShapeKey,
             addrVerdict,
             (long long)currentAddrKey, (long long)seg.exec.capturedInputAddrKey);

    if (!shapeStable) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** SHAPE KEY CHANGED: CUDA graph would INVALIDATE and re-capture. "
               "Identify which input shapes changed between executions.");

      // Detailed: find which external inputs changed shape
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
              auto* arr = externalArrays[extIdx];
              DSP_DIAG(EMULATED_REPLAY,
                       "    ext[%d] type=%d shape=[%s] dtype=%d",
                       extIdx, (int)slot.wiring.inputSourceTypes[i],
                       ShapeUtils::shapeAsString(arr).c_str(),
                       (int)arr->dataType());
            }
          }
        }
      }

      seg.exec.cachedShapeKey = currentShapeKey;
    }

    if (!addrStable) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** ADDRESS KEY CHANGED: capture buffer D2D copies needed. "
               "Placeholders with new addresses require staging buffer updates.");
      seg.exec.recordReplayInputAddrKey(currentAddrKey,
                                        "emulated_replay_addr_changed");
    }

    // Replay readiness assessment
    if (shapeStable && addrStable) {
      seg.exec.markArgsCurrent();      // Generation counter: no refresh needed
      DSP_DIAG(EMULATED_REPLAY,
               "  REPLAY READY: shapes and addresses stable — "
               "CUDA graph replay would succeed without re-capture. (fast path enabled)");
    } else {
      seg.exec.markArgsStale();         // Generation counter: force refresh
      if (shapeStable && !addrStable) {
        DSP_DIAG(EMULATED_REPLAY,
                 "  REPLAY with D2D: shapes stable but addresses changed — "
                 "CUDA graph would replay after capture buffer D2D copies.");
      } else {
        DSP_DIAG(EMULATED_REPLAY,
                 "  RE-CAPTURE needed: shape change requires full graph re-capture.");
      }
    }
  }
  // else: fast path — no key computation, no stability check, just execute

  // ── Record or replay a functional command program ─────────────────────
  bool functionalRecordable = false;
  bool functionalReplaySucceeded = false;
  bool functionalProgramReady = false;
#if !defined(SD_VULKAN)
  bool functionalCaptureStarted = false;
  bool functionalReplayExpected = false;
  int functionalReplayCountBefore = 0;
  int functionalReplayDelta = 0;
  std::vector<FunctionalReplayPointerBinding> functionalPointerSnapshot;
  functionalRecordable = seg.def.isFunctionalReplayEligible && !hasControlFlow_;
  auto* functionalHandle =
      dynamic_cast<FunctionalReplayHandle*>(seg.exec.replayHandle.get());

  if (functionalRecordable && seg.exec.segPhase.needsCapture()) {
    if (seg.exec.replayHandle != nullptr && functionalHandle == nullptr) {
      DSP_DIAG(EMULATED_REPLAY,
               "  replacing non-functional replay handle backend=%s",
               seg.exec.replayHandle->backendName());
      seg.exec.replayHandle.reset();
    }
    if (seg.exec.replayHandle == nullptr) {
      seg.exec.replayHandle = GraphReplayFactory::createFunctional();
      functionalHandle =
          dynamic_cast<FunctionalReplayHandle*>(seg.exec.replayHandle.get());
    }
    if (functionalHandle == nullptr) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** FUNCTIONAL CAPTURE UNAVAILABLE for recordable seg[%d-%d]",
               seg.def.startSlot, seg.def.endSlot);
      SegmentLifecycle::invalidateSegmentCaptures(
          this, seg, "functional_replay_factory_unavailable");
      return Status::BAD_GRAPH;
    }

    if (!functionalHandle->hasReplayProgram()) {
      if (functionalHandle->getState() != ReplayState::EMPTY) {
        functionalHandle->abortCapture();
      }
      if (!functionalHandle->beginCapture(nullptr)) {
        DSP_DIAG(EMULATED_REPLAY,
                 "  ** FUNCTIONAL CAPTURE BEGIN FAILED for seg[%d-%d]",
                 seg.def.startSlot, seg.def.endSlot);
        SegmentLifecycle::invalidateSegmentCaptures(
            this, seg, "functional_replay_begin_failed");
        return Status::BAD_GRAPH;
      }
      functionalCaptureStarted = true;
      DSP_DIAG(EMULATED_REPLAY,
               "  FUNCTIONAL CAPTURE: recording seg[%d-%d]",
               seg.def.startSlot, seg.def.endSlot);
    }
  }

  functionalReplayExpected =
      functionalHandle != nullptr &&
      functionalHandle->hasReplayProgram() &&
      !planLifecycle_.isSlotBySlot() &&
      executeCount_ >= 2 && !hasControlFlow_;
  if (functionalHandle != nullptr) {
    functionalReplayCountBefore =
        functionalHandle->getStatistics().replayCount;
  }

  if (functionalReplayExpected) {
    Status pointerStatus = buildFunctionalReplayPointerSnapshot(
        seg, slots_, numSlots_, externalArrays, numExt,
        outputSlots_, totalOutputSlots_, &functionalPointerSnapshot);
    if (pointerStatus == Status::OK) {
      pointerStatus =
          functionalHandle->validatePointerSnapshotForReplay(
              functionalPointerSnapshot);
    }
    if (pointerStatus != Status::OK) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** FUNCTIONAL POINTER PREFLIGHT FAILED: seg[%d-%d] "
               "status=%s (%d) bindings=%zu",
               seg.def.startSlot, seg.def.endSlot,
               statusName_seg(pointerStatus),
               static_cast<int>(pointerStatus),
               functionalPointerSnapshot.size());
      SegmentLifecycle::invalidateSegmentCaptures(
          this, seg, "functional_replay_pointer_preflight_failed");
      return pointerStatus;
    }
  }
#endif

  auto tSlotStart = std::chrono::high_resolution_clock::now();
  Status status = Status::OK;
  try {
    if (functionalCaptureStarted || functionalReplayExpected) {
      // Functional replay owns the alias publication for this invocation. A
      // fresh placeholder wrapper is expected when callers replace/close
      // inputs between calls, so allow identity/view publication to refresh
      // under the existing narrow shape-change guard without relaxing the
      // ordinary frozen-phase ownership checks elsewhere.
      ShapeChangeWarmupGuard functionalPublicationGuard(
          *this, seg.def.startSlot, seg.def.endSlot);
      status = executeSegmentSlotBySlot(
          seg, externalArrays, numExt, stream);
    } else {
      status = executeSegmentSlotBySlot(
          seg, externalArrays, numExt, stream);
    }
  } catch (...) {
#if !defined(SD_VULKAN)
    if (functionalCaptureStarted && functionalHandle != nullptr) {
      functionalHandle->abortCapture();
    }
    if (functionalCaptureStarted || functionalReplayExpected) {
      SegmentLifecycle::invalidateSegmentCaptures(
          this, seg,
          functionalCaptureStarted
              ? "functional_replay_capture_exception"
              : "functional_replay_execution_exception");
    }
#endif
    throw;
  }

  auto tSlotEnd = std::chrono::high_resolution_clock::now();
  auto slotUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    tSlotEnd - tSlotStart)
                    .count();

  // Dispatch overhead estimate: ~15us per effective op (shape inference + dispatch)
  // Identity/fused-tail ops are skipped by executeSlot, so don't count them.
  int estimatedSkippedOps = 0;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    if (slots_[s].isIdentityOp() ||
        slots_[s].fusedChain.isFusedChainTail) {
      estimatedSkippedOps++;
    }
  }
  int effectiveDispatchOps = segSize - estimatedSkippedOps;
  long long estimatedDispatchUs = effectiveDispatchOps * 15LL;

  DSP_DIAG(EMULATED_REPLAY,
           "  execution: %lldus total (%d ops, %d dispatched, %d skipped)%s",
           (long long)slotUs, segSize, effectiveDispatchOps, estimatedSkippedOps,
           fastPath ? " [FAST PATH]" : "");
  DSP_DIAG(EMULATED_REPLAY,
           "  overhead estimate: ~%lldus dispatch + ~%lldus compute = %lldus. "
           "Graph replay would save ~%lldus (%.0f%%)",
           estimatedDispatchUs,
           (long long)slotUs - estimatedDispatchUs,
           (long long)slotUs,
           estimatedDispatchUs,
           slotUs > 0 ? (100.0 * estimatedDispatchUs / slotUs) : 0.0);

  if (status != Status::OK) {
    seg.exec.markArgsStale();
    DSP_DIAG(EMULATED_REPLAY,
             "  ** EXECUTION FAILED: status=%s (%d)",
             statusName_seg(status), static_cast<int>(status));
#if !defined(SD_VULKAN)
    if (functionalCaptureStarted && functionalHandle != nullptr) {
      functionalHandle->abortCapture();
    }
    if (functionalCaptureStarted || functionalReplayExpected) {
      SegmentLifecycle::invalidateSegmentCaptures(
          this, seg,
          functionalCaptureStarted
              ? "functional_replay_capture_failed"
              : "functional_replay_execution_failed");
    }
#endif
    return status;
  }

#if !defined(SD_VULKAN)
  if (functionalCaptureStarted) {
    if (!functionalHandle->endCapture(nullptr) ||
        !functionalHandle->finalize()) {
      functionalHandle->abortCapture();
      DSP_DIAG(EMULATED_REPLAY,
               "  ** FUNCTIONAL CAPTURE FINALIZE FAILED for seg[%d-%d]",
               seg.def.startSlot, seg.def.endSlot);
      SegmentLifecycle::invalidateSegmentCaptures(
          this, seg, "functional_replay_finalize_failed");
      return Status::BAD_GRAPH;
    }
    DSP_DIAG(EMULATED_REPLAY,
             "  FUNCTIONAL CAPTURE READY: seg[%d-%d] commands=%d",
             seg.def.startSlot, seg.def.endSlot,
             functionalHandle->getRecordedOpCount());
  }

  functionalProgramReady =
      functionalHandle != nullptr &&
      functionalHandle->hasReplayProgram();
  if (functionalHandle != nullptr) {
    int replayCountAfter =
        functionalHandle->getStatistics().replayCount;
    functionalReplayDelta =
        replayCountAfter - functionalReplayCountBefore;
    functionalReplaySucceeded = functionalReplayDelta > 0;
  }

  if ((functionalCaptureStarted || functionalReplaySucceeded) &&
      functionalHandle != nullptr) {
    functionalPointerSnapshot.clear();
    Status pointerStatus = buildFunctionalReplayPointerSnapshot(
        seg, slots_, numSlots_, externalArrays, numExt,
        outputSlots_, totalOutputSlots_, &functionalPointerSnapshot);
    if (pointerStatus == Status::OK) {
      pointerStatus = functionalCaptureStarted
                          ? functionalHandle->publishPointerSnapshot(
                                functionalPointerSnapshot)
                          : functionalHandle->commitPointerSnapshot(
                                functionalPointerSnapshot);
    }
    if (pointerStatus != Status::OK) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** FUNCTIONAL POINTER COMMIT FAILED: seg[%d-%d] "
               "phase=%s status=%s (%d) bindings=%zu",
               seg.def.startSlot, seg.def.endSlot,
               functionalCaptureStarted ? "capture" : "replay",
               statusName_seg(pointerStatus),
               static_cast<int>(pointerStatus),
               functionalPointerSnapshot.size());
      SegmentLifecycle::invalidateSegmentCaptures(
          this, seg,
          functionalCaptureStarted
              ? "functional_replay_pointer_publish_failed"
              : "functional_replay_pointer_commit_failed");
      return pointerStatus;
    }

    if (functionalCaptureStarted) {
      DSP_DIAG(EMULATED_REPLAY,
               "  FUNCTIONAL POINTER SNAPSHOT: seg[%d-%d] bindings=%zu",
               seg.def.startSlot, seg.def.endSlot,
               functionalPointerSnapshot.size());
    } else {
      const auto& changes = functionalHandle->getLastPointerChanges();
      DSP_DIAG(EMULATED_REPLAY,
               "  FUNCTIONAL POINTER CHECK: seg[%d-%d] bindings=%d "
               "changed=%d array=%d dataBuffer=%d primary=%d special=%d "
               "shapeInfo=%d offset=%d metadata=%d comparisons=%lld",
               seg.def.startSlot, seg.def.endSlot,
               changes.bindingCount, changes.changedBindings,
               changes.arrayChanges, changes.dataBufferChanges,
               changes.primaryBufferChanges, changes.specialBufferChanges,
               changes.shapeInfoChanges, changes.offsetChanges,
               changes.metadataChanges,
               functionalHandle->getPointerComparisonCount());
    }
  }

  if (functionalReplaySucceeded) {
    seg.exec.lastReplayExecCount = executeCount_;
    totalGraphReplays_ += functionalReplayDelta;
  }
#endif

  // ── EMULATED_REPLAY segPhase lifecycle transitions ─────────────────────
  // The first pass warms the segment. The second pass records and publishes an
  // immutable functional program for static capturable segments. Later passes
  // replay that program. Data-dependent/control-flow segments remain explicit
  // diagnostic slot-by-slot emulation and never report a replay launch.
  const bool ownsFunctionalReplayLifecycle =
      seg.def.selectedBackend == SelectedBackend::EMULATED_REPLAY;
  if (ownsFunctionalReplayLifecycle && execCount == 0 &&
      seg.exec.segPhase.needsWarmup()) {
    DSP_DIAG(EMULATED_REPLAY,
             "  LIFECYCLE: seg[%d-%d] BUILDING:WARMUP -> BUILDING:CAPTURING "
             "(warmup done, execCount=%d)",
             seg.def.startSlot, seg.def.endSlot, execCount);
    SegmentLifecycle::skipToCapturing(
        seg.exec, "emulated_replay",
        seg.def.startSlot, seg.def.endSlot);
  } else if (ownsFunctionalReplayLifecycle && execCount >= 1 &&
             seg.exec.segPhase.needsCapture()) {
    if (functionalRecordable) {
      if (!functionalProgramReady) {
        DSP_DIAG(EMULATED_REPLAY,
                 "  ** FUNCTIONAL PROGRAM MISSING for recordable seg[%d-%d]",
                 seg.def.startSlot, seg.def.endSlot);
        SegmentLifecycle::invalidateSegmentCaptures(
            this, seg, "functional_replay_program_missing");
        return Status::BAD_GRAPH;
      }
      SegmentLifecycle::markFunctionalReplaySealed(
          seg.exec, seg.def.startSlot, seg.def.endSlot);
    } else {
      SegmentLifecycle::markEmulatedSealed(
          seg.exec, seg.def.startSlot, seg.def.endSlot);
    }
  }

  if (functionalReplaySucceeded) {
    DSP_DIAG(EMULATED_REPLAY,
             "  FUNCTIONAL REPLAY OK: seg[%d-%d] totalReplays=%lld",
             seg.def.startSlot, seg.def.endSlot,
             static_cast<long long>(totalGraphReplays_));
  }

  // executeSegmentSlotBySlot increments seg.exec.executionCount after either
  // normal execution or successful functional replay.
  return Status::OK;
}

}  // namespace graph
}  // namespace sd
