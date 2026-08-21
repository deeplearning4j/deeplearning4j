/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <dsp/runtime/dsp_runtime_c.h>
#include <dsp/runtime/detail/DspRuntimeInternal.h>
#include <dsp/runtime/detail/SdxTextGenerationMetadata.h>

#include <array/DataTypeUtils.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <execution/LaunchContext.h>
#include <graph/DspDiagnostics.h>
#include <ops/declarable/helpers/autoregressive_decode.h>
#include <ops/declarable/helpers/token_sample.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using sd::DataType;
using sd::LongType;
using sd::NDArray;
using sd::NDArrayFactory;
using sd::dsp::runtime::detail::TextGenerationMetadata;
using sd::dsp::runtime::detail::TextGenerationRecurrentState;
using sd::dsp::runtime::detail::TextGenerationRecurrentStateKind;
using sd::ops::helpers::AutoregressiveDecodeConfig;
using sd::ops::helpers::TokenSampleConfig;
using sd::ops::helpers::TokenSampleResult;

struct NamedArray {
  std::string name;
  std::unique_ptr<NDArray> array;
};

struct GenerationPolicy {
  int32_t maxNewTokens = 0;
  int32_t minNewTokens = 0;
  double temperature = 0.0;
  int32_t topK = 0;
  double topP = 1.0;
  double minP = 0.0;
  double repetitionPenalty = 1.0;
  double frequencyPenalty = 0.0;
  double presencePenalty = 0.0;
  double typicalP = 1.0;
  double xtcProbability = 0.0;
  double xtcThreshold = 0.1;
  int64_t seed = 0;
};

uint64_t elapsedNanos(Clock::time_point start, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

bool compatibleStructSize(uint32_t size, size_t required) {
  return size == 0 || size >= required;
}

bool finite(double value) {
  return std::isfinite(value);
}

void assignScalar(NDArray* array, double value) {
  array->assign(value);
}

std::unique_ptr<NDArray> createArray(
    const std::vector<LongType>& shape,
    DataType dataType) {
  return std::unique_ptr<NDArray>(NDArrayFactory::create(
      'c', shape, dataType, sd::LaunchContext::defaultContext()));
}

std::unique_ptr<NDArray> createLongScalar(LongType value) {
  return std::unique_ptr<NDArray>(
      NDArrayFactory::create<LongType>(
          value, sd::LaunchContext::defaultContext()));
}

NDArray* findNamed(
    std::vector<NamedArray>& arrays,
    const std::string& name) {
  for (auto& entry : arrays) {
    if (entry.name == name) return entry.array.get();
  }
  return nullptr;
}

const NDArray* findNamed(
    const std::vector<NamedArray>& arrays,
    const std::string& name) {
  for (const auto& entry : arrays) {
    if (entry.name == name) return entry.array.get();
  }
  return nullptr;
}

bool addNamed(
    std::vector<NamedArray>* arrays,
    const std::string& name,
    std::unique_ptr<NDArray> array,
    std::string* error) {
  if (arrays == nullptr || name.empty() || array == nullptr) {
    if (error != nullptr) *error = "cannot bind an empty generation input";
    return false;
  }
  if (findNamed(*arrays, name) != nullptr) {
    if (error != nullptr) {
      *error = "duplicate generation input name in metadata: " + name;
    }
    return false;
  }
  arrays->push_back(NamedArray{name, std::move(array)});
  return true;
}

bool isEos(const TextGenerationMetadata& metadata, LongType token) {
  return std::find(
             metadata.eosIds.begin(),
             metadata.eosIds.end(),
             static_cast<int>(token)) != metadata.eosIds.end();
}

double maskFillValue(DataType dataType) {
  return dataType == DataType::HALF ? -65504.0 : -1.0e9;
}

bool outputIndexByName(
    sdx_context_t* context,
    const std::string& name,
    int32_t* outIndex) {
  if (context == nullptr || outIndex == nullptr || name.empty()) return false;
  const int32_t count = sdxGetNumOutputs(context);
  for (int32_t i = 0; i < count; ++i) {
    const char* candidate = sdxGetOutputName(context, i);
    if (candidate != nullptr && name == candidate) {
      *outIndex = i;
      return true;
    }
  }
  return false;
}

bool inputIndexByName(
    sdx_context_t* context,
    const std::string& name,
    int32_t* outIndex) {
  if (context == nullptr || outIndex == nullptr || name.empty()) return false;
  const int32_t count = sdxGetNumInputs(context);
  for (int32_t i = 0; i < count; ++i) {
    const char* candidate = sdxGetInputName(context, i);
    if (candidate != nullptr && name == candidate) {
      *outIndex = i;
      return true;
    }
  }
  return false;
}

bool orderPublicInputs(
    sdx_context_t* context,
    std::vector<NamedArray>& named,
    std::vector<NDArray*>* ordered,
    std::string* error) {
  if (context == nullptr || ordered == nullptr) {
    if (error != nullptr) *error = "generation context is null";
    return false;
  }

  const int32_t count = sdxGetNumInputs(context);
  if (count < 0) {
    if (error != nullptr) *error = "generation context has no input signature";
    return false;
  }

  ordered->assign(static_cast<size_t>(count), nullptr);
  std::unordered_set<std::string> consumed;
  for (int32_t i = 0; i < count; ++i) {
    const char* rawName = sdxGetInputName(context, i);
    if (rawName == nullptr || rawName[0] == '\0') {
      if (error != nullptr) {
        *error = "generation context contains an unnamed public input";
      }
      return false;
    }
    const std::string name(rawName);
    NDArray* array = findNamed(named, name);
    if (array == nullptr) {
      if (error != nullptr) {
        *error = "text-generation metadata does not bind public graph input: " +
                 name;
      }
      return false;
    }
    (*ordered)[static_cast<size_t>(i)] = array;
    consumed.insert(name);
  }

  if (consumed.size() != named.size()) {
    for (const auto& entry : named) {
      if (consumed.find(entry.name) == consumed.end()) {
        if (error != nullptr) {
          *error = "text-generation metadata names an input absent from the "
                   "bound graph signature: " +
                   entry.name;
        }
        return false;
      }
    }
  }
  return true;
}

sdx_status_t createBoundContext(
    sdx_model_t* model,
    const std::vector<std::string>& outputNames,
    sdx_context_t** outContext) {
  std::vector<const char*> rawNames;
  rawNames.reserve(outputNames.size());
  for (const auto& name : outputNames) rawNames.push_back(name.c_str());

  sdx_context_options_t options{};
  options.struct_size = sizeof(options);
  options.bind_model_parameters = 1;
  return sdxCreateContextWithOptions(
      model,
      rawNames.empty() ? nullptr : rawNames.data(),
      static_cast<int32_t>(rawNames.size()),
      &options,
      outContext);
}

bool validatePolicy(const GenerationPolicy& policy, std::string* error) {
  if (policy.maxNewTokens <= 0) {
    if (error != nullptr) *error = "max_new_tokens must be positive";
    return false;
  }
  if (policy.minNewTokens < 0 ||
      policy.minNewTokens > policy.maxNewTokens) {
    if (error != nullptr) {
      *error = "min_new_tokens must be in [0, max_new_tokens]";
    }
    return false;
  }
  if (policy.topK < 0) {
    if (error != nullptr) *error = "top_k cannot be negative";
    return false;
  }
  if (!finite(policy.temperature) || policy.temperature < 0.0 ||
      !finite(policy.topP) || !finite(policy.minP) ||
      policy.minP < 0.0 || policy.minP > 1.0 ||
      !finite(policy.repetitionPenalty) ||
      policy.repetitionPenalty <= 0.0 ||
      !finite(policy.frequencyPenalty) ||
      !finite(policy.presencePenalty) ||
      !finite(policy.typicalP) ||
      policy.typicalP <= 0.0 || policy.typicalP > 1.0 ||
      !finite(policy.xtcProbability) ||
      policy.xtcProbability < 0.0 || policy.xtcProbability > 1.0 ||
      !finite(policy.xtcThreshold) ||
      policy.xtcThreshold < 0.0 || policy.xtcThreshold > 1.0) {
    if (error != nullptr) {
      *error = "generation sampling options contain an invalid value";
    }
    return false;
  }
  return true;
}

GenerationPolicy metadataPolicy(const TextGenerationMetadata& metadata) {
  GenerationPolicy result;
  result.maxNewTokens = metadata.sampling.maxNewTokens;
  result.minNewTokens = metadata.sampling.minNewTokens;
  result.temperature = metadata.sampling.temperature;
  result.topK = metadata.sampling.topK;
  result.topP = metadata.sampling.topP;
  result.minP = metadata.sampling.minP;
  result.repetitionPenalty = metadata.sampling.repetitionPenalty;
  result.frequencyPenalty = metadata.sampling.frequencyPenalty;
  result.presencePenalty = metadata.sampling.presencePenalty;
  result.typicalP = metadata.sampling.typicalP;
  result.xtcProbability = metadata.sampling.xtcProbability;
  result.xtcThreshold = metadata.sampling.xtcThreshold;
  result.seed = metadata.sampling.seed;
  return result;
}

bool resolvePolicy(
    const TextGenerationMetadata& metadata,
    const sdx_generation_options_t* options,
    GenerationPolicy* out,
    std::string* error) {
  if (out == nullptr) return false;
  if (options == nullptr) {
    *out = metadataPolicy(metadata);
    return validatePolicy(*out, error);
  }
  if (!compatibleStructSize(options->struct_size, sizeof(*options))) {
    if (error != nullptr) {
      *error = "sdx_generation_options_t is smaller than this ABI requires";
    }
    return false;
  }

  GenerationPolicy result;
  result.maxNewTokens = options->max_new_tokens;
  result.minNewTokens = options->min_new_tokens;
  result.temperature = options->temperature;
  result.topK = options->top_k;
  result.topP = options->top_p;
  result.minP = options->min_p;
  result.repetitionPenalty =
      options->repetition_penalty <= 0.0 ? 1.0 : options->repetition_penalty;
  result.frequencyPenalty = options->frequency_penalty;
  result.presencePenalty = options->presence_penalty;
  result.typicalP = options->typical_p;
  result.xtcProbability = options->xtc_probability;
  result.xtcThreshold = options->xtc_threshold;
  result.seed = options->seed;
  if (!validatePolicy(result, error)) return false;
  *out = result;
  return true;
}

bool validateCallbacks(
    const sdx_generation_callbacks_t* callbacks,
    std::string* error) {
  if (callbacks == nullptr) return true;
  if (!compatibleStructSize(callbacks->struct_size, sizeof(*callbacks))) {
    if (error != nullptr) {
      *error = "sdx_generation_callbacks_t is smaller than this ABI requires";
    }
    return false;
  }
  return true;
}

TokenSampleConfig sampleConfig(
    const GenerationPolicy& policy,
    const TextGenerationMetadata& metadata,
    int generatedOffset) {
  TokenSampleConfig config;
  config.temperature = policy.temperature;
  config.topK = policy.topK;
  config.topP = policy.topP;
  config.minP = policy.minP;
  config.repPenalty = policy.repetitionPenalty;
  config.freqPenalty = policy.frequencyPenalty;
  config.presPenalty = policy.presencePenalty;
  config.minNewTokens = policy.minNewTokens;
  config.generatedTokenOffset = generatedOffset;
  config.stopTokenIds =
      metadata.eosIds.empty() ? nullptr : metadata.eosIds.data();
  config.stopTokenCount = static_cast<int>(metadata.eosIds.size());
  config.seed = policy.seed > 0
                    ? static_cast<LongType>(policy.seed) + generatedOffset
                    : 0;
  config.typicalP = policy.typicalP;
  config.xtcProbability = policy.xtcProbability;
  config.xtcThreshold = policy.xtcThreshold;
  return config;
}

std::unique_ptr<NDArray> historyArray(
    const std::vector<LongType>& history) {
  if (history.empty()) return nullptr;
  auto result = createArray(
      {1, static_cast<LongType>(history.size())}, DataType::INT64);
  for (size_t i = 0; i < history.size(); ++i) {
    result->p(static_cast<LongType>(i), history[i]);
  }
  return result;
}

bool sampleFromLogits(
    NDArray* logits,
    int32_t requestedPosition,
    const std::vector<LongType>& history,
    const GenerationPolicy& policy,
    const TextGenerationMetadata& metadata,
    int generatedOffset,
    LongType* outToken,
    std::string* error) {
  if (logits == nullptr || outToken == nullptr) {
    if (error != nullptr) *error = "generation graph did not produce logits";
    return false;
  }

  NDArray* logitsForSample = logits;
  std::unique_ptr<NDArray> slice;
  const int rank = logits->rankOf();
  if (rank == 3) {
    if (logits->sizeAt(0) != 1 || logits->sizeAt(1) <= 0 ||
        logits->sizeAt(2) <= 0) {
      if (error != nullptr) {
        *error = "logits must have shape [1, sequence, vocabulary]";
      }
      return false;
    }
    const LongType sequence = logits->sizeAt(1);
    const LongType position =
        sequence == 1 ? 0 : static_cast<LongType>(requestedPosition);
    if (position < 0 || position >= sequence) {
      if (error != nullptr) {
        *error = "requested logits position is outside the graph output";
      }
      return false;
    }
    slice.reset((*logits)(
        std::vector<LongType>{
            0, 1, position, position + 1, 0, logits->sizeAt(2)},
        true));
    logitsForSample = slice.get();
  } else if (rank == 2) {
    if (logits->sizeAt(0) != 1 || logits->sizeAt(1) <= 0) {
      if (error != nullptr) *error = "rank-2 logits must have shape [1, vocabulary]";
      return false;
    }
  } else if (rank == 1) {
    if (logits->sizeAt(0) <= 0) {
      if (error != nullptr) *error = "logits vocabulary cannot be empty";
      return false;
    }
  } else {
    if (error != nullptr) {
      *error = "only rank-1, rank-2, and rank-3 logits are supported";
    }
    return false;
  }

  logitsForSample->syncToHost();
  auto sampled = createArray({1}, DataType::INT64);
  auto prior = historyArray(history);
  TokenSampleConfig config =
      sampleConfig(policy, metadata, generatedOffset);
  TokenSampleResult result;
  sd::ops::helpers::tokenSamplePolicy(
      logitsForSample,
      sampled.get(),
      prior.get(),
      config,
      &result,
      sd::LaunchContext::defaultContext());
  sampled->syncToHost();
  *outToken = sampled->e<LongType>(0);
  return true;
}

bool copyPrefillKv(
    NDArray* source,
    int32_t cacheCapacity,
    DataType expectedType,
    std::unique_ptr<NDArray>* out,
    std::string* error) {
  if (source == nullptr || out == nullptr) {
    if (error != nullptr) *error = "prefill graph did not produce a KV tensor";
    return false;
  }
  if (source->rankOf() != 4 || source->sizeAt(0) != 1 ||
      source->sizeAt(1) <= 0 || source->sizeAt(2) <= 0 ||
      source->sizeAt(3) <= 0 || source->sizeAt(1) > cacheCapacity) {
    if (error != nullptr) {
      *error = "prefill KV tensor must use BSHD [1, sequence, heads, dim]";
    }
    return false;
  }
  if (source->dataType() != expectedType) {
    if (error != nullptr) {
      *error = "prefill KV tensor dtype does not match text-generation metadata";
    }
    return false;
  }

  source->syncToHost();
  std::unique_ptr<NDArray> contiguous;
  NDArray* copySource = source;
  if (source->ordering() != 'c' || source->ews() != 1) {
    contiguous.reset(source->dup('c'));
    copySource = contiguous.get();
  }

  auto destination = createArray(
      {1, static_cast<LongType>(cacheCapacity), source->sizeAt(2),
       source->sizeAt(3)},
      expectedType);
  assignScalar(destination.get(), 0.0);

  const size_t elementSize =
      sd::DataTypeUtils::sizeOfElement(expectedType);
  const uint64_t elements = static_cast<uint64_t>(copySource->lengthOf());
  if (elementSize != 0 &&
      elements > std::numeric_limits<size_t>::max() / elementSize) {
    if (error != nullptr) *error = "prefill KV tensor byte size overflow";
    return false;
  }
  const size_t bytes = static_cast<size_t>(elements) * elementSize;
  std::memcpy(destination->buffer(), copySource->buffer(), bytes);
  destination->tickWriteHost();
  *out = std::move(destination);
  return true;
}

bool validateRecurrentArray(
    NDArray* array,
    const TextGenerationRecurrentState& state,
    std::string* error) {
  if (array == nullptr) {
    if (error != nullptr) {
      *error = "recurrent state output is null: " + state.output;
    }
    return false;
  }
  if (array->dataType() != state.dataType) {
    if (error != nullptr) {
      *error = "recurrent state dtype does not match metadata: " + state.output;
    }
    return false;
  }
  if (array->rankOf() != static_cast<LongType>(state.shape.size())) {
    if (error != nullptr) {
      *error = "recurrent state rank does not match metadata: " + state.output;
    }
    return false;
  }
  for (size_t i = 0; i < state.shape.size(); ++i) {
    if (array->sizeAt(static_cast<int>(i)) !=
        static_cast<LongType>(state.shape[i])) {
      if (error != nullptr) {
        *error = "recurrent state shape does not match metadata: " + state.output;
      }
      return false;
    }
  }
  return true;
}

std::unique_ptr<NDArray> createRecurrentArray(
    const TextGenerationRecurrentState& state) {
  std::vector<LongType> shape(state.shape.begin(), state.shape.end());
  // createArray() returns a host-owned, zero-initialized buffer. Do not route
  // initialization through assign(), which can make an accelerator/device
  // buffer authoritative and leave the host pointer unavailable to the SDX
  // tensor-view ABI.
  return createArray(shape, state.dataType);
}

bool copyRecurrentArrayInto(
    NDArray* source,
    NDArray* destination,
    const TextGenerationRecurrentState& state,
    std::string* error) {
  if (!validateRecurrentArray(source, state, error) ||
      !validateRecurrentArray(destination, state, error)) {
    return false;
  }

  // Recurrent outputs can be returned by an accelerator segment with the
  // device copy authoritative. Materialize both sides at this context
  // boundary, then copy bytes explicitly so the next host-only tensor view
  // never observes a borrowed or null pointer.
  source->forceSyncToHost();
  destination->forceSyncToHost();

  const size_t elementSize =
      sd::DataTypeUtils::sizeOfElement(state.dataType);
  const uint64_t elements = static_cast<uint64_t>(source->lengthOf());
  if (elementSize != 0 &&
      elements > std::numeric_limits<size_t>::max() / elementSize) {
    if (error != nullptr) {
      *error = "recurrent state byte size overflow: " + state.output;
    }
    return false;
  }
  const size_t bytes = static_cast<size_t>(elements) * elementSize;

  auto* sourceData = source->dataBuffer();
  auto* destinationData = destination->dataBuffer();
  if (bytes > 0 && (sourceData == nullptr || destinationData == nullptr)) {
    if (error != nullptr) {
      *error = "recurrent state has no DataBuffer: " + state.output;
    }
    return false;
  }
  if (bytes > 0 && destination->buffer() == nullptr) {
    // Plan-owned recurrent inputs may be metadata-only placeholders. They
    // still need a real host allocation before being exposed to SDX.
    destinationData->allocatePrimary();
  }

  void* sourceBuffer = source->buffer();
  void* destinationBuffer = destination->buffer();
  if (bytes > 0 && (sourceBuffer == nullptr || destinationBuffer == nullptr)) {
    if (error != nullptr) {
      *error = "recurrent state has no host buffer: " + state.output;
    }
    return false;
  }
  if (bytes > 0 &&
      (sourceData->getLenInBytes() < bytes ||
       destinationData->getLenInBytes() < bytes)) {
    if (error != nullptr) {
      *error = "recurrent state DataBuffer is smaller than its shape: " +
               state.output;
    }
    return false;
  }

  if (bytes > 0) {
    std::memcpy(destinationBuffer, sourceBuffer, bytes);
    destination->tickWriteHost();
  }
  return true;
}

bool copyRecurrentArray(
    NDArray* source,
    const TextGenerationRecurrentState& state,
    std::unique_ptr<NDArray>* out,
    std::string* error) {
  auto destination = createRecurrentArray(state);
  if (!copyRecurrentArrayInto(source, destination.get(), state, error)) {
    return false;
  }

  *out = std::move(destination);
  return true;
}

}  // namespace

struct sdx_generation_session {
  sdx_model_t* model = nullptr;
  TextGenerationMetadata metadata;
  std::mutex mutex;
  std::atomic<bool> cancelRequested{false};

  sdx_context_t* prefillContext = nullptr;
  sdx_context_t* decodeContext = nullptr;
  std::vector<NamedArray> prefillOwned;
  std::vector<NDArray*> prefillPublic;
  std::vector<NamedArray> decodeOwned;
  std::vector<NDArray*> decodePublic;

  std::vector<int> decodeKvPlanIndices;
  std::vector<NDArray*> decodeKvPlanArrays;
  std::vector<int> decodeGdnStatePlanIndices;
  std::vector<int> decodeGdnStateOutputIndices;
  std::vector<int> decodeConvStatePlanIndices;
  std::vector<int> decodeConvStateOutputIndices;
  int decodeInputIdsPlanIndex = -1;
  int decodeCausalMaskPlanIndex = -1;
  int decodePositionOffsetPlanIndex = -1;
  int decodeCachePositionPlanIndex = -1;
  int decodeActualSequenceLengthPlanIndex = -1;
  int decodeLogitsOutputIndex = -1;

  std::unique_ptr<NDArray> dummyEmbedding;
  std::unique_ptr<NDArray> dummyEmbeddingTable;

  bool hasPrompt = false;
  bool decodeReady = false;
  bool reachedEos = false;
  int32_t promptTokenCount = 0;
  int32_t cachePosition = 0;
  int32_t activeContextCapacity = 0;
  LongType lastToken = -1;
  int32_t totalGenerated = 0;
  std::vector<LongType> history;

  uint64_t lastPrefillNanos = 0;
  bool hasExecutionReport = false;
  sdx_execution_report_t lastExecutionReport{};
};

namespace {

void destroyContext(sdx_context_t** context) {
  if (context != nullptr && *context != nullptr) {
    sdxDestroyContext(*context);
    *context = nullptr;
  }
}

void captureExecutionReport(
    sdx_generation_session_t* session,
    const sdx_context_t* context) {
  if (session == nullptr || context == nullptr) return;
  sdx_execution_report_t report{};
  report.struct_size = sizeof(report);
  if (sdxGetExecutionReport(context, &report) == SDX_STATUS_OK) {
    session->lastExecutionReport = report;
    session->hasExecutionReport = true;
  }
}

void clearExecutionState(sdx_generation_session_t* session) {
  if (session == nullptr) return;
  destroyContext(&session->decodeContext);
  destroyContext(&session->prefillContext);
  session->decodeKvPlanArrays.clear();
  session->decodeKvPlanIndices.clear();
  session->decodeGdnStatePlanIndices.clear();
  session->decodeGdnStateOutputIndices.clear();
  session->decodeConvStatePlanIndices.clear();
  session->decodeConvStateOutputIndices.clear();
  session->decodePublic.clear();
  session->decodeOwned.clear();
  session->prefillPublic.clear();
  session->prefillOwned.clear();
  session->dummyEmbedding.reset();
  session->dummyEmbeddingTable.reset();
  session->decodeInputIdsPlanIndex = -1;
  session->decodeCausalMaskPlanIndex = -1;
  session->decodePositionOffsetPlanIndex = -1;
  session->decodeCachePositionPlanIndex = -1;
  session->decodeActualSequenceLengthPlanIndex = -1;
  session->decodeLogitsOutputIndex = -1;
  session->hasPrompt = false;
  session->decodeReady = false;
  session->reachedEos = false;
  session->promptTokenCount = 0;
  session->cachePosition = 0;
  session->activeContextCapacity = 0;
  session->lastToken = -1;
  session->totalGenerated = 0;
  session->history.clear();
  session->lastPrefillNanos = 0;
  session->hasExecutionReport = false;
  session->lastExecutionReport = {};
}

sdx_status_t fail(
    sdx_generation_session_t* session,
    sdx_status_t status,
    const std::string& message) {
  if (session != nullptr) {
    sd::dsp::runtime::detail::setModelError(session->model, message);
  }
  return status;
}

void clearError(sdx_generation_session_t* session) {
  if (session != nullptr) {
    sd::dsp::runtime::detail::setModelError(session->model, "");
  }
}

bool cancellationRequested(
    sdx_generation_session_t* session,
    const sdx_generation_callbacks_t* callbacks) {
  if (session == nullptr) return true;
  if (session->cancelRequested.load(std::memory_order_acquire)) return true;
  return callbacks != nullptr && callbacks->should_cancel != nullptr &&
         callbacks->should_cancel(callbacks->user_data) != 0;
}

void commitToken(
    sdx_generation_session_t* session,
    LongType token,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* output,
    int32_t capacity,
    int32_t* count) {
  const int32_t index = *count;
  if (output != nullptr && index < capacity) output[index] = token;
  *count = index + 1;
  session->lastToken = token;
  session->totalGenerated++;
  session->history.push_back(token);
  if (isEos(session->metadata, token)) session->reachedEos = true;
  if (callbacks != nullptr && callbacks->on_token != nullptr) {
    callbacks->on_token(token, callbacks->user_data);
  }
}

void fillReport(
    sdx_generation_session_t* session,
    int32_t finishReason,
    int32_t generatedThisCall,
    uint64_t elapsed,
    uint64_t prefill,
    uint64_t decode,
    sdx_generation_report_t* outReport) {
  if (outReport == nullptr || session == nullptr) return;
  sdx_generation_report_t report{};
  report.struct_size = sizeof(report);
  report.finish_reason = finishReason;
  report.prompt_token_count = session->promptTokenCount;
  report.generated_token_count = generatedThisCall;
  report.total_generated_token_count = session->totalGenerated;
  report.context_position = session->cachePosition;
  report.elapsed_time_ns = elapsed;
  report.prefill_time_ns = prefill;
  report.decode_time_ns = decode;
  report.decode_tokens_per_second =
      decode == 0
          ? 0.0
          : static_cast<double>(generatedThisCall) * 1.0e9 /
                static_cast<double>(decode);
  report.backend_report_available = 0;
  report.requested_backend = static_cast<int32_t>(SDX_BACKEND_AUTO);
  report.applied_backend = static_cast<int32_t>(SDX_BACKEND_AUTO);
  report.backend_status_code = static_cast<int32_t>(SDX_STATUS_OK);
  report.used_fallback = -1;
  report.requested_gpu_target = static_cast<int32_t>(SDX_GPU_TARGET_AUTO);
  report.applied_gpu_target = static_cast<int32_t>(SDX_GPU_TARGET_AUTO);
  report.plan_phase = -1;
  report.execution_count = 0;

  // A route name or requested model option is not proof of execution. The
  // report is captured immediately after native execution and retained even
  // when the short-lived prefill context is destroyed before a one-token call
  // returns.
  if (session->hasExecutionReport) {
    const auto& execution = session->lastExecutionReport;
    report.backend_report_available = 1;
    report.requested_backend = execution.requested_backend;
    report.applied_backend = execution.applied_backend;
    report.backend_status_code = execution.status_code;
    report.used_fallback = execution.used_fallback;
    report.requested_gpu_target = execution.requested_gpu_target;
    report.applied_gpu_target = execution.applied_gpu_target;
    report.plan_phase = execution.plan_phase;
    report.execution_count = execution.execution_count;
  }

  size_t destinationSize = outReport->struct_size;
  if (destinationSize == 0) destinationSize = sizeof(report);
  std::memcpy(outReport, &report, std::min(destinationSize, sizeof(report)));
}

bool addPrefillInputs(
    sdx_generation_session_t* session,
    const int64_t* prompt,
    int32_t promptCount,
    std::string* error) {
  const auto& metadata = session->metadata;
  // The graph inputs are dynamically shaped. maxPrefillLength is an admission
  // limit, not a physical buffer size: materializing it here makes the causal
  // mask quadratic in the model's advertised context even for a tiny prompt.
  auto inputIds = createArray(
      {1, static_cast<LongType>(promptCount)},
      DataType::INT64);
  for (int32_t i = 0; i < promptCount; ++i) {
    inputIds->p(static_cast<LongType>(i), static_cast<LongType>(prompt[i]));
  }
  if (!addNamed(
          &session->prefillOwned,
          metadata.inputIds,
          std::move(inputIds),
          error)) {
    return false;
  }

  auto mask = createArray(
      {1, 1, static_cast<LongType>(promptCount),
       static_cast<LongType>(promptCount)},
      metadata.maskDataType);
  assignScalar(mask.get(), maskFillValue(metadata.maskDataType));
  for (int32_t row = 0; row < promptCount; ++row) {
    for (int32_t column = 0; column <= row; ++column) {
      const LongType index =
          static_cast<LongType>(row) * promptCount + column;
      mask->p(index, 0.0);
    }
  }
  if (!addNamed(
          &session->prefillOwned,
          metadata.causalMask,
          std::move(mask),
          error) ||
      !addNamed(
          &session->prefillOwned,
          metadata.positionOffset,
          createLongScalar(0),
          error) ||
      !addNamed(
          &session->prefillOwned,
          metadata.cachePosition,
          createLongScalar(0),
          error) ||
      !addNamed(
          &session->prefillOwned,
          metadata.actualSequenceLength,
          createLongScalar(promptCount),
          error)) {
    return false;
  }

  for (const auto& name : metadata.kvKeyInputs) {
    if (!addNamed(
            &session->prefillOwned,
            name,
            createArray({0}, metadata.kvDataType),
            error)) {
      return false;
    }
  }
  for (const auto& name : metadata.kvValueInputs) {
    if (!addNamed(
            &session->prefillOwned,
            name,
            createArray({0}, metadata.kvDataType),
            error)) {
      return false;
    }
  }
  for (const auto& state : metadata.recurrentStates) {
    if (!addNamed(
            &session->prefillOwned,
            state.input,
            createRecurrentArray(state),
            error)) {
      return false;
    }
  }
  return true;
}

sdx_status_t runPrefill(
    sdx_generation_session_t* session,
    const int64_t* prompt,
    int32_t promptCount,
    const GenerationPolicy& policy,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* output,
    int32_t capacity,
    int32_t* count) {
  const auto start = Clock::now();
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_BEGIN phase=prefill prompt_tokens=%d cache_capacity=%d",
      promptCount,
      session->activeContextCapacity);
  std::string error;
  if (!addPrefillInputs(session, prompt, promptCount, &error)) {
    return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
  }

  std::vector<std::string> outputNames;
  outputNames.reserve(
      1 + session->metadata.prefillKeyOutputs.size() +
      session->metadata.prefillValueOutputs.size() +
      session->metadata.recurrentStates.size());
  outputNames.push_back(session->metadata.prefillLogits);
  outputNames.insert(
      outputNames.end(),
      session->metadata.prefillKeyOutputs.begin(),
      session->metadata.prefillKeyOutputs.end());
  outputNames.insert(
      outputNames.end(),
      session->metadata.prefillValueOutputs.begin(),
      session->metadata.prefillValueOutputs.end());
  for (const auto& state : session->metadata.recurrentStates) {
    outputNames.push_back(state.output);
  }

  sdx_status_t status =
      createBoundContext(session->model, outputNames, &session->prefillContext);
  if (status != SDX_STATUS_OK) return status;
  if (!orderPublicInputs(
          session->prefillContext,
          session->prefillOwned,
          &session->prefillPublic,
          &error)) {
    return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
  }

  const auto prefillExecuteStart = Clock::now();
  DSP_DIAG(TIMING, "SDX_PHASE_BEGIN phase=prefill_plan_execute");
  status = sd::dsp::runtime::detail::runOwnedArrays(
      session->prefillContext, session->prefillPublic);
  if (status != SDX_STATUS_OK) return status;
  captureExecutionReport(session, session->prefillContext);
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_END phase=prefill_plan_execute elapsed_us=%llu plan_phase=%d execution_count=%d",
      static_cast<unsigned long long>(
          elapsedNanos(prefillExecuteStart, Clock::now()) / 1000ULL),
      session->lastExecutionReport.plan_phase,
      session->lastExecutionReport.execution_count);
  if (cancellationRequested(session, callbacks)) {
    session->lastPrefillNanos = elapsedNanos(start, Clock::now());
    return SDX_STATUS_OK;
  }

  int32_t logitsIndex = -1;
  if (!outputIndexByName(
          session->prefillContext,
          session->metadata.prefillLogits,
          &logitsIndex)) {
    return fail(
        session,
        SDX_STATUS_EXECUTION_FAILED,
        "prefill logits output is absent from the compiled graph");
  }
  NDArray* logits =
      sd::dsp::runtime::detail::contextOutputArray(
          session->prefillContext, logitsIndex);
  LongType firstToken = -1;
  if (!sampleFromLogits(
          logits,
          promptCount - 1,
          session->history,
          policy,
          session->metadata,
          session->totalGenerated,
          &firstToken,
          &error)) {
    return fail(session, SDX_STATUS_EXECUTION_FAILED, error);
  }

  const auto stateTransferStart = Clock::now();
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_BEGIN phase=prefill_state_transfer cache_capacity=%d kv_tensors=%zu recurrent_states=%zu",
      session->activeContextCapacity,
      session->metadata.prefillKeyOutputs.size() +
          session->metadata.prefillValueOutputs.size(),
      session->metadata.recurrentStates.size());
  std::vector<std::unique_ptr<NDArray>> fullKv;
  fullKv.reserve(
      session->metadata.prefillKeyOutputs.size() +
      session->metadata.prefillValueOutputs.size());
  for (const auto& name : session->metadata.prefillKeyOutputs) {
    int32_t index = -1;
    if (!outputIndexByName(session->prefillContext, name, &index)) {
      return fail(
          session,
          SDX_STATUS_EXECUTION_FAILED,
          "prefill key output is absent from the compiled graph: " + name);
    }
    std::unique_ptr<NDArray> destination;
    if (!copyPrefillKv(
            sd::dsp::runtime::detail::contextOutputArray(
                session->prefillContext, index),
            session->activeContextCapacity,
            session->metadata.kvDataType,
            &destination,
            &error)) {
      return fail(session, SDX_STATUS_EXECUTION_FAILED, error);
    }
    fullKv.push_back(std::move(destination));
  }
  for (const auto& name : session->metadata.prefillValueOutputs) {
    int32_t index = -1;
    if (!outputIndexByName(session->prefillContext, name, &index)) {
      return fail(
          session,
          SDX_STATUS_EXECUTION_FAILED,
          "prefill value output is absent from the compiled graph: " + name);
    }
    std::unique_ptr<NDArray> destination;
    if (!copyPrefillKv(
            sd::dsp::runtime::detail::contextOutputArray(
                session->prefillContext, index),
            session->activeContextCapacity,
            session->metadata.kvDataType,
            &destination,
            &error)) {
      return fail(session, SDX_STATUS_EXECUTION_FAILED, error);
    }
    fullKv.push_back(std::move(destination));
  }

  std::vector<std::unique_ptr<NDArray>> recurrentStates;
  recurrentStates.reserve(session->metadata.recurrentStates.size());
  for (const auto& state : session->metadata.recurrentStates) {
    int32_t index = -1;
    if (!outputIndexByName(session->prefillContext, state.output, &index)) {
      return fail(
          session,
          SDX_STATUS_EXECUTION_FAILED,
          "prefill recurrent output is absent from the compiled graph: " +
              state.output);
    }
    std::unique_ptr<NDArray> destination;
    if (!copyRecurrentArray(
            sd::dsp::runtime::detail::contextOutputArray(
                session->prefillContext, index),
            state,
            &destination,
            &error)) {
      return fail(session, SDX_STATUS_EXECUTION_FAILED, error);
    }
    recurrentStates.push_back(std::move(destination));
  }
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_END phase=prefill_state_transfer elapsed_us=%llu",
      static_cast<unsigned long long>(
          elapsedNanos(stateTransferStart, Clock::now()) / 1000ULL));

  destroyContext(&session->prefillContext);
  session->prefillPublic.clear();
  session->prefillOwned.clear();

  size_t kv = 0;
  for (const auto& name : session->metadata.kvKeyInputs) {
    if (!addNamed(
            &session->decodeOwned,
            name,
            std::move(fullKv[kv++]),
            &error)) {
      return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
    }
  }
  for (const auto& name : session->metadata.kvValueInputs) {
    if (!addNamed(
            &session->decodeOwned,
            name,
            std::move(fullKv[kv++]),
            &error)) {
      return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
    }
  }
  for (size_t i = 0; i < session->metadata.recurrentStates.size(); ++i) {
    if (!addNamed(
            &session->decodeOwned,
            session->metadata.recurrentStates[i].input,
            std::move(recurrentStates[i]),
            &error)) {
      return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
    }
  }

  session->hasPrompt = true;
  session->promptTokenCount = promptCount;
  session->cachePosition = promptCount;
  commitToken(session, firstToken, callbacks, output, capacity, count);
  session->lastPrefillNanos = elapsedNanos(start, Clock::now());
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_END phase=prefill elapsed_us=%llu emitted_tokens=1",
      static_cast<unsigned long long>(session->lastPrefillNanos / 1000ULL));
  return SDX_STATUS_OK;
}

bool addDecodeInputs(
    sdx_generation_session_t* session,
    std::string* error) {
  const auto& metadata = session->metadata;
  if (!addNamed(
          &session->decodeOwned,
          metadata.inputIds,
          createArray({1, 1}, DataType::INT64),
          error)) {
    return false;
  }
  findNamed(session->decodeOwned, metadata.inputIds)
      ->p(0, session->lastToken);

  auto mask = createArray(
      {1, 1, 1, static_cast<LongType>(session->activeContextCapacity)},
      metadata.maskDataType);
  assignScalar(mask.get(), maskFillValue(metadata.maskDataType));
  for (int32_t i = 0; i <= session->cachePosition; ++i) {
    mask->p(static_cast<LongType>(i), 0.0);
  }

  return addNamed(
             &session->decodeOwned,
             metadata.causalMask,
             std::move(mask),
             error) &&
         addNamed(
             &session->decodeOwned,
             metadata.positionOffset,
             createLongScalar(session->cachePosition),
             error) &&
         addNamed(
             &session->decodeOwned,
             metadata.cachePosition,
             createLongScalar(session->cachePosition),
             error) &&
         addNamed(
             &session->decodeOwned,
             metadata.actualSequenceLength,
             createLongScalar(1),
             error);
}

bool resolveDecodePlanBindings(
    sdx_generation_session_t* session,
    std::string* error) {
  auto planIndex = [&](const std::string& name, int* out) -> bool {
    const int index =
        sd::dsp::runtime::detail::contextPlanInputIndex(
            session->decodeContext, name);
    if (index < 0) {
      if (error != nullptr) {
        *error = "decode plan is missing metadata input: " + name;
      }
      return false;
    }
    *out = index;
    return true;
  };

  if (!planIndex(
          session->metadata.inputIds,
          &session->decodeInputIdsPlanIndex) ||
      !planIndex(
          session->metadata.causalMask,
          &session->decodeCausalMaskPlanIndex) ||
      !planIndex(
          session->metadata.positionOffset,
          &session->decodePositionOffsetPlanIndex) ||
      !planIndex(
          session->metadata.cachePosition,
          &session->decodeCachePositionPlanIndex) ||
      !planIndex(
          session->metadata.actualSequenceLength,
          &session->decodeActualSequenceLengthPlanIndex)) {
    return false;
  }

  session->decodeKvPlanIndices.clear();
  for (const auto& name : session->metadata.kvKeyInputs) {
    int index = -1;
    if (!planIndex(name, &index)) return false;
    session->decodeKvPlanIndices.push_back(index);
  }
  for (const auto& name : session->metadata.kvValueInputs) {
    int index = -1;
    if (!planIndex(name, &index)) return false;
    session->decodeKvPlanIndices.push_back(index);
  }

  session->decodeKvPlanArrays.clear();
  for (int index : session->decodeKvPlanIndices) {
    NDArray* array =
        sd::dsp::runtime::detail::contextPlanInputArray(
            session->decodeContext, index);
    if (array == nullptr) {
      if (error != nullptr) {
        *error = "decode plan did not retain a KV input binding";
      }
      return false;
    }
    session->decodeKvPlanArrays.push_back(array);
  }

  session->decodeGdnStatePlanIndices.clear();
  session->decodeGdnStateOutputIndices.clear();
  session->decodeConvStatePlanIndices.clear();
  session->decodeConvStateOutputIndices.clear();
  for (const auto& state : session->metadata.recurrentStates) {
    int inputIndex = -1;
    int32_t outputIndex = -1;
    if (!planIndex(state.input, &inputIndex)) return false;
    if (!outputIndexByName(
            session->decodeContext, state.output, &outputIndex)) {
      if (error != nullptr) {
        *error = "decode recurrent output is absent from the compiled graph: " +
                 state.output;
      }
      return false;
    }
    if (state.kind == TextGenerationRecurrentStateKind::GDN) {
      session->decodeGdnStatePlanIndices.push_back(inputIndex);
      session->decodeGdnStateOutputIndices.push_back(outputIndex);
    } else {
      session->decodeConvStatePlanIndices.push_back(inputIndex);
      session->decodeConvStateOutputIndices.push_back(outputIndex);
    }
  }

  if (!outputIndexByName(
          session->decodeContext,
          session->metadata.logits,
          &session->decodeLogitsOutputIndex)) {
    if (error != nullptr) {
      *error = "decode logits output is absent from the compiled graph";
    }
    return false;
  }
  return true;
}

sdx_status_t warmDecode(
    sdx_generation_session_t* session,
    const GenerationPolicy& policy,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* output,
    int32_t capacity,
    int32_t* count,
    uint64_t* decodeNanos) {
  const auto start = Clock::now();
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_BEGIN phase=decode_warmup cache_position=%d cache_capacity=%d",
      session->cachePosition,
      session->activeContextCapacity);
  std::string error;
  if (!addDecodeInputs(session, &error)) {
    return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
  }

  std::vector<std::string> outputNames{session->metadata.logits};
  for (const auto& state : session->metadata.recurrentStates) {
    outputNames.push_back(state.output);
  }
  sdx_status_t status = createBoundContext(
      session->model,
      outputNames,
      &session->decodeContext);
  if (status != SDX_STATUS_OK) return status;
  if (!orderPublicInputs(
          session->decodeContext,
          session->decodeOwned,
          &session->decodePublic,
          &error)) {
    return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
  }

  int32_t publicIndex = -1;
  const std::vector<std::string> placeholderNames{
      session->metadata.inputIds,
      session->metadata.causalMask,
      session->metadata.positionOffset,
      session->metadata.cachePosition,
      session->metadata.actualSequenceLength};
  for (const auto& name : placeholderNames) {
    if (!inputIndexByName(session->decodeContext, name, &publicIndex)) {
      return fail(
          session,
          SDX_STATUS_INVALID_ARGUMENT,
          "decode graph is missing public input: " + name);
    }
    status = sdxMarkInputPlaceholder(session->decodeContext, publicIndex);
    if (status != SDX_STATUS_OK) return status;
  }
  for (const auto& name : session->metadata.kvKeyInputs) {
    if (!inputIndexByName(session->decodeContext, name, &publicIndex)) {
      return fail(
          session,
          SDX_STATUS_INVALID_ARGUMENT,
          "decode graph is missing KV input: " + name);
    }
    status = sdxMarkInputVariable(session->decodeContext, publicIndex);
    if (status != SDX_STATUS_OK) return status;
  }
  for (const auto& name : session->metadata.kvValueInputs) {
    if (!inputIndexByName(session->decodeContext, name, &publicIndex)) {
      return fail(
          session,
          SDX_STATUS_INVALID_ARGUMENT,
          "decode graph is missing KV input: " + name);
    }
    status = sdxMarkInputVariable(session->decodeContext, publicIndex);
    if (status != SDX_STATUS_OK) return status;
  }
  for (const auto& state : session->metadata.recurrentStates) {
    if (!inputIndexByName(
            session->decodeContext, state.input, &publicIndex)) {
      return fail(
          session,
          SDX_STATUS_INVALID_ARGUMENT,
          "decode graph is missing recurrent state input: " + state.input);
    }
    status = sdxMarkInputVariable(session->decodeContext, publicIndex);
    if (status != SDX_STATUS_OK) return status;
  }

  const auto slotWarmupStart = Clock::now();
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_BEGIN phase=decode_slot_by_slot_warmup public_inputs=%zu",
      session->decodePublic.size());
  status = sd::dsp::runtime::detail::runOwnedArrays(
      session->decodeContext, session->decodePublic);
  if (status != SDX_STATUS_OK) return status;
  captureExecutionReport(session, session->decodeContext);
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_END phase=decode_slot_by_slot_warmup elapsed_us=%llu plan_phase=%d execution_count=%d",
      static_cast<unsigned long long>(
          elapsedNanos(slotWarmupStart, Clock::now()) / 1000ULL),
      session->lastExecutionReport.plan_phase,
      session->lastExecutionReport.execution_count);

  NDArray* logits =
      sd::dsp::runtime::detail::contextOutputArray(
          session->decodeContext, 0);
  LongType nextToken = -1;
  if (!sampleFromLogits(
          logits,
          0,
          session->history,
          policy,
          session->metadata,
          session->totalGenerated,
          &nextToken,
          &error)) {
    return fail(session, SDX_STATUS_EXECUTION_FAILED, error);
  }

  if (!resolveDecodePlanBindings(session, &error)) {
    return fail(session, SDX_STATUS_EXECUTION_FAILED, error);
  }
  for (const auto& state : session->metadata.recurrentStates) {
    const int inputIndex =
        sd::dsp::runtime::detail::contextPlanInputIndex(
            session->decodeContext, state.input);
    int32_t outputIndex = -1;
    if (inputIndex < 0 ||
        !outputIndexByName(
            session->decodeContext, state.output, &outputIndex)) {
      return fail(
          session,
          SDX_STATUS_EXECUTION_FAILED,
          "failed to resolve recurrent state feedback: " + state.input);
    }
    NDArray* source =
        sd::dsp::runtime::detail::contextOutputArray(
            session->decodeContext, outputIndex);
    NDArray* destination =
        sd::dsp::runtime::detail::contextPlanInputArray(
            session->decodeContext, inputIndex);
    if (!copyRecurrentArrayInto(source, destination, state, &error)) {
      return fail(session, SDX_STATUS_EXECUTION_FAILED, error);
    }
  }
  const auto freezeStart = Clock::now();
  DSP_DIAG(TIMING, "SDX_PHASE_BEGIN phase=decode_freeze_shapes");
  status = sdxFreezeShapes(session->decodeContext);
  if (status != SDX_STATUS_OK) return status;
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_END phase=decode_freeze_shapes elapsed_us=%llu",
      static_cast<unsigned long long>(
          elapsedNanos(freezeStart, Clock::now()) / 1000ULL));

  session->dummyEmbedding =
      createArray({1, 1, 1}, DataType::FLOAT32);
  session->dummyEmbeddingTable =
      createArray({1, 1}, DataType::FLOAT32);
  assignScalar(session->dummyEmbedding.get(), 0.0);
  assignScalar(session->dummyEmbeddingTable.get(), 0.0);

  session->cachePosition++;
  session->decodeReady = true;
  commitToken(session, nextToken, callbacks, output, capacity, count);
  const uint64_t warmupNanos = elapsedNanos(start, Clock::now());
  if (decodeNanos != nullptr) {
    *decodeNanos += warmupNanos;
  }
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_END phase=decode_warmup elapsed_us=%llu emitted_tokens=1",
      static_cast<unsigned long long>(warmupNanos / 1000ULL));
  return SDX_STATUS_OK;
}

struct NativeCallbackState {
  sdx_generation_session_t* session = nullptr;
  const sdx_generation_callbacks_t* callbacks = nullptr;
  int64_t* output = nullptr;
  int32_t capacity = 0;
  int32_t* count = nullptr;
};

void nativeTokenCallback(LongType token, void* userData) {
  auto* state = static_cast<NativeCallbackState*>(userData);
  commitToken(
      state->session,
      token,
      state->callbacks,
      state->output,
      state->capacity,
      state->count);
}

bool nativeCancelCallback(void* userData) {
  auto* state = static_cast<NativeCallbackState*>(userData);
  return cancellationRequested(state->session, state->callbacks);
}

sdx_status_t runNativeDecode(
    sdx_generation_session_t* session,
    int32_t requestedTokens,
    const GenerationPolicy& policy,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* output,
    int32_t capacity,
    int32_t* count,
    uint64_t* decodeNanos) {
  if (requestedTokens <= 0) return SDX_STATUS_OK;
  const int32_t remainingContext =
      session->activeContextCapacity - session->cachePosition;
  const int32_t steps = std::min(requestedTokens, remainingContext);
  if (steps <= 0) return SDX_STATUS_OK;

  NDArray* inputIds =
      sd::dsp::runtime::detail::contextPlanInputArray(
          session->decodeContext, session->decodeInputIdsPlanIndex);
  NDArray* mask =
      sd::dsp::runtime::detail::contextPlanInputArray(
          session->decodeContext, session->decodeCausalMaskPlanIndex);
  NDArray* positionOffset =
      sd::dsp::runtime::detail::contextPlanInputArray(
          session->decodeContext, session->decodePositionOffsetPlanIndex);
  NDArray* cachePosition =
      sd::dsp::runtime::detail::contextPlanInputArray(
          session->decodeContext, session->decodeCachePositionPlanIndex);
  NDArray* actualLength =
      sd::dsp::runtime::detail::contextPlanInputArray(
          session->decodeContext,
          session->decodeActualSequenceLengthPlanIndex);
  if (inputIds == nullptr || mask == nullptr || positionOffset == nullptr ||
      cachePosition == nullptr || actualLength == nullptr) {
    return fail(
        session,
        SDX_STATUS_EXECUTION_FAILED,
        "decode plan lost a mutable input binding");
  }

  inputIds->p(0, session->lastToken);
  positionOffset->p(0, static_cast<LongType>(session->cachePosition));
  cachePosition->p(0, static_cast<LongType>(session->cachePosition));
  actualLength->p(0, static_cast<LongType>(1));
  inputIds->syncToDevice();
  positionOffset->syncToDevice();
  cachePosition->syncToDevice();
  actualLength->syncToDevice();

  auto generated =
      createArray({static_cast<LongType>(steps)}, DataType::INT64);
  auto tokenCount = createArray({1}, DataType::INT64);
  auto timing = createArray({10}, DataType::FLOAT32);
  assignScalar(tokenCount.get(), 0.0);
  assignScalar(timing.get(), 0.0);

  AutoregressiveDecodeConfig config{};
  config.planHandle =
      sd::dsp::runtime::detail::contextPlan(session->decodeContext);
  config.planExternalInputs = nullptr;
  config.numPlanExternalInputs =
      sd::dsp::runtime::detail::contextPlanInputCount(
          session->decodeContext);
  config.planOutputs = nullptr;
  config.numPlanOutputs =
      sd::dsp::runtime::detail::contextOutputCount(
          session->decodeContext);
  config.extInputContext =
      sd::dsp::runtime::detail::contextGraph(session->decodeContext);
  config.embeddingsExtIdx = -1;
  config.maskExtIdx = -1;
  config.causalMaskExtIdx = session->decodeCausalMaskPlanIndex;
  config.posIdsExtIdx = -1;
  config.inputIdsExtIdx = session->decodeInputIdsPlanIndex;
  config.logitsOutputIdx = session->decodeLogitsOutputIndex;
  config.kvInputExtIndices =
      session->decodeKvPlanIndices.empty()
          ? nullptr
          : session->decodeKvPlanIndices.data();
  config.kvOutputIndices = nullptr;
  config.positionOffsetExtIdx = session->decodePositionOffsetPlanIndex;
  config.cachePositionExtIdx = session->decodeCachePositionPlanIndex;
  config.actualSequenceLengthExtIdx =
      session->decodeActualSequenceLengthPlanIndex;
  config.planOwnsKvScatter = true;
  config.gdnStateExtIndices =
      session->decodeGdnStatePlanIndices.empty()
          ? nullptr
          : session->decodeGdnStatePlanIndices.data();
  config.gdnStateOutputIndices =
      session->decodeGdnStateOutputIndices.empty()
          ? nullptr
          : session->decodeGdnStateOutputIndices.data();
  config.numGdnStatePairs =
      static_cast<int>(session->decodeGdnStatePlanIndices.size());
  config.convStateExtIndices =
      session->decodeConvStatePlanIndices.empty()
          ? nullptr
          : session->decodeConvStatePlanIndices.data();
  config.convStateOutputIndices =
      session->decodeConvStateOutputIndices.empty()
          ? nullptr
          : session->decodeConvStateOutputIndices.data();
  config.numConvStatePairs =
      static_cast<int>(session->decodeConvStatePlanIndices.size());
  config.sampleConfig =
      sampleConfig(
          policy, session->metadata, session->totalGenerated);

  NativeCallbackState callbackState{
      session, callbacks, output, capacity, count};
  config.tokenCallback = nativeTokenCallback;
  config.cancelCallback = nativeCancelCallback;
  config.callbackUserData = &callbackState;

  const auto start = Clock::now();
  const int32_t before = *count;
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_BEGIN phase=native_decode steps=%d cache_position=%d cache_capacity=%d",
      steps,
      session->cachePosition,
      session->activeContextCapacity);
  try {
    sd::ops::helpers::autoregressiveDecode(
        session->dummyEmbedding.get(),
        session->dummyEmbeddingTable.get(),
        inputIds,
        mask,
        nullptr,
        session->decodeKvPlanArrays.empty()
            ? nullptr
            : session->decodeKvPlanArrays.data(),
        static_cast<int>(session->metadata.kvKeyInputs.size()),
        generated.get(),
        tokenCount.get(),
        timing.get(),
        steps,
        session->cachePosition,
        session->metadata.eosIds,
        policy.temperature,
        policy.topK,
        policy.topP,
        policy.repetitionPenalty,
        sd::LaunchContext::defaultContext(),
        &config);
  } catch (const std::exception& e) {
    session->cachePosition += *count - before;
    return fail(
        session,
        SDX_STATUS_EXECUTION_FAILED,
        std::string("native autoregressive decode failed: ") + e.what());
  } catch (...) {
    session->cachePosition += *count - before;
    return fail(
        session,
        SDX_STATUS_EXECUTION_FAILED,
        "native autoregressive decode failed");
  }

  tokenCount->syncToHost();
  const int32_t emitted =
      static_cast<int32_t>(tokenCount->e<LongType>(0));
  if (emitted != *count - before) {
    return fail(
        session,
        SDX_STATUS_EXECUTION_FAILED,
        "native decode callback count does not match token count");
  }
  session->cachePosition += emitted;
  const uint64_t nativeDecodeNanos = elapsedNanos(start, Clock::now());
  if (decodeNanos != nullptr) {
    *decodeNanos += nativeDecodeNanos;
  }
  DSP_DIAG(
      TIMING,
      "SDX_PHASE_END phase=native_decode elapsed_us=%llu emitted_tokens=%d",
      static_cast<unsigned long long>(nativeDecodeNanos / 1000ULL),
      emitted);
  return SDX_STATUS_OK;
}

int32_t finishReason(
    sdx_generation_session_t* session,
    const sdx_generation_callbacks_t* callbacks,
    int32_t generatedThisCall,
    int32_t requested) {
  if (session->reachedEos) return SDX_GENERATION_FINISH_EOS;
  if (cancellationRequested(session, callbacks)) {
    return SDX_GENERATION_FINISH_CANCELLED;
  }
  if (session->cachePosition >= session->activeContextCapacity &&
      generatedThisCall < requested) {
    return SDX_GENERATION_FINISH_CONTEXT_LIMIT;
  }
  return generatedThisCall >= requested
             ? SDX_GENERATION_FINISH_MAX_TOKENS
             : SDX_GENERATION_FINISH_NONE;
}

sdx_status_t validateCall(
    sdx_generation_session_t* session,
    const GenerationPolicy& policy,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* output,
    int32_t capacity,
    int32_t* count,
    sdx_generation_report_t* report) {
  if (session == nullptr || count == nullptr || capacity < 0) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  std::string error;
  if (!validateCallbacks(callbacks, &error)) {
    return fail(session, SDX_STATUS_INCOMPATIBLE_ABI, error);
  }
  if (output != nullptr && capacity < policy.maxNewTokens) {
    return fail(
        session,
        SDX_STATUS_INVALID_ARGUMENT,
        "out_capacity must be at least max_new_tokens");
  }
  if (report != nullptr &&
      report->struct_size != 0 &&
      report->struct_size < sizeof(uint32_t)) {
    return fail(
        session,
        SDX_STATUS_INCOMPATIBLE_ABI,
        "sdx_generation_report_t is smaller than this ABI requires");
  }
  *count = 0;
  return SDX_STATUS_OK;
}

sdx_status_t continueLocked(
    sdx_generation_session_t* session,
    const GenerationPolicy& policy,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* output,
    int32_t capacity,
    int32_t* count,
    sdx_generation_report_t* report,
    bool fromGenerate) {
  const auto callStart = Clock::now();
  uint64_t prefillNanos = fromGenerate ? session->lastPrefillNanos : 0;
  uint64_t decodeNanos = 0;

  if (!session->hasPrompt) {
    return fail(
        session,
        SDX_STATUS_INVALID_ARGUMENT,
        "generation continuation requires a prior prompt");
  }

  if (!session->reachedEos &&
      !cancellationRequested(session, callbacks) &&
      *count < policy.maxNewTokens &&
      !session->decodeReady &&
      session->cachePosition < session->activeContextCapacity) {
    sdx_status_t status = warmDecode(
        session,
        policy,
        callbacks,
        output,
        capacity,
        count,
        &decodeNanos);
    if (status != SDX_STATUS_OK) return status;
  }

  if (!session->reachedEos &&
      !cancellationRequested(session, callbacks) &&
      *count < policy.maxNewTokens &&
      session->cachePosition < session->activeContextCapacity) {
    sdx_status_t status = runNativeDecode(
        session,
        policy.maxNewTokens - *count,
        policy,
        callbacks,
        output,
        capacity,
        count,
        &decodeNanos);
    if (status != SDX_STATUS_OK) return status;
  }

  const int32_t reason =
      finishReason(
          session, callbacks, *count, policy.maxNewTokens);
  fillReport(
      session,
      reason,
      *count,
      elapsedNanos(callStart, Clock::now()) + prefillNanos,
      prefillNanos,
      decodeNanos,
      report);
  clearError(session);
  return SDX_STATUS_OK;
}

}  // namespace

extern "C" {

SDX_API sdx_status_t sdxCreateGenerationSession(
    sdx_model_t* model,
    const sdx_generation_session_options_t* options,
    sdx_generation_session_t** outSession) {
  if (model == nullptr || outSession == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  *outSession = nullptr;
  if (options != nullptr &&
      !compatibleStructSize(options->struct_size, sizeof(*options))) {
    sd::dsp::runtime::detail::setModelError(
        model,
        "sdx_generation_session_options_t is smaller than this ABI requires");
    return SDX_STATUS_INCOMPATIBLE_ABI;
  }

  const char* metadataPath = sdxGetTextGenerationConfigPath(model);
  if (metadataPath == nullptr || metadataPath[0] == '\0') {
    sd::dsp::runtime::detail::setModelError(
        model,
        "bundle does not declare a text-generation metadata asset");
    return SDX_STATUS_UNSUPPORTED;
  }

  std::unique_ptr<sdx_generation_session_t> session(
      new (std::nothrow) sdx_generation_session_t());
  if (session == nullptr) {
    sd::dsp::runtime::detail::setModelError(
        model, "failed to allocate generation session");
    return SDX_STATUS_EXECUTION_FAILED;
  }
  session->model = model;

  std::string error;
  if (!sd::dsp::runtime::detail::loadTextGenerationMetadata(
          metadataPath, &session->metadata, &error)) {
    sd::dsp::runtime::detail::setModelError(model, error);
    return SDX_STATUS_UNSUPPORTED;
  }
  sd::dsp::runtime::detail::setModelError(model, "");
  *outSession = session.release();
  return SDX_STATUS_OK;
}

SDX_API void sdxDestroyGenerationSession(
    sdx_generation_session_t* session) {
  if (session == nullptr) return;
  {
    std::lock_guard<std::mutex> lock(session->mutex);
    clearExecutionState(session);
  }
  delete session;
}

SDX_API sdx_status_t sdxResetGenerationSession(
    sdx_generation_session_t* session) {
  if (session == nullptr) return SDX_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lock(session->mutex);
  clearExecutionState(session);
  session->cancelRequested.store(false, std::memory_order_release);
  clearError(session);
  return SDX_STATUS_OK;
}

SDX_API void sdxCancelGeneration(
    sdx_generation_session_t* session) {
  if (session != nullptr) {
    session->cancelRequested.store(true, std::memory_order_release);
  }
}

SDX_API sdx_status_t sdxGenerationGenerate(
    sdx_generation_session_t* session,
    const int64_t* promptTokenIds,
    int32_t numPromptTokens,
    const sdx_generation_options_t* options,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* outTokenIds,
    int32_t outCapacity,
    int32_t* outCount,
    sdx_generation_report_t* outReport) {
  if (session == nullptr || promptTokenIds == nullptr ||
      numPromptTokens <= 0 ||
      numPromptTokens > session->metadata.maxPrefillLength ||
      numPromptTokens >= session->metadata.contextLength) {
    return session == nullptr
               ? SDX_STATUS_INVALID_ARGUMENT
               : fail(
                     session,
                     SDX_STATUS_INVALID_ARGUMENT,
                     "prompt length must fit the fixed prefill and context envelope");
  }

  GenerationPolicy policy;
  std::string error;
  if (!resolvePolicy(session->metadata, options, &policy, &error)) {
    return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
  }

  std::lock_guard<std::mutex> lock(session->mutex);
  sdx_status_t status = validateCall(
      session,
      policy,
      callbacks,
      outTokenIds,
      outCapacity,
      outCount,
      outReport);
  if (status != SDX_STATUS_OK) return status;

  clearExecutionState(session);
  session->cancelRequested.store(false, std::memory_order_release);
  const int32_t reservedNewTokens =
      std::max(policy.maxNewTokens, session->metadata.sampling.maxNewTokens);
  const int64_t requestedCapacity =
      static_cast<int64_t>(numPromptTokens) + reservedNewTokens;
  session->activeContextCapacity = static_cast<int32_t>(std::min<int64_t>(
      session->metadata.contextLength, requestedCapacity));
  DSP_DIAG(
      TIMING,
      "SDX_CACHE_CAPACITY prompt_tokens=%d request_tokens=%d reserved_tokens=%d active=%d model_context=%d",
      numPromptTokens,
      policy.maxNewTokens,
      reservedNewTokens,
      session->activeContextCapacity,
      session->metadata.contextLength);
  session->history.reserve(
      static_cast<size_t>(numPromptTokens + policy.maxNewTokens));
  for (int32_t i = 0; i < numPromptTokens; ++i) {
    session->history.push_back(
        static_cast<LongType>(promptTokenIds[i]));
  }

  const auto callStart = Clock::now();
  status = runPrefill(
      session,
      promptTokenIds,
      numPromptTokens,
      policy,
      callbacks,
      outTokenIds,
      outCapacity,
      outCount);
  if (status != SDX_STATUS_OK) return status;

  if (session->reachedEos ||
      cancellationRequested(session, callbacks) ||
      *outCount >= policy.maxNewTokens) {
    const int32_t reason =
        finishReason(
            session, callbacks, *outCount, policy.maxNewTokens);
    fillReport(
        session,
        reason,
        *outCount,
        elapsedNanos(callStart, Clock::now()),
        session->lastPrefillNanos,
        0,
        outReport);
    clearError(session);
    return SDX_STATUS_OK;
  }

  return continueLocked(
      session,
      policy,
      callbacks,
      outTokenIds,
      outCapacity,
      outCount,
      outReport,
      true);
}

SDX_API sdx_status_t sdxGenerationContinue(
    sdx_generation_session_t* session,
    const sdx_generation_options_t* options,
    const sdx_generation_callbacks_t* callbacks,
    int64_t* outTokenIds,
    int32_t outCapacity,
    int32_t* outCount,
    sdx_generation_report_t* outReport) {
  if (session == nullptr) return SDX_STATUS_INVALID_ARGUMENT;

  GenerationPolicy policy;
  std::string error;
  if (!resolvePolicy(session->metadata, options, &policy, &error)) {
    return fail(session, SDX_STATUS_INVALID_ARGUMENT, error);
  }

  std::lock_guard<std::mutex> lock(session->mutex);
  sdx_status_t status = validateCall(
      session,
      policy,
      callbacks,
      outTokenIds,
      outCapacity,
      outCount,
      outReport);
  if (status != SDX_STATUS_OK) return status;
  session->cancelRequested.store(false, std::memory_order_release);
  return continueLocked(
      session,
      policy,
      callbacks,
      outTokenIds,
      outCapacity,
      outCount,
      outReport,
      false);
}

}  // extern "C"
