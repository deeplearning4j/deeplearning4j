/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_DSP_RUNTIME_DETAIL_SDX_TEXT_GENERATION_METADATA_H
#define LIBND4J_DSP_RUNTIME_DETAIL_SDX_TEXT_GENERATION_METADATA_H

#include <array/DataType.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sd {
namespace dsp {
namespace runtime {
namespace detail {

struct TextGenerationSamplingDefaults {
  int32_t maxNewTokens = 128;
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

/**
 * Strict metadata for the first portable mobile causal-LM profile.
 *
 * Every graph name is bundle-authored. Runtime name guessing is intentionally
 * prohibited because it makes AOT artifact selection and mobile failures
 * nondeterministic.
 */
struct TextGenerationMetadata {
  int32_t formatVersion = 0;
  std::string profile;

  std::string inputIds;
  std::string causalMask;
  std::string positionOffset;
  std::string cachePosition;
  std::string actualSequenceLength;
  std::string logits;
  std::string prefillLogits;
  std::vector<std::string> kvKeyInputs;
  std::vector<std::string> kvValueInputs;
  std::vector<std::string> prefillKeyOutputs;
  std::vector<std::string> prefillValueOutputs;

  std::string kvLayout;
  DataType kvDataType = DataType::FLOAT32;
  DataType maskDataType = DataType::FLOAT32;
  bool planOwnsKvScatter = true;

  int32_t contextLength = 0;
  int32_t maxPrefillLength = 0;
  int32_t maxBatchSize = 1;

  int64_t bosId = -1;
  int64_t padId = -1;
  int64_t unkId = -1;
  std::vector<int> eosIds;

  TextGenerationSamplingDefaults sampling;
  std::string chatTemplatePath;
};

bool loadTextGenerationMetadata(
    const std::string& path,
    TextGenerationMetadata* out,
    std::string* error);

}  // namespace detail
}  // namespace runtime
}  // namespace dsp
}  // namespace sd

#endif  // LIBND4J_DSP_RUNTIME_DETAIL_SDX_TEXT_GENERATION_METADATA_H
