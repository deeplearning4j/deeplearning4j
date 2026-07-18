/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_HELPERS_EINSUM_SHAPE_H
#define LIBND4J_HELPERS_EINSUM_SHAPE_H

#include <helpers/shape.h>
#include <system/op_boilerplate.h>

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

struct EinsumEquation {
  std::vector<std::string> inputSubs;
  std::string outputSub;
  bool hasExplicitOutput = false;
};

SD_INLINE EinsumEquation parseEinsumEquation(const std::string& equation) {
  EinsumEquation parsed;

  std::string normalized;
  for (char character : equation) {
    if (character != ' ') normalized += character;
  }

  const auto arrowPosition = normalized.find("->");
  std::string inputPart;
  if (arrowPosition != std::string::npos) {
    inputPart = normalized.substr(0, arrowPosition);
    parsed.outputSub = normalized.substr(arrowPosition + 2);
    parsed.hasExplicitOutput = true;
  } else {
    inputPart = normalized;
  }

  std::istringstream inputStream(inputPart);
  std::string token;
  while (std::getline(inputStream, token, ',')) {
    parsed.inputSubs.push_back(token);
  }

  if (!parsed.hasExplicitOutput) {
    std::map<char, int> labelCounts;
    for (const auto& subscript : parsed.inputSubs) {
      for (char label : subscript) {
        ++labelCounts[label];
      }
    }
    for (const auto& entry : labelCounts) {
      if (entry.second == 1) parsed.outputSub += entry.first;
    }
    parsed.hasExplicitOutput = true;
  }

  return parsed;
}

SD_INLINE std::vector<LongType> einsumOutputShape(
    const std::string& equation,
    const std::vector<const LongType*>& inputShapeInfos) {
  const auto parsed = parseEinsumEquation(equation);
  if (parsed.inputSubs.size() != inputShapeInfos.size()) {
    THROW_EXCEPTION(
        "EINSUM: equation input count does not match number of arrays provided");
  }

  std::map<char, LongType> labelSizes;
  for (size_t inputIndex = 0; inputIndex < parsed.inputSubs.size();
       ++inputIndex) {
    const auto& subscript = parsed.inputSubs[inputIndex];
    const int rank = shape::rank(inputShapeInfos[inputIndex]);
    if (static_cast<int>(subscript.size()) != rank) {
      THROW_EXCEPTION("EINSUM: input rank does not match subscript length");
    }

    const auto* dimensions = shape::shapeOf(inputShapeInfos[inputIndex]);
    for (size_t dimension = 0; dimension < subscript.size(); ++dimension) {
      const char label = subscript[dimension];
      const LongType dimensionSize = dimensions[dimension];
      const auto existing = labelSizes.find(label);
      if (existing != labelSizes.end()) {
        if (existing->second != dimensionSize) {
          THROW_EXCEPTION(
              "EINSUM: label has inconsistent dimension sizes across inputs");
        }
      } else {
        labelSizes.emplace(label, dimensionSize);
      }
    }
  }

  std::vector<LongType> outputShape;
  for (char label : parsed.outputSub) {
    const auto dimension = labelSizes.find(label);
    if (dimension == labelSizes.end()) {
      THROW_EXCEPTION("EINSUM: output label not found in any input");
    }
    outputShape.push_back(dimension->second);
  }

  return outputShape;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_EINSUM_SHAPE_H
