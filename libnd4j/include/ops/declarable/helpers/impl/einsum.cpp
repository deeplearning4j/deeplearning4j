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

#include <ops/declarable/helpers/einsum.h>
#include <ops/declarable/DeclarableOp.h>

#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// ==================== Equation Parsing ====================

struct EinsumEquation {
  std::vector<std::string> inputSubs;
  std::string outputSub;
  bool hasExplicitOutput;
};

static EinsumEquation parseEquation(const std::string& equation) {
  EinsumEquation eq;

  std::string normalized;
  for (char c : equation) {
    if (c != ' ') normalized += c;
  }

  auto arrowPos = normalized.find("->");
  std::string inputPart;
  if (arrowPos != std::string::npos) {
    inputPart = normalized.substr(0, arrowPos);
    eq.outputSub = normalized.substr(arrowPos + 2);
    eq.hasExplicitOutput = true;
  } else {
    inputPart = normalized;
    eq.hasExplicitOutput = false;
  }

  std::istringstream iss(inputPart);
  std::string token;
  while (std::getline(iss, token, ',')) {
    eq.inputSubs.push_back(token);
  }

  if (!eq.hasExplicitOutput) {
    std::map<char, int> labelCount;
    for (auto& sub : eq.inputSubs) {
      for (char c : sub) {
        labelCount[c]++;
      }
    }
    for (auto& kv : labelCount) {
      if (kv.second == 1) {
        eq.outputSub += kv.first;
      }
    }
    eq.hasExplicitOutput = true;
  }

  return eq;
}

// ==================== Shape Computation ====================

std::vector<LongType> einsumOutputShape(const std::string& equation,
                                         const std::vector<const LongType*>& inputShapeInfos) {
  auto eq = parseEquation(equation);

  if (eq.inputSubs.size() != inputShapeInfos.size())
    THROW_EXCEPTION("EINSUM: equation input count does not match number of arrays provided");

  std::map<char, LongType> labelSizes;
  for (size_t i = 0; i < eq.inputSubs.size(); i++) {
    const auto& sub = eq.inputSubs[i];
    int rank = shape::rank(inputShapeInfos[i]);
    if ((int)sub.size() != rank)
      THROW_EXCEPTION("EINSUM: input rank does not match subscript length");

    for (size_t j = 0; j < sub.size(); j++) {
      char label = sub[j];
      LongType dimSize = shape::shapeOf(inputShapeInfos[i])[j];
      if (labelSizes.count(label)) {
        if (labelSizes[label] != dimSize)
          THROW_EXCEPTION("EINSUM: label has inconsistent dimension sizes across inputs");
      } else {
        labelSizes[label] = dimSize;
      }
    }
  }

  std::vector<LongType> outShape;
  for (char c : eq.outputSub) {
    if (!labelSizes.count(c))
      THROW_EXCEPTION("EINSUM: output label not found in any input");
    outShape.push_back(labelSizes[c]);
  }

  return outShape;
}

// ==================== Single-Input Einsum ====================

static void einsumSingleInput(LaunchContext* context, const std::string& inputSub,
                               const std::string& outputSub, NDArray& input, NDArray& output) {
  std::map<char, std::vector<int>> labelPositions;
  for (int i = 0; i < (int)inputSub.size(); i++) {
    labelPositions[inputSub[i]].push_back(i);
  }

  bool hasRepeatedLabels = false;
  for (auto& kv : labelPositions) {
    if (kv.second.size() > 1) {
      hasRepeatedLabels = true;
      break;
    }
  }

  if (hasRepeatedLabels) {
    char repeatedLabel = 0;
    for (auto& kv : labelPositions) {
      if (kv.second.size() > 1) {
        repeatedLabel = kv.first;
        break;
      }
    }

    auto& positions = labelPositions[repeatedLabel];
    if (positions.size() != 2)
      THROW_EXCEPTION("EINSUM: only pairs of repeated labels supported");

    if (inputSub.size() == 2 && outputSub.empty()) {
      // "ii->" trace: sum of diagonal elements
      LongType n = input.sizeAt(0);
      double sum = 0;
      for (LongType i = 0; i < n; i++) {
        sum += input.e<double>(i, i);
      }
      output.p<double>(0, sum);
    } else if (inputSub.size() == 2 && outputSub.size() == 1) {
      // "ii->i" diagonal extraction
      LongType n = input.sizeAt(0);
      for (LongType i = 0; i < n; i++) {
        output.p<double>(i, input.e<double>(i, i));
      }
    } else {
      THROW_EXCEPTION("EINSUM: unsupported repeated-label pattern");
    }
    return;
  }

  // No repeated labels: permute + optional reduce (sum)
  std::set<char> outputLabels(outputSub.begin(), outputSub.end());
  std::vector<int> sumAxes;
  std::string remainingLabels;
  for (int i = 0; i < (int)inputSub.size(); i++) {
    if (outputLabels.count(inputSub[i]) == 0) {
      sumAxes.push_back(i);
    } else {
      remainingLabels += inputSub[i];
    }
  }

  NDArray* current = &input;
  NDArray* reduced = nullptr;

  if (!sumAxes.empty()) {
    std::vector<LongType> sumAxesLong(sumAxes.begin(), sumAxes.end());
    reduced = current->reduceAlongDimension(reduce::Sum, &sumAxesLong, false);
    current = reduced;
  } else {
    remainingLabels = inputSub;
  }

  if (remainingLabels != outputSub && !outputSub.empty()) {
    std::vector<LongType> permutation;
    for (char c : outputSub) {
      auto pos = remainingLabels.find(c);
      if (pos == std::string::npos)
        THROW_EXCEPTION("EINSUM: output label not found after reduction");
      permutation.push_back((LongType)pos);
    }
    auto permuted = current->permute(permutation, false, false);
    output.assign(permuted);
    delete permuted;
  } else {
    output.assign(current);
  }

  if (reduced != nullptr) delete reduced;
}

// ==================== Two-Input Einsum ====================

static void einsumTwoInputs(LaunchContext* context, const std::string& subA, const std::string& subB,
                             const std::string& subOut, NDArray& A, NDArray& B, NDArray& output) {
  // Build label-to-size map
  std::map<char, LongType> labelSize;
  for (int i = 0; i < (int)subA.size(); i++) labelSize[subA[i]] = A.sizeAt(i);
  for (int i = 0; i < (int)subB.size(); i++) labelSize[subB[i]] = B.sizeAt(i);

  std::set<char> aSet(subA.begin(), subA.end());
  std::set<char> bSet(subB.begin(), subB.end());
  std::set<char> outSet(subOut.begin(), subOut.end());

  // Classify labels: batch (in A,B,output), contract (in A,B, not output),
  // freeA (in A and output, not B), freeB (in B and output, not A)
  std::vector<char> batchLabels, contractLabels, freeALabels, freeBLabels;
  std::set<char> classified;

  for (char c : subA) {
    if (classified.count(c)) continue;
    classified.insert(c);
    if (bSet.count(c) && outSet.count(c)) batchLabels.push_back(c);
    else if (bSet.count(c) && !outSet.count(c)) contractLabels.push_back(c);
    else if (!bSet.count(c) && outSet.count(c)) freeALabels.push_back(c);
  }
  for (char c : subB) {
    if (classified.count(c)) continue;
    classified.insert(c);
    if (!aSet.count(c) && outSet.count(c)) freeBLabels.push_back(c);
  }

  // Build permutations: A → [batch, freeA, contract], B → [batch, contract, freeB]
  std::vector<LongType> permA, permB;
  for (char c : batchLabels) permA.push_back((LongType)subA.find(c));
  for (char c : freeALabels) permA.push_back((LongType)subA.find(c));
  for (char c : contractLabels) permA.push_back((LongType)subA.find(c));

  for (char c : batchLabels) permB.push_back((LongType)subB.find(c));
  for (char c : contractLabels) permB.push_back((LongType)subB.find(c));
  for (char c : freeBLabels) permB.push_back((LongType)subB.find(c));

  // Apply permutations
  bool identityA = true, identityB = true;
  for (int i = 0; i < (int)permA.size(); i++) if (permA[i] != i) { identityA = false; break; }
  for (int i = 0; i < (int)permB.size(); i++) if (permB[i] != i) { identityB = false; break; }

  NDArray *workA, *workB;
  NDArray *toDeleteA = nullptr, *toDeleteB = nullptr;
  if (identityA) {
    workA = &A;
  } else {
    toDeleteA = A.permute(permA, false, false);
    workA = new NDArray(toDeleteA->dup('c'));
    delete toDeleteA;
    toDeleteA = workA;
  }
  if (identityB) {
    workB = &B;
  } else {
    toDeleteB = B.permute(permB, false, false);
    workB = new NDArray(toDeleteB->dup('c'));
    delete toDeleteB;
    toDeleteB = workB;
  }

  // Compute product sizes
  LongType batchProd = 1, freeAProd = 1, freeBProd = 1, contractProd = 1;
  for (char c : batchLabels) batchProd *= labelSize[c];
  for (char c : freeALabels) freeAProd *= labelSize[c];
  for (char c : freeBLabels) freeBProd *= labelSize[c];
  for (char c : contractLabels) contractProd *= labelSize[c];

  // Reshape for matmul
  bool hasBatch = !batchLabels.empty();
  NDArray *reshA, *reshB;
  std::vector<LongType> shapeA, shapeB;
  if (hasBatch) {
    shapeA = {batchProd, freeAProd, contractProd};
    shapeB = {batchProd, contractProd, freeBProd};
  } else {
    shapeA = {freeAProd, contractProd};
    shapeB = {contractProd, freeBProd};
  }
  reshA = workA->reshape('c', shapeA);
  reshB = workB->reshape('c', shapeB);

  // Matrix multiply (MmulHelper::mmul handles both 2D and batched 3D)
  auto* mmulResult = MmulHelper::mmul(reshA, reshB);

  delete reshA;
  delete reshB;
  if (toDeleteA) delete toDeleteA;
  if (toDeleteB) delete toDeleteB;

  // Build result label order: [batch, freeA, freeB]
  std::vector<LongType> resultFullShape;
  std::string resultLabels;
  for (char c : batchLabels) { resultFullShape.push_back(labelSize[c]); resultLabels += c; }
  for (char c : freeALabels) { resultFullShape.push_back(labelSize[c]); resultLabels += c; }
  for (char c : freeBLabels) { resultFullShape.push_back(labelSize[c]); resultLabels += c; }

  if (resultFullShape.empty()) {
    // Scalar output (e.g., dot product "i,i->")
    output.p(0, mmulResult->e<double>(0));
  } else {
    auto* reshapedResult = mmulResult->reshape('c', resultFullShape);
    if (resultLabels == subOut) {
      output.assign(reshapedResult);
    } else {
      // Permute to output order
      std::vector<LongType> outPerm;
      for (char c : subOut) {
        outPerm.push_back((LongType)resultLabels.find(c));
      }
      auto* finalResult = reshapedResult->permute(outPerm, false, false);
      output.assign(finalResult);
      delete finalResult;
    }
    delete reshapedResult;
  }

  delete mmulResult;
}

// ==================== Main Entry Point ====================

void einsum(LaunchContext* context, const std::string& equation,
            const std::vector<NDArray*>& inputs, NDArray& output) {
  auto eq = parseEquation(equation);

  if (eq.inputSubs.size() != inputs.size())
    THROW_EXCEPTION("EINSUM: equation input count does not match number of arrays provided");
  if (inputs.empty())
    THROW_EXCEPTION("EINSUM: at least one input required");

  if (inputs.size() == 1) {
    einsumSingleInput(context, eq.inputSubs[0], eq.outputSub, *inputs[0], output);
  } else if (inputs.size() == 2) {
    einsumTwoInputs(context, eq.inputSubs[0], eq.inputSubs[1], eq.outputSub,
                    *inputs[0], *inputs[1], output);
  } else {
    // N-input: pairwise left-to-right reduction
    std::string subA = eq.inputSubs[0];
    std::string subB = eq.inputSubs[1];

    std::set<char> neededLabels;
    for (char c : eq.outputSub) neededLabels.insert(c);
    for (size_t i = 2; i < eq.inputSubs.size(); i++) {
      for (char c : eq.inputSubs[i]) neededLabels.insert(c);
    }

    std::string intermediateSub;
    std::set<char> seen;
    for (char c : subA) {
      if (neededLabels.count(c) && !seen.count(c)) {
        intermediateSub += c;
        seen.insert(c);
      }
    }
    for (char c : subB) {
      if (neededLabels.count(c) && !seen.count(c)) {
        intermediateSub += c;
        seen.insert(c);
      }
    }

    std::map<char, LongType> labelSizes;
    for (size_t i = 0; i < eq.inputSubs.size(); i++) {
      for (size_t j = 0; j < eq.inputSubs[i].size(); j++) {
        labelSizes[eq.inputSubs[i][j]] = inputs[i]->sizeAt((int)j);
      }
    }

    std::vector<LongType> intermediateShape;
    for (char c : intermediateSub) {
      intermediateShape.push_back(labelSizes[c]);
    }

    auto intermediate = NDArrayFactory::create_('c', intermediateShape, inputs[0]->dataType(), context);
    einsumTwoInputs(context, subA, subB, intermediateSub, *inputs[0], *inputs[1], *intermediate);

    for (size_t i = 2; i < inputs.size(); i++) {
      subA = intermediateSub;
      subB = eq.inputSubs[i];

      if (i == inputs.size() - 1) {
        einsumTwoInputs(context, subA, subB, eq.outputSub, *intermediate, *inputs[i], output);
        delete intermediate;
      } else {
        neededLabels.clear();
        for (char c : eq.outputSub) neededLabels.insert(c);
        for (size_t k = i + 1; k < eq.inputSubs.size(); k++) {
          for (char c : eq.inputSubs[k]) neededLabels.insert(c);
        }

        intermediateSub.clear();
        seen.clear();
        for (char c : subA) {
          if (neededLabels.count(c) && !seen.count(c)) {
            intermediateSub += c;
            seen.insert(c);
          }
        }
        for (char c : subB) {
          if (neededLabels.count(c) && !seen.count(c)) {
            intermediateSub += c;
            seen.insert(c);
          }
        }

        intermediateShape.clear();
        for (char c : intermediateSub) {
          intermediateShape.push_back(labelSizes[c]);
        }

        auto newIntermediate = NDArrayFactory::create_('c', intermediateShape, inputs[0]->dataType(), context);
        einsumTwoInputs(context, subA, subB, intermediateSub, *intermediate, *inputs[i], *newIntermediate);
        delete intermediate;
        intermediate = newIntermediate;
      }
    }
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
