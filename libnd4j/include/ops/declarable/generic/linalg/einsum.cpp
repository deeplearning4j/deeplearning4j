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

//
// Created by GS <sgazeos@gmail.com> at 2/26/2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/blas.h>
#include <system/op_boilerplate.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <cctype>

#if NOT_EXCLUDED(OP_einsum)
namespace sd {
namespace ops {
DECLARE_TYPES(einsum) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}


std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

void parse_einsum_input(const std::string& equation, std::vector<std::string>& input_labels, std::string& output_labels) {
  auto parts = split(equation, '-');
  input_labels = split(parts[0], ',');

  if (parts.size() == 2) {
    output_labels = split(parts[1], '>')[1];
  } else {
    std::unordered_map<char, int> count;
    for (const auto& labels : input_labels) {
      for (char label : labels) {
        if (std::isalpha(label)) {
          count[label]++;
        } else {
          throw std::runtime_error("Invalid character in input labels.");
        }
      }
    }
    for (const auto& entry : count) {
      if (entry.second == 1) {
        output_labels += entry.first;
      }
    }
  }
}

CUSTOM_OP_IMPL(einsum, -2, 1, false, 0, 0) {
  // Get the einsum equation string
  std::string equation = block.getSArguments()->at(0);

  // Parse the einsum equation string and extract relevant information
  std::vector<std::string> input_labels;
  std::string output_labels;
  parse_einsum_input(equation, input_labels, output_labels);

  // Perform the einsum operation based on the parsed information
  std::vector<std::vector<sd::LongType>> input_axes(input_labels.size());
  std::vector<sd::LongType> output_axes;

  for (size_t i = 0; i < input_labels.size(); ++i) {
    for (char label : input_labels[i]) {
      auto pos = output_labels.find(label);
      if (pos != std::string::npos) {
        input_axes[i].push_back(pos);
      } else {
        output_labels += label;
        input_axes[i].push_back(output_labels.size() - 1);
      }
    }
  }

  // Optimize the computation using inplace sum operations
  auto output = OUTPUT_VARIABLE(0);
  sd::ops::reduce_sum reduceSumOp;
  sd::ops::multiply multiplyOp;
  sd::ops::matmul matMulOp;

  for (size_t i = 0; i < block.width(); ++i) {
    auto inputVar = INPUT_VARIABLE(i);

    // Create an NDArray from input_axes[i] using ConstantShapeHelper
    sd::ShapeDescriptor shapeDescriptor(sd::DataType::INT64, 'c', input_axes[i]);
    auto input_axes_vector = sd::ConstantShapeHelper::getInstance().createShapeInfo(&shapeDescriptor);

    // Optimize for matrix multiplication case
    if (inputVar->rankOf() == 2 && output->rankOf() == 2 &&
        input_axes[i].size() == 1 && input_axes[i][0] == 1) {
      if (i == 0) {
        output->assign(inputVar);
      } else {
        matMulOp.execute({output, inputVar}, {output}, {},{});
      }
    } else {
      if (i == 0) {
        reduceSumOp.execute({inputVar}, {output}, {input_axes_vector});
      } else {
        sd::ResultSet reducedResult = reduceSumOp.evaluate({inputVar}, {input_axes_vector});
        multiplyOp.execute({output, reducedResult.at(0)}, {output}, {},{});
      }
    }
  }

  // Transpose the result to match the output_labels order
  std::vector<sd::LongType> transpose_order(output_labels.size());
  for (size_t i = 0; i < output_labels.size(); ++i) {
    transpose_order[i] = std::distance(output_labels.begin(), std::find(output_labels.begin(), output_labels.end(), 'a' + i));
  }
  output->permutei(transpose_order);

  return Status::OK;
}
}

DECLARE_SHAPE_FN(einsum) {
  // Calculate the output shape based on the einsum equation string and input shapes
  std::string equation = block.getSArguments()->at(0);
  std::vector<std::string> input_labels;
  std::string output_labels;
  parse_einsum_input(equation, input_labels, output_labels);

  std::vector<sd::LongType> output_shape(output_labels.size(), -1);

  for (size_t i = 0; i < input_labels.size(); ++i) {
    for (size_t j = 0; j < input_labels[i].size(); ++j) {
      char label = input_labels[i][j];
      size_t pos = output_labels.find(label);
      if (output_shape[pos] == -1 || output_shape[pos] == inputShape->at(i)[j]) {
        output_shape[pos] = inputShape->at(i)[j];
      } else {
        throw std::runtime_error("Incompatible shapes for einsum operation.");
      }
    }
  }

  auto outputDataType = ArrayOptions::dataType(inputShape->at(0));
  sd::ShapeDescriptor descriptor(outputDataType, 'c', output_shape);
  auto outputShapeInfo = sd::ConstantShapeHelper::getInstance().createShapeInfo(&descriptor);
  return SHAPELIST(outputShapeInfo);
}
}  // namespace ops
  // namespace sd
#endif
