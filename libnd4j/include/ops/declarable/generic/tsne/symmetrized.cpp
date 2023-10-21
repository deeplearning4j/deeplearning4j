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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_barnes_symmetrized)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace sd {
namespace ops {
NDArray* rowCountsPtr = nullptr;

CUSTOM_OP_IMPL(barnes_symmetrized, 3, 3, false, 0, -1) {
  auto rowP = INPUT_VARIABLE(0);
  auto colP = INPUT_VARIABLE(1);
  auto valP = INPUT_VARIABLE(2);
  auto N = rowP->lengthOf() - 1;
  auto outputRows = OUTPUT_VARIABLE(0);
  auto outputCols = OUTPUT_VARIABLE(1);
  auto outputVals = OUTPUT_VARIABLE(2);

  if (block.getIArguments()->size() > 0) N = INT_ARG(0);

  if (rowCountsPtr) {
    helpers::barnes_symmetrize(rowP, colP, valP, N, outputRows, outputCols, outputVals, rowCountsPtr);
    delete rowCountsPtr;
    return sd::Status::OK;
  }
  return Logger::logKernelFailureMsg("barnes_symmetrized: Cannot loop due wrong input data.");
}

DECLARE_TYPES(barnes_symmetrized) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {DataType::INT32})
      ->setAllowedInputTypes(1, {DataType::INT32})
      ->setAllowedInputTypes(2, {ALL_INTS, ALL_FLOATS})
      ->setAllowedOutputTypes(1, {DataType::INT32})
      ->setAllowedOutputTypes(1, {DataType::INT32})
      ->setAllowedOutputTypes(2, {ALL_INTS, ALL_FLOATS})
      ->setSameMode(false);
}

DECLARE_SHAPE_FN(barnes_symmetrized) {
  auto valPShapeInfo = inputShape->at(2);
  sd::LongType* outShapeInfo;
  auto rowP = INPUT_VARIABLE(0);
  auto colP = INPUT_VARIABLE(1);
  auto N = rowP->lengthOf() - 1;
  if (block.getIArguments()->size() > 0) N = INT_ARG(0);
  auto dataType = rowP->dataType();  // ArrayOptions::dataType(inputShape->at(0));
  NDArray* rowCounts = NDArrayFactory::create_<int>('c', {N}, block.launchContext());  // rowP->dup();
  sd::LongType len = helpers::barnes_row_count(rowP, colP, N, *rowCounts);
  rowCounts->syncToHost();
  if (len <= 0) THROW_EXCEPTION("barnes_symmetrized: Cannot allocate shape due non-positive len.");
  rowCountsPtr = rowCounts;
   outShapeInfo =
      sd::ShapeBuilders::createShapeInfo(ArrayOptions::dataType(valPShapeInfo), 'c', {1, len}, block.getWorkspace());
  auto outColsShapeInfo = sd::ShapeBuilders::createShapeInfo(dataType, 'c', {1, len}, block.getWorkspace());
  auto outRowsShapeInfo = sd::ShapeBuilders::createShapeInfo(dataType, 'c', {1, N + 1}, block.getWorkspace());
  return SHAPELIST(CONSTANT(outRowsShapeInfo), CONSTANT(outColsShapeInfo), CONSTANT(outShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
