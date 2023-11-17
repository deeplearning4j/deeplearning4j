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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_skipgram)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/sg_cb.h>

namespace sd {
namespace ops {


CONFIGURABLE_OP_IMPL(skipgram_inference, 6, 6, true, -2, -2) {
  //construct codes and indices from the IARGS
  //we do this to avoid serialization overhead from the JVM for frequently created small arrays
  auto numCodes = I_ARG(0);
  auto numIndices = I_ARG(1);
  auto numIterations = I_ARG(2);
  //2 for the number of indices/codes 1 for the iteration 3 for the mandatory args
  auto numMin = numIndices + numCodes + 2  + 1 + 3;
  std::vector<LongType> *codes = new std::vector<LongType>();
  std::vector<LongType> *indices = new std::vector<LongType>();

  int currIdx = 3;
  for(int i = 0; i < numCodes; i++) {
    codes->push_back(I_ARG(currIdx));
    currIdx++;
  }

  for(int i = 0; i < numIndices; i++) {
    indices->push_back(I_ARG(currIdx));
    currIdx++;
  }

  const std::vector<LongType> *indicesVec = indices;
  const std::vector<LongType> *codesVec = codes;

  std::vector<LongType> *indicesSize = new std::vector<LongType>();
  indicesSize->push_back(indices->size());
  const std::vector<LongType> *indicesShape = indicesSize;


  std::vector<LongType> *codesSize = new std::vector<LongType>();
  codesSize->push_back(codes->size());
  const std::vector<LongType> *codesShape = codesSize;


  auto indicesArrOne = NDArrayFactory::create('c',*indicesShape,*indicesVec);
  auto indicesArr = new NDArray(indicesArrOne);
  auto codesArrOne = NDArrayFactory::create('c',*codesShape,*codesVec);
  auto codesArr = new NDArray(codesArrOne);


  auto target = I_ARG(currIdx++);
  auto ngStarter = I_ARG(currIdx++);
  auto randomValue = I_ARG(currIdx++);
  auto numWorkers = block.numI() > numMin ? INT_ARG(currIdx++) : omp_get_max_threads();
  auto nsRounds = block.numI() > numMin + 1 ? INT_ARG(currIdx++) : 0;

  auto alpha = T_ARG(0);

  // required part


  auto syn0 = INPUT_VARIABLE(0);
  auto syn1 = INPUT_VARIABLE(1);
  auto syn1neg = INPUT_VARIABLE(2);

  auto expTable = INPUT_VARIABLE(3);
  auto negTable = INPUT_VARIABLE(4);


  auto inferenceVector = INPUT_VARIABLE(5);





  auto isInference = block.numB() > 0 ? B_ARG(0) : false;
  auto isPreciseMode = block.numB() > 1 ? B_ARG(1) : false;

  REQUIRE_TRUE(block.isInplace(), 0, "SkipGram: this operation requires inplace execution only");

  REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0,
               "SkipGram: all syn tables must have the same data type");
  REQUIRE_TRUE(syn0->dataType() == expTable->dataType(), 0,
               "SkipGram: expTable must have the same data type as syn0 table");

  helpers::skipgramInference(*syn0,
                                      *syn1,
                                      *syn1neg,
                                      *expTable,
                                      *negTable,
                                      target,
                                      ngStarter,
                                      nsRounds,
                                      *indicesArr,
                                      *codesArr,
                                      alpha,
                                      randomValue,
                                      *inferenceVector,
                                      isPreciseMode,
                                      numWorkers,1e-4,numIterations);


  delete codes;
  delete indices;
  delete indicesArr;
  delete codesArr;
  delete indicesSize;
  delete codesSize;


  return Status::OK;
}


DECLARE_TYPES(skipgram_inference) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedOutputTypes(ANY);
}


CONFIGURABLE_OP_IMPL(skipgram, 12, 12, true, 0, 0) {
  auto target = INPUT_VARIABLE(0);
  auto ngStarter = INPUT_VARIABLE(1);

  // required part
  auto indices = INPUT_VARIABLE(2);
  auto codes = INPUT_VARIABLE(3);
  auto syn0 = INPUT_VARIABLE(4);
  auto syn1 = INPUT_VARIABLE(5);
  auto syn1neg = INPUT_VARIABLE(6);

  auto expTable = INPUT_VARIABLE(7);
  auto negTable = INPUT_VARIABLE(8);

  auto alpha = INPUT_VARIABLE(9);
  auto randomValue = INPUT_VARIABLE(10);

  auto inferenceVector = INPUT_VARIABLE(11);


  auto numWorkers = block.numI() > 0 ? INT_ARG(0) : omp_get_max_threads();
  auto nsRounds = block.numI() > 1 ? INT_ARG(1) : 0;
  auto iterations = block.numI() > 2  && inferenceVector != nullptr ? INT_ARG(2) : 1;

  auto isInference = block.numB() > 0 ? B_ARG(0) : false;
  auto isPreciseMode = block.numB() > 1 ? B_ARG(1) : false;

  auto minLearningRate = block.numT() > 0 ? T_ARG(0) : 1e-4;


  REQUIRE_TRUE(block.isInplace(), 0, "SkipGram: this operation requires inplace execution only");

  REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0,
               "SkipGram: all syn tables must have the same data type");
  REQUIRE_TRUE(syn0->dataType() == expTable->dataType(), 0,
               "SkipGram: expTable must have the same data type as syn0 table");

  helpers::skipgram(*syn0, *syn1, *syn1neg, *expTable, *negTable, *target, *ngStarter, nsRounds, *indices,
                             *codes, *alpha, *randomValue, *inferenceVector, isPreciseMode, numWorkers,iterations,minLearningRate);

  return Status::OK;
}

DECLARE_TYPES(skipgram) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, INT32)
      ->setAllowedInputTypes(1, INT32)
      ->setAllowedInputTypes(2, INT32)
      ->setAllowedInputTypes(3, {ALL_INTS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedInputTypes(6, {ALL_FLOATS})
      ->setAllowedInputTypes(7, {ALL_FLOATS})
      ->setAllowedInputTypes(8, {ALL_FLOATS})
      ->setAllowedInputTypes(9, {ALL_FLOATS})
      ->setAllowedInputTypes(10, INT64)
      ->setAllowedInputTypes(11, {ALL_FLOATS})
      ->setAllowedOutputTypes(ANY);
}


}  // namespace ops
}  // namespace sd

#endif
