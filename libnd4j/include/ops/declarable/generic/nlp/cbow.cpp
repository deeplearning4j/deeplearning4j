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
#include <array/NDArrayFactory.h>
#if NOT_EXCLUDED(OP_cbow)

#include <ops/declarable/headers/nlp.h>
#include <ops/declarable/helpers/sg_cb.h>

#include <vector>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(cbow_inference, 6, 6, true, -2, -2) {
  // The codes, indices, context and locked words travel in the IARGS, to avoid serialization overhead from the JVM
  // for frequently created small arrays. The layout is the one CbowInference writes:
  //   [numCodes, numIndices, numContext, numLockedWords, iterations, numWorkers, nsRounds,
  //    codes..., indices..., context..., lockedWords..., target, ngStarter, numLabels, randomValue, ...]
  REQUIRE_TRUE(block.numI() >= 11, 0, "CBOW inference: at least 11 integer arguments are required, but got %i",
               static_cast<int>(block.numI()));
  REQUIRE_TRUE(block.numT() > 0, 0, "CBOW inference: the learning rate is the first float argument");

  const sd::LongType numCodes = I_ARG(0);
  const sd::LongType numIndices = I_ARG(1);
  const sd::LongType numContext = I_ARG(2);
  const sd::LongType numLockedWords = I_ARG(3);
  REQUIRE_TRUE(numCodes >= 0 && numIndices >= 0 && numContext >= 0 && numLockedWords >= 0 &&
                   static_cast<sd::LongType>(block.numI()) >= 11 + numCodes + numIndices + numContext + numLockedWords,
               0,
               "CBOW inference: the integer arguments do not hold %lld codes, %lld indices, %lld context words and "
               "%lld locked words",
               static_cast<long long>(numCodes), static_cast<long long>(numIndices),
               static_cast<long long>(numContext), static_cast<long long>(numLockedWords));

  const int iterations = static_cast<int>(I_ARG(4));
  const int numWorkers = static_cast<int>(I_ARG(5));
  const int nsRounds = static_cast<int>(I_ARG(6));
  REQUIRE_TRUE(iterations > 0, 0, "CBOW inference: the number of iterations must be positive, but got %i", iterations);

  int currIdx = 7;
  std::vector<int> codesVec;
  for (sd::LongType i = 0; i < numCodes; i++) {
    codesVec.push_back(static_cast<int>(I_ARG(currIdx++)));
  }

  std::vector<int> indicesVec;
  for (sd::LongType i = 0; i < numIndices; i++) {
    indicesVec.push_back(static_cast<int>(I_ARG(currIdx++)));
  }

  std::vector<int> contextVec;
  for (sd::LongType i = 0; i < numContext; i++) {
    contextVec.push_back(static_cast<int>(I_ARG(currIdx++)));
  }

  std::vector<int> lockedWordsVec;
  for (sd::LongType i = 0; i < numLockedWords; i++) {
    lockedWordsVec.push_back(static_cast<int>(I_ARG(currIdx++)));
  }

  const int target = static_cast<int>(I_ARG(currIdx++));
  const int ngStarter = static_cast<int>(I_ARG(currIdx++));
  const int numLabels = static_cast<int>(I_ARG(currIdx++));
  const sd::LongType randomValue = I_ARG(currIdx++);

  auto alpha = T_ARG(0);
  auto minLearningRate = block.numT() > 1 ? T_ARG(1) : 1e-3;

  auto syn0 = INPUT_VARIABLE(0);
  auto syn1 = INPUT_VARIABLE(1);
  auto syn1neg = INPUT_VARIABLE(2);

  auto expTable = INPUT_VARIABLE(3);
  auto negTable = INPUT_VARIABLE(4);

  auto inferenceVector = INPUT_VARIABLE(5);

  // the first boolean argument is trainWords, as it is for cbow; it is true when absent
  auto trainWords = block.numB() > 0 ? B_ARG(0) : true;

  REQUIRE_TRUE(block.isInplace(), 0, "CBOW: this operation requires inplace execution only");

  REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0,
               "CBOW: all syn tables must have the same data type");
  REQUIRE_TRUE(syn0->dataType() == expTable->dataType(), 0,
               "CBOW: expTable must have the same data type as syn0 table");
  // an empty optional input is absent: negative sampling cannot run without the rows and the table it draws from
  REQUIRE_TRUE(nsRounds <= 0 || (!syn1neg->isEmpty() && !negTable->isEmpty()), 0,
               "CBOW: %i negative sampling rounds need non-empty syn1Neg and negTable", nsRounds);
  // the helpers read the tables as rows of the width of syn0, and the negative table and the inference vector as syn0's type
  REQUIRE_TRUE(syn0->rankOf() == 2, 0, "CBOW: syn0 must be a matrix, but got rank %i", syn0->rankOf());
  REQUIRE_TRUE(syn1->isEmpty() || (syn1->rankOf() == 2 && syn1->sizeAt(0) >= syn0->sizeAt(0) &&
                                   syn1->sizeAt(1) == syn0->sizeAt(1)),
               0, "CBOW: syn1 must be as wide as syn0 and have a row for every row of it");
  REQUIRE_TRUE(syn1neg->isEmpty() || (syn1neg->rankOf() == 2 && syn1neg->sizeAt(0) >= syn0->sizeAt(0) &&
                                      syn1neg->sizeAt(1) == syn0->sizeAt(1)),
               0, "CBOW: syn1Neg must be as wide as syn0 and have a row for every row of it");
  REQUIRE_TRUE(inferenceVector->isEmpty() || (inferenceVector->dataType() == syn0->dataType() &&
                                              inferenceVector->lengthOf() == syn0->sizeAt(1)),
               0, "CBOW: the inference vector must have the data type of syn0 and the length of one of its rows");
  REQUIRE_TRUE(nsRounds <= 0 || (negTable->dataType() == syn0->dataType() && syn0->sizeAt(0) > 1), 0,
               "CBOW: negative sampling needs a negTable of the data type of syn0 and a vocabulary of at least two words");

  auto indicesArr = sd::ops::helpers::integerArgumentsArray(indicesVec);
  auto codesArr = sd::ops::helpers::integerArgumentsArray(codesVec);
  auto contextArr = sd::ops::helpers::integerArgumentsArray(contextVec);
  auto lockedWordsArr = sd::ops::helpers::integerArgumentsArray(lockedWordsVec);

  try {
    sd::ops::helpers::cbowInference(
        *syn0,
        *syn1,
        *syn1neg,
        *expTable,
        *negTable,
        target,
        ngStarter,
        nsRounds,
        *contextArr,
        *lockedWordsArr,
        *indicesArr,
        *codesArr,
        alpha,
        randomValue,
        numLabels,
        *inferenceVector,
        trainWords,
        numWorkers,iterations,minLearningRate);
  } catch (...) {
    delete indicesArr;
    delete codesArr;
    delete contextArr;
    delete lockedWordsArr;
    throw;
  }

  delete indicesArr;
  delete codesArr;
  delete contextArr;
  delete lockedWordsArr;

  return sd::Status::OK;
}

DECLARE_TYPES(cbow_inference) {
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedOutputTypes(sd::DataType::ANY);
}


CONFIGURABLE_OP_IMPL(cbow, 15, 15, true, 0, 0) {
  auto target = INPUT_VARIABLE(0);
  auto ngStarter = INPUT_VARIABLE(1);

  // required part
  auto context = INPUT_VARIABLE(2);
  auto indices = INPUT_VARIABLE(3);
  auto codes = INPUT_VARIABLE(4);

  auto syn0 = INPUT_VARIABLE(5);
  auto syn1 = INPUT_VARIABLE(6);
  auto syn1neg = INPUT_VARIABLE(7);

  auto expTable = INPUT_VARIABLE(8);
  auto negTable = INPUT_VARIABLE(9);

  auto alpha = INPUT_VARIABLE(10);
  auto randomValue = INPUT_VARIABLE(11);
  auto numLabels = INPUT_VARIABLE(12);

  auto lockedWords = INPUT_VARIABLE(13);

  auto inferenceVector = INPUT_VARIABLE(14);

  auto numWorkers = block.numI() > 0 ? INT_ARG(0) : omp_get_max_threads();
  auto nsRounds = block.numI() > 1 ? INT_ARG(1) : 0;
  auto iterations = block.numI() > 2 ? INT_ARG(2) : 1;

  auto trainWords = block.numB() > 0 ? B_ARG(0) : true;
  auto isInference = block.numB() > 1 ? B_ARG(1) : false;

  auto minLearningRate = block.numT() > 0 ? T_ARG(0) : 1e-3;

  REQUIRE_TRUE(block.isInplace(), 0, "CBOW: this operation requires inplace execution only");

  REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0,
               "CBOW: all syn tables must have the same data type");
  REQUIRE_TRUE(syn0->dataType() == expTable->dataType(), 0,
               "CBOW: expTable must have the same data type as syn0 table");
  REQUIRE_TRUE(iterations > 0, 0, "CBOW: the number of iterations must be positive, but got %lld",
               static_cast<long long>(iterations));
  // an empty optional input is absent: negative sampling cannot run without the word it samples for, the rows and the
  // table it draws from
  REQUIRE_TRUE(nsRounds <= 0 || (!ngStarter->isEmpty() && !syn1neg->isEmpty() && !negTable->isEmpty()), 0,
               "CBOW: %lld negative sampling rounds need non-empty ngStarter, syn1Neg and negTable",
               static_cast<long long>(nsRounds));
  // the helpers read the tables as rows of the width of syn0, and the negative table and the inference vector as syn0's type
  REQUIRE_TRUE(syn0->rankOf() == 2, 0, "CBOW: syn0 must be a matrix, but got rank %i", syn0->rankOf());
  REQUIRE_TRUE(syn1->isEmpty() || (syn1->rankOf() == 2 && syn1->sizeAt(0) >= syn0->sizeAt(0) &&
                                   syn1->sizeAt(1) == syn0->sizeAt(1)),
               0, "CBOW: syn1 must be as wide as syn0 and have a row for every row of it");
  REQUIRE_TRUE(syn1neg->isEmpty() || (syn1neg->rankOf() == 2 && syn1neg->sizeAt(0) >= syn0->sizeAt(0) &&
                                      syn1neg->sizeAt(1) == syn0->sizeAt(1)),
               0, "CBOW: syn1Neg must be as wide as syn0 and have a row for every row of it");
  REQUIRE_TRUE(inferenceVector->isEmpty() || (inferenceVector->dataType() == syn0->dataType() &&
                                              inferenceVector->lengthOf() == syn0->sizeAt(1)),
               0, "CBOW: the inference vector must have the data type of syn0 and the length of one of its rows");
  REQUIRE_TRUE(nsRounds <= 0 || (negTable->dataType() == syn0->dataType() && syn0->sizeAt(0) > 1), 0,
               "CBOW: negative sampling needs a negTable of the data type of syn0 and a vocabulary of at least two words");

  sd::ops::helpers::cbow(*syn0, *syn1, *syn1neg, *expTable, *negTable, *target, *ngStarter, nsRounds, *context,
                         *lockedWords, *indices, *codes, *alpha, *randomValue, *numLabels, *inferenceVector, trainWords,
                         numWorkers,minLearningRate,iterations);

  return sd::Status::OK;
}

DECLARE_TYPES(cbow) {
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::INT32)
      ->setAllowedInputTypes(1, sd::DataType::INT32)
      ->setAllowedInputTypes(2, sd::DataType::INT32)
      ->setAllowedInputTypes(3, sd::DataType::INT32)
      ->setAllowedInputTypes(4, {ALL_INTS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedInputTypes(6, {ALL_FLOATS})
      ->setAllowedInputTypes(7, {ALL_FLOATS})
      ->setAllowedInputTypes(8, {ALL_FLOATS})
      ->setAllowedInputTypes(9, {ALL_FLOATS})
      ->setAllowedInputTypes(10, {ALL_FLOATS})
      ->setAllowedInputTypes(11, sd::DataType::INT64)
      ->setAllowedInputTypes(12, sd::DataType::INT32)
      ->setAllowedInputTypes(13, sd::DataType::INT32)
      ->setAllowedInputTypes(14, {ALL_FLOATS})
      ->setAllowedOutputTypes(sd::DataType::ANY);
}
}  // namespace ops
}  // namespace sd

#endif
