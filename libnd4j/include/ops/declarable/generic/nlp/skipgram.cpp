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
#if NOT_EXCLUDED(OP_skipgram)

#include <ops/declarable/headers/nlp.h>
#include <ops/declarable/helpers/sg_cb.h>

namespace sd {
namespace ops {


CONFIGURABLE_OP_IMPL(skipgram_inference, 6, 6, true, -2, -2) {
  // The codes and indices travel in the IARGS, to avoid serialization overhead from the JVM for frequently created small
  // arrays. The layout is the one SkipGramInference writes:
  //   [numCodes, numIndices, iterations, codes..., indices..., target, ngStarter, randomValue, numWorkers, nsRounds]
  // the last two are optional
  REQUIRE_TRUE(block.numI() >= 6, 0, "SkipGram inference: at least 6 integer arguments are required, but got %i",
               static_cast<int>(block.numI()));
  REQUIRE_TRUE(block.numT() > 0, 0, "SkipGram inference: the learning rate is the first float argument");

  const sd::LongType numCodes = I_ARG(0);
  const sd::LongType numIndices = I_ARG(1);
  const sd::LongType numIterations = I_ARG(2);
  // 3 for the header, the codes and indices, 3 for the word to train, the word to sample for and the random value
  const sd::LongType numMin = 3 + numCodes + numIndices + 3;
  REQUIRE_TRUE(numCodes >= 0 && numIndices >= 0 && static_cast<sd::LongType>(block.numI()) >= numMin, 0,
               "SkipGram inference: the integer arguments do not hold %lld codes and %lld indices",
               static_cast<long long>(numCodes), static_cast<long long>(numIndices));
  REQUIRE_TRUE(numIterations > 0, 0, "SkipGram: the number of iterations must be positive, but got %lld",
               static_cast<long long>(numIterations));

  int currIdx = 3;
  std::vector<int> codesVec;
  for (sd::LongType i = 0; i < numCodes; i++) {
    codesVec.push_back(static_cast<int>(I_ARG(currIdx++)));
  }

  std::vector<int> indicesVec;
  for (sd::LongType i = 0; i < numIndices; i++) {
    indicesVec.push_back(static_cast<int>(I_ARG(currIdx++)));
  }

  const int target = static_cast<int>(I_ARG(currIdx++));
  const int ngStarter = static_cast<int>(I_ARG(currIdx++));
  const sd::LongType randomValue = I_ARG(currIdx++);
  const int numWorkers =
      static_cast<sd::LongType>(block.numI()) > numMin ? static_cast<int>(I_ARG(currIdx++)) : omp_get_max_threads();
  const int nsRounds =
      static_cast<sd::LongType>(block.numI()) > numMin + 1 ? static_cast<int>(I_ARG(currIdx++)) : 0;

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
  // an empty optional input is absent: negative sampling cannot run without the rows and the table it draws from
  REQUIRE_TRUE(nsRounds <= 0 || (!syn1neg->isEmpty() && !negTable->isEmpty()), 0,
               "SkipGram: %i negative sampling rounds need non-empty syn1Neg and negTable", nsRounds);
  // the helpers read the tables as rows of the width of syn0, and the negative table and the inference vector as syn0's type
  REQUIRE_TRUE(syn0->rankOf() == 2, 0, "SkipGram: syn0 must be a matrix, but got rank %i", syn0->rankOf());
  REQUIRE_TRUE(syn1->isEmpty() || (syn1->rankOf() == 2 && syn1->sizeAt(0) >= syn0->sizeAt(0) &&
                                   syn1->sizeAt(1) == syn0->sizeAt(1)),
               0, "SkipGram: syn1 must be as wide as syn0 and have a row for every row of it");
  REQUIRE_TRUE(syn1neg->isEmpty() || (syn1neg->rankOf() == 2 && syn1neg->sizeAt(0) >= syn0->sizeAt(0) &&
                                      syn1neg->sizeAt(1) == syn0->sizeAt(1)),
               0, "SkipGram: syn1Neg must be as wide as syn0 and have a row for every row of it");
  REQUIRE_TRUE(inferenceVector->isEmpty() || (inferenceVector->dataType() == syn0->dataType() &&
                                              inferenceVector->lengthOf() == syn0->sizeAt(1)),
               0, "SkipGram: the inference vector must have the data type of syn0 and the length of one of its rows");
  REQUIRE_TRUE(nsRounds <= 0 || (negTable->dataType() == syn0->dataType() && syn0->sizeAt(0) > 1), 0,
               "SkipGram: negative sampling needs a negTable of the data type of syn0 and a vocabulary of at least two "
               "words");

  // an array cannot be made of no data: a configuration without codes or indices gets the empty array
  auto indicesArr = sd::ops::helpers::integerArgumentsArray(indicesVec);
  auto codesArr = sd::ops::helpers::integerArgumentsArray(codesVec);

  try {
    sd::ops::helpers::skipgramInference(*syn0,
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
  } catch (...) {
    delete indicesArr;
    delete codesArr;
    throw;
  }

  delete indicesArr;
  delete codesArr;

  return sd::Status::OK;
}


DECLARE_TYPES(skipgram_inference) {
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
  auto iterations = block.numI() > 2 ? INT_ARG(2) : 1;

  auto isInference = block.numB() > 0 ? B_ARG(0) : false;
  auto isPreciseMode = block.numB() > 1 ? B_ARG(1) : false;

  auto minLearningRate = block.numT() > 0 ? T_ARG(0) : 1e-4;


  REQUIRE_TRUE(block.isInplace(), 0, "SkipGram: this operation requires inplace execution only");

  REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0,
               "SkipGram: all syn tables must have the same data type");
  REQUIRE_TRUE(syn0->dataType() == expTable->dataType(), 0,
               "SkipGram: expTable must have the same data type as syn0 table");
  REQUIRE_TRUE(iterations > 0, 0, "SkipGram: the number of iterations must be positive, but got %lld",
               static_cast<long long>(iterations));
  // an empty optional input is absent: negative sampling cannot run without the word it samples for, the rows and the
  // table it draws from
  REQUIRE_TRUE(nsRounds <= 0 || (!ngStarter->isEmpty() && !syn1neg->isEmpty() && !negTable->isEmpty()), 0,
               "SkipGram: %lld negative sampling rounds need non-empty ngStarter, syn1Neg and negTable",
               static_cast<long long>(nsRounds));
  // the helpers read the tables as rows of the width of syn0, and the negative table and the inference vector as syn0's type
  REQUIRE_TRUE(syn0->rankOf() == 2, 0, "SkipGram: syn0 must be a matrix, but got rank %i", syn0->rankOf());
  REQUIRE_TRUE(syn1->isEmpty() || (syn1->rankOf() == 2 && syn1->sizeAt(0) >= syn0->sizeAt(0) &&
                                   syn1->sizeAt(1) == syn0->sizeAt(1)),
               0, "SkipGram: syn1 must be as wide as syn0 and have a row for every row of it");
  REQUIRE_TRUE(syn1neg->isEmpty() || (syn1neg->rankOf() == 2 && syn1neg->sizeAt(0) >= syn0->sizeAt(0) &&
                                      syn1neg->sizeAt(1) == syn0->sizeAt(1)),
               0, "SkipGram: syn1Neg must be as wide as syn0 and have a row for every row of it");
  REQUIRE_TRUE(inferenceVector->isEmpty() || (inferenceVector->dataType() == syn0->dataType() &&
                                              inferenceVector->lengthOf() == syn0->sizeAt(1)),
               0, "SkipGram: the inference vector must have the data type of syn0 and the length of one of its rows");
  REQUIRE_TRUE(nsRounds <= 0 || (negTable->dataType() == syn0->dataType() && syn0->sizeAt(0) > 1), 0,
               "SkipGram: negative sampling needs a negTable of the data type of syn0 and a vocabulary of at least two "
               "words");

  sd::ops::helpers::skipgram(*syn0, *syn1, *syn1neg, *expTable, *negTable, *target, *ngStarter, nsRounds, *indices,
                             *codes, *alpha, *randomValue, *inferenceVector, isPreciseMode, numWorkers,iterations,minLearningRate);

  return sd::Status::OK;
}

DECLARE_TYPES(skipgram) {
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::INT32)
      ->setAllowedInputTypes(1, sd::DataType::INT32)
      ->setAllowedInputTypes(2, sd::DataType::INT32)
      ->setAllowedInputTypes(3, {ALL_INTS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedInputTypes(6, {ALL_FLOATS})
      ->setAllowedInputTypes(7, {ALL_FLOATS})
      ->setAllowedInputTypes(8, {ALL_FLOATS})
      ->setAllowedInputTypes(9, {ALL_FLOATS})
      ->setAllowedInputTypes(10, sd::DataType::INT64)
      ->setAllowedInputTypes(11, {ALL_FLOATS})
      ->setAllowedOutputTypes(sd::DataType::ANY);
}


}  // namespace ops
}  // namespace sd

#endif
