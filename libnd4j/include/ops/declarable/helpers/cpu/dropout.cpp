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
#include <execution/Threads.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/helpers/dropout.h>

#include <memory>
#include <vector>
#if NOT_EXCLUDED(OP_dropout)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void dropoutSimple(NDArray* input, NDArray* output, double probValue, int seed, NDArray* mask) {
  sd::graph::RandomGenerator nodeRng(3019L, seed);
  int inLen = input->lengthOf();
  std::vector<sd::LongType> inShape = {inLen};
  std::vector<sd::LongType> outShape = {output->lengthOf()};
  auto flattenedInput = input->reshape('c',inShape,false);
  auto flattenedOutput = output->reshape('c',outShape,false);

  // Flatten mask to properly use linear indexing
  NDArray* flattenedMask = nullptr;
  if (mask != nullptr) {
    std::vector<sd::LongType> maskShape = {mask->lengthOf()};
    flattenedMask = mask->reshape('c', maskShape, false);
  }

  // Get typed buffer pointers once to avoid O(n^2) sync overhead from per-element p()/e() calls
  NDArray::preparePrimaryUse({flattenedOutput, flattenedMask}, {flattenedInput});
  auto inputBuf = flattenedInput->bufferAsT<T>();
  auto outputBuf = flattenedOutput->bufferAsT<T>();
  T* maskBuf = (flattenedMask != nullptr) ? flattenedMask->bufferAsT<T>() : nullptr;

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      float val = nodeRng.relativeT<T>(e, T(0.f), T(1.f));
      // Keep the value if val < probValue (probValue is keep probability)
      bool keep = val < probValue;
      // Store binary mask: 1 if kept, 0 if dropped
      if (maskBuf != nullptr && e < flattenedMask->lengthOf()) {
        auto maskOffset = flattenedMask->getOffset(e);
        maskBuf[maskOffset] = keep ? static_cast<T>(1) : static_cast<T>(0);
      }
      // Output is input when kept, 0 otherwise (OUTPUT_NULLIFIED already zeros it)
      if (keep) {
        auto outOffset = flattenedOutput->getOffset(e);
        auto inOffset = flattenedInput->getOffset(e);
        outputBuf[outOffset] = inputBuf[inOffset];
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, inLen);

  NDArray::registerPrimaryUse({flattenedOutput, flattenedMask}, {flattenedInput});

  delete flattenedInput;
  delete flattenedOutput;
  if (flattenedMask != nullptr) {
    delete flattenedMask;
  }
}
BUILD_SINGLE_TEMPLATE( void dropoutSimple, (NDArray* input, NDArray* output, double probValue, int seed,NDArray *mask),
                      SD_FLOAT_TYPES);

template <typename T>
sd::Status dropOutFunctor_(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                           double probValue, NDArray* mask) {

  if (reduceShape == nullptr) {
    dropoutSimple<T>(input, output, probValue, seed, mask);
  } else {
    REQUIRE_TRUE(reduceShape->lengthOf() <= input->rankOf(), 0, "dropout: Noise shape should be fittable to input");

    std::vector<sd::LongType> dims(reduceShape->lengthOf());

    bool fit = true;
    for (size_t i = 0; i < dims.size(); i++) {
      if (fit) {
        dims[i] = reduceShape->e<sd::LongType>(i);
        for (int e = 0; e < input->rankOf(); ++e)
          if (fit)
            if (input->sizeAt(e) % dims[i]) {
              fit = false;
            }
      }
    }

    // check dims to fit input
    REQUIRE_TRUE(fit, 0, "dropout: Noise shape should fit to input rank.");
    std::unique_ptr<NDArray> chunk(new NDArray('c', dims, output->dataType(), output->getContext()));
    float assign = 1.f;
    chunk->assign(assign);
    dropoutSimple<T>(chunk.get(), chunk.get(), probValue, seed, nullptr);
    // broadcast chunk to full matrix
    mask->assign(assign);

    *mask += *chunk;
    NDArray *assign5 = *input * *mask;
    output->assign(assign5);
    delete assign5;
  }

  return sd::Status::OK;
}

sd::Status dropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                          double probValue, NDArray* mask) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, return dropOutFunctor_, (context, input, output, reduceShape, seed, probValue,mask),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( sd::Status dropOutFunctor_, (graph::Context & context, NDArray* input, NDArray* output,
    NDArray* reduceShape, int seed, double probValue,NDArray *mask);
, SD_FLOAT_TYPES);

/////////////////////////////////// backprpopagations ///////////////////////////////////////////////
template <typename T>
static Status dropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                NDArray* reduceShape, int seed, double probValue, NDArray* mask) {
  // Use assign and in-place multiply to avoid temporary NDArray creation
  // which can cause ownership issues with the assignment operator
  output->assign(gradOut);
  *output *= *mask;
  return sd::Status::OK;
}

template <typename T>
static Status alphaDropOutFunctor_(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape,
                                   int seed, double probValue, double alpha, double alpha1, double beta,
                                   NDArray* mask) {

  sd::graph::RandomGenerator nodeRng(3019L, seed);

  // Get typed buffer pointers once to avoid O(n^2) sync overhead from per-element p()/e() calls
  NDArray::preparePrimaryUse({output, mask}, {input});
  auto inputBuf = input->bufferAsT<T>();
  auto outputBuf = output->bufferAsT<T>();
  auto maskBuf = mask->bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      T randVal = nodeRng.relativeT(e, T(0.f), T(1.f));
      auto inOffset = input->getOffset(e);
      T xVal = inputBuf[inOffset];
      T maskVal = randVal >= static_cast<T>(probValue) ? static_cast<T>(alpha * beta + alpha1) : static_cast<T>(alpha + alpha1);
      auto maskOffset = mask->getOffset(e);
      maskBuf[maskOffset] = maskVal;
      auto outOffset = output->getOffset(e);
      outputBuf[outOffset] = randVal >= static_cast<T>(probValue) ? static_cast<T>(alpha * beta + alpha1) : static_cast<T>(alpha * static_cast<double>(xVal) + alpha1);
    }
  };

  samediff::Threads::parallel_for(func, 0, input->lengthOf());

  NDArray::registerPrimaryUse({output, mask}, {input});

  return sd::Status::OK;
}

template <typename T>
sd::Status alphaDropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                  NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1,
                                  double beta, NDArray* mask) {
  // Use in-place operations to avoid temporary NDArray creation
  // which can cause ownership issues with the assignment operator
  *output *= *gradOut;
  *output *= *mask;
  return sd::Status::OK;
}

sd::Status dropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                            NDArray* reduceShape, int seed, double probValue, NDArray* mask) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return dropOutFunctorBP_,
                        (context, input, gradOut, output, reduceShape, seed, probValue,mask), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE( sd::Status dropOutFunctorBP_,
                      (::Context & context, NDArray* input, NDArray* gradOut, NDArray* output,
                          NDArray* reduceShape, int seed, double probValue,NDArray* mask),
                      SD_FLOAT_TYPES);

sd::Status alphaDropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                               double probValue, double alpha, double alpha1, double beta, NDArray* mask) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctor_,
                        (context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta,mask), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE( sd::Status alphaDropOutFunctor_,
                      (graph::Context & context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                          double probValue, double alpha, double alpha1, double beta,NDArray* mask),
                      SD_FLOAT_TYPES);

sd::Status alphaDropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                 NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1,
                                 double beta, NDArray* mask) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctorBP_,
                        (context, input, gradOut, output, reduceShape, seed, probValue, alpha, alpha1, beta,mask),
                        SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE( sd::Status alphaDropOutFunctorBP_,
                      (graph::Context & context, NDArray* input, NDArray* gradOut, NDArray* output,
                          NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta,NDArray *mask),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif