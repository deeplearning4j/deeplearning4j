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

  auto flattenedInput = input->reshape('c',{inLen},false);
  auto flattenedOutput = output->reshape('c',{output->lengthOf()},false);
  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      float val = nodeRng.relativeT<T>(e, T(0.f), T(1.f));
      //dropout mask might not be the same length
      if (mask != nullptr && e < mask->lengthOf()) mask->p<T>(e, val);
      if (val < probValue) flattenedOutput.p<T>(e, flattenedInput.e<T>(e));
    }
  };

  samediff::Threads::parallel_for(func, 0, inLen);
}
BUILD_SINGLE_TEMPLATE(template void dropoutSimple, (NDArray* input, NDArray* output, double probValue, int seed,NDArray *mask),
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
    for (auto i = 0; i < dims.size(); i++) {
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
    chunk->assign(1.f);
    dropoutSimple<T>(chunk.get(), chunk.get(), probValue, seed, nullptr);
    // broadcast chunk to full matrix
    mask->assign(1.f);

    *mask += *chunk;

    output->assign(*input * *mask);
  }

  return sd::Status::OK;
}

sd::Status dropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                          double probValue, NDArray* mask) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, return dropOutFunctor_, (context, input, output, reduceShape, seed, probValue,mask),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template sd::Status dropOutFunctor_, (graph::Context & context, NDArray* input, NDArray* output,
    NDArray* reduceShape, int seed, double probValue,NDArray *mask);
, SD_FLOAT_TYPES);

/////////////////////////////////// backprpopagations ///////////////////////////////////////////////
template <typename T>
static Status dropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                NDArray* reduceShape, int seed, double probValue, NDArray* mask) {
  *output = *gradOut * *mask;
  return sd::Status::OK;
}

template <typename T>
static Status alphaDropOutFunctor_(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape,
                                   int seed, double probValue, double alpha, double alpha1, double beta,
                                   NDArray* mask) {

  sd::graph::RandomGenerator nodeRng(3019L, seed);

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      float randVal = nodeRng.relativeT(e, T(0.f), T(1.f));
      float xVal = input->e<float>(e);
      float maskVal = randVal >= probValue ? alpha * beta + alpha1 : alpha * 1 + alpha1;
      mask->p<float>(e, maskVal);
      output->p<float>(e, randVal >= probValue ? alpha * beta + alpha1 : alpha * xVal + alpha1);
    }
  };

  samediff::Threads::parallel_for(func, 0, input->lengthOf());

  return sd::Status::OK;
}

template <typename T>
sd::Status alphaDropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                  NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1,
                                  double beta, NDArray* mask) {

  *output *= *gradOut * *mask;
  return sd::Status::OK;
}

sd::Status dropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                            NDArray* reduceShape, int seed, double probValue, NDArray* mask) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return dropOutFunctorBP_,
                        (context, input, gradOut, output, reduceShape, seed, probValue,mask), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template sd::Status dropOutFunctorBP_,
                      (graph::Context & context, NDArray* input, NDArray* gradOut, NDArray* output,
                          NDArray* reduceShape, int seed, double probValue,NDArray* mask),
                      SD_FLOAT_TYPES);

sd::Status alphaDropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                               double probValue, double alpha, double alpha1, double beta, NDArray* mask) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctor_,
                        (context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta,mask), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template sd::Status alphaDropOutFunctor_,
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
BUILD_SINGLE_TEMPLATE(template sd::Status alphaDropOutFunctorBP_,
                      (graph::Context & context, NDArray* input, NDArray* gradOut, NDArray* output,
                          NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta,NDArray *mask),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif