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
static void dropoutSimple(NDArray const* input, NDArray* output, double probValue, int seed) {
  sd::graph::RandomGenerator nodeRng(3019L, seed);
  int inLen = input->lengthOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      float val = nodeRng.relativeT<T>(e, T(0.f), T(1.f));

      if (val < probValue) output->p<T>(e, input->e<T>(e));
    }
  };

  samediff::Threads::parallel_for(func, 0, inLen);
}
BUILD_SINGLE_TEMPLATE(template void dropoutSimple, (NDArray const* input, NDArray* output, double probValue, int seed),
                      SD_FLOAT_TYPES);

template <typename T>
sd::Status dropOutFunctor_(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                           double probValue) {

  if (reduceShape == nullptr) {
    dropoutSimple<T>(input, output, probValue, seed);
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
    dropoutSimple<T>(chunk.get(), chunk.get(), probValue, seed);
    // broadcast chunk to full matrix
    std::unique_ptr<NDArray> dropOutMultiplier(new NDArray(*input));
    dropOutMultiplier->assign(1.f);

    *dropOutMultiplier += *chunk;

    output->assign(*input * *dropOutMultiplier);
  }

  return sd::Status::OK;
}

sd::Status dropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                          double probValue) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, return dropOutFunctor_, (context, input, output, reduceShape, seed, probValue),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template sd::Status dropOutFunctor_, (graph::Context & context, NDArray* input, NDArray* output,
                                                            NDArray* reduceShape, int seed, double probValue);
                      , SD_FLOAT_TYPES);

/////////////////////////////////// backrpopagations ///////////////////////////////////////////////
template <typename T>
static sd::Status dropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                    NDArray* reduceShape, int seed, double probValue) {
  auto res = dropOutFunctor(context, input, output, reduceShape, seed, probValue);

  if (sd::Status::OK == res)
    for (sd::LongType e = 0; e < output->lengthOf(); e++) {
      if (output->e<float>(e) != 0.f) output->p<T>(e, gradOut->e<double>(e) / probValue);
      //            else (*output)(e) = T(0.f);
    }

  return res;
}

template <typename T>
static sd::Status alphaDropOutFunctor_(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape,
                                       int seed, double probValue, double alpha, double alpha1, double beta) {

  sd::graph::RandomGenerator nodeRng(3019L, seed);

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      float randVal = nodeRng.relativeT(e, T(0.f), T(1.f));
      float xVal = input->e<float>(e);
      output->p<float>(e, randVal >= probValue ? alpha * beta + alpha1 : alpha * xVal + alpha1);
    }
  };

  samediff::Threads::parallel_for(func, 0, input->lengthOf());

  return sd::Status::OK;
}

template <typename T>
sd::Status alphaDropOutFunctorBP_(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                  NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1,
                                  double beta) {
  auto res = alphaDropOutFunctor(context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta);
  if (res == sd::Status::OK) {
    (*output) *= alpha;
    (*output) *= (*gradOut);
  }
  return res;
}

sd::Status dropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                            NDArray* reduceShape, int seed, double probValue) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return dropOutFunctorBP_,
                        (context, input, gradOut, output, reduceShape, seed, probValue), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template sd::Status dropOutFunctorBP_,
                      (graph::Context & context, NDArray* input, NDArray* gradOut, NDArray* output,
                       NDArray* reduceShape, int seed, double probValue),
                      SD_FLOAT_TYPES);

sd::Status alphaDropOutFunctor(graph::Context& context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                               double probValue, double alpha, double alpha1, double beta) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctor_,
                        (context, input, output, reduceShape, seed, probValue, alpha, alpha1, beta), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template sd::Status alphaDropOutFunctor_,
                      (graph::Context & context, NDArray* input, NDArray* output, NDArray* reduceShape, int seed,
                       double probValue, double alpha, double alpha1, double beta),
                      SD_FLOAT_TYPES);

sd::Status alphaDropOutFunctorBP(graph::Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                 NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1,
                                 double beta) {
  BUILD_SINGLE_SELECTOR(context.dataType(), return alphaDropOutFunctorBP_,
                        (context, input, gradOut, output, reduceShape, seed, probValue, alpha, alpha1, beta),
                        SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template sd::Status alphaDropOutFunctorBP_,
                      (graph::Context & context, NDArray* input, NDArray* gradOut, NDArray* output,
                       NDArray* reduceShape, int seed, double probValue, double alpha, double alpha1, double beta),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif