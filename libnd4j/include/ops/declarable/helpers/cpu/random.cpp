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
//  @author sgazeos@gmail.com
//
#include <ops/declarable/helpers/random.h>
#include <memory>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/RandomLauncher.h>
#include <helpers/ShapeUtils.h>
#if NOT_EXCLUDED(OP_random)
namespace sd {
namespace ops {
namespace helpers {


template <typename Z>
static void fillRandomGamma_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                             NDArray* output) {
  using C = RandomComputeT<Z>;
  // Output element i samples with parameters i % length of alpha and beta broadcast together.
  LongType* parameterShape = alpha->shapeInfo();
  if (beta != nullptr)
    ShapeUtils::evalBroadcastShapeInfo(alpha->shapeInfo(), beta->shapeInfo(), true, parameterShape,
                                       context->getWorkspace());
  NDArray* alphas = randomParameter(alpha, parameterShape, DataTypeUtils::fromT<C>(), context);
  NDArray* rates = beta != nullptr ? randomParameter(beta, parameterShape, DataTypeUtils::fromT<C>(), context) : nullptr;

  NDArray::preparePrimaryUse({output}, {alphas, rates});
  const C* alphaBuf = alphas->bufferAsT<C>();
  const C* rateBuf = rates != nullptr ? rates->bufferAsT<C>() : nullptr;
  const LongType parameters = alphas->lengthOf();
  Z* outputBuf = output->bufferAsT<Z>();
  const LongType* outputShapeInfo = output->shapeInfo();
  const LongType rank = shape::rank(outputShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    LongType coords[SD_MAX_RANK];
    LongType offset;
    for (auto i = start; i < stop; i++) {
      const LongType p = i % parameters;
      const C sample = sampleGamma<C>(rng, i, alphaBuf[p], rateBuf != nullptr ? rateBuf[p] : C(1));
      INDEX2COORDS(i, rank, shape::shapeOf(outputShapeInfo), coords);
      COORDS2INDEX(rank, shape::stride(outputShapeInfo), coords, offset);
      outputBuf[offset] = static_cast<Z>(sample);
    }
  };
  samediff::Threads::parallel_for(func, 0, output->lengthOf());
  NDArray::registerPrimaryUse({output}, {alphas, rates});
  rng.rewindH(output->lengthOf());

  delete alphas;
  delete rates;
}

void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                     NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomGamma_, (context, rng, alpha, beta, output), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE( void fillRandomGamma_,
                      (LaunchContext * context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                       NDArray* output),
                      SD_FLOAT_TYPES);

template <typename Z>
static void fillRandomPoisson_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
  using C = RandomComputeT<Z>;
  // Output element i samples with lambda i % lambda's length.
  NDArray* lambdas = randomParameter(lambda, lambda->shapeInfo(), DataTypeUtils::fromT<C>(), context);

  NDArray::preparePrimaryUse({output}, {lambdas});
  const C* lambdaBuf = lambdas->bufferAsT<C>();
  const LongType parameters = lambdas->lengthOf();
  Z* outputBuf = output->bufferAsT<Z>();
  const LongType* outputShapeInfo = output->shapeInfo();
  const LongType rank = shape::rank(outputShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    LongType coords[SD_MAX_RANK];
    LongType offset;
    for (auto i = start; i < stop; i++) {
      const C sample = samplePoisson<C>(rng, i, lambdaBuf[i % parameters]);
      INDEX2COORDS(i, rank, shape::shapeOf(outputShapeInfo), coords);
      COORDS2INDEX(rank, shape::stride(outputShapeInfo), coords, offset);
      outputBuf[offset] = static_cast<Z>(sample);
    }
  };
  samediff::Threads::parallel_for(func, 0, output->lengthOf());
  NDArray::registerPrimaryUse({output}, {lambdas});
  rng.rewindH(output->lengthOf());

  delete lambdas;
}

void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomPoisson_, (context, rng, lambda, output), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void fillRandomPoisson_,
                      (LaunchContext * context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output),
                      SD_FLOAT_TYPES);

template <typename T>
void fillRandomUniform_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max,
                        NDArray* output) {
  T minVal = T(0);
  T maxVal = DataTypeUtils::max<T>();
  if (min) minVal = min->t<T>(0);
  if (max) maxVal = max->t<T>(0);

  if (output->isR())
    RandomLauncher::fillUniform(context, rng, output, minVal, maxVal);
  else {
    PRAGMA_OMP_PARALLEL_FOR
    for (sd::LongType i = 0; i < output->lengthOf(); i++) {
      output->r<T>(i) = rng.relativeT<T>(i, minVal, maxVal);
    }
    // The floating fill above rewinds inside the random launcher; the next fill must draw anew.
    rng.rewindH(output->lengthOf());
  }
}

void fillRandomUniform(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max,
                       NDArray* output) {
  BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomUniform_, (context, rng, min, max, output), SD_NUMERIC_TYPES);
}

// used https://en.wikipedia.org/wiki/Categorical_distribution
// methods: gumbel trick + softmax + argmax
template <typename Tx, typename Tz>
void fillRandomMultiNomial_(LaunchContext* context, graph::RandomGenerator& rng, NDArray& input, NDArray& output,
                            const sd::LongType numOfSamples, const int dimC) {
  const Tx* x = input.bufferAsT<Tx>();
  Tz* z = output.bufferAsT<Tz>();

  Tx minVal = DataTypeUtils::min_positive<Tx>();
  Tx maxVal = static_cast<Tx>(1.0);

  auto dimA = (0 == dimC) ? 1 : 0;
  const sd::LongType batchValue = output.sizeAt(dimC);
  const sd::LongType numOfClassX = input.sizeAt(dimA);

  const sd::LongType zDimAstride = output.stridesOf()[dimA];
  const sd::LongType xDimAstride = input.stridesOf()[dimA];
  const sd::LongType zDimCstride = output.stridesOf()[dimC];
  const sd::LongType xDimCstride = input.stridesOf()[dimC];

  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto nBatchIndex = start_x; nBatchIndex < stop_x; nBatchIndex += inc_x) {
      for (auto nSampleIndexInBatch = start_y; nSampleIndexInBatch < stop_y; nSampleIndexInBatch += inc_y) {
        const Tx* xTad = x + (nBatchIndex * xDimCstride);
        Tz* zTad = z + (nBatchIndex * zDimCstride);
        Tz& arg = zTad[nSampleIndexInBatch * zDimAstride];
        Tx Max = -minVal;

        auto nSamplesPerBatch = nBatchIndex * numOfClassX * numOfSamples;
        auto nClassesPerSample = nSampleIndexInBatch * numOfClassX;
        for (sd::LongType nClass = 0; nClass < numOfClassX; nClass += 1) {
          auto nIndex = nSamplesPerBatch + nClassesPerSample + nClass;
          auto unifornLog =
              sd::math::sd_log<Tx, Tx>(-sd::math::sd_log<Tx, Tx>(rng.relativeT<Tx>(nIndex, minVal, maxVal)));
          Tx tValue = (xTad[nClass * xDimAstride] - unifornLog);
          if (tValue > Max) {
            Max = tValue;
            arg = nClass;
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, batchValue, 1, 0, numOfSamples, 1);
  rng.rewindH(output.lengthOf() * numOfClassX);

  return;
}

void fillRandomMultiNomial(LaunchContext* context, graph::RandomGenerator& rng, NDArray& input, NDArray& output,
                           const sd::LongType numOfSamples, const int dimC) {
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), fillRandomMultiNomial_,
                        (context, rng, input, output, numOfSamples, dimC), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif