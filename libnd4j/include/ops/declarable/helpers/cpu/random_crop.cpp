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
#include <ops/declarable/helpers/random_crop.h>
#include <graph/Context.h>

#if NOT_EXCLUDED(OP_random_shuffle)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static sd::Status _randomCropFunctor(graph::Context& context, NDArray* input, NDArray* shape, NDArray* output,
                                     int seed) {
  graph::RandomGenerator rngX(context.getRng());
  // functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(rng,
  // output->buffer(), output->shapeInfo(), std::vector<T>({T(0.), shape->e(last)}).data());
  // NativeOpExecutioner::execRandom(random::UniformDistribution, rng, output->buffer(), output->shapeInfo(),
  // std::vector<T>({T(0.), shape->e<T>(last)}).data());
  sd::LongType last = shape->lengthOf() - 1;

  rngX.setSeed(seed);
  // functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(rng,
  // output->buffer(), output->shapeInfo(), std::vector<T>({T(0.), shape->getScalar(last)}).data());
  for (sd::LongType e = 0; e < output->lengthOf(); ++e) {
    output->p(e, rngX.relativeT<T>(e, 0, shape->e<sd::LongType>(last)));
  }
  sd::LongType maxIndex = output->argMax();
  sd::LongType startPos = output->e<sd::LongType>(maxIndex);
  sd::LongType lastDim = input->sizeAt(-1);
  // sd_printf("Before processing: %i %i. Output length %i\n", maxIndex, startPos, output->lengthOf());
  sd::LongType pos = 0;
  sd::LongType width = startPos + shape->e<sd::LongType>(last);
  if (width >= lastDim) {
    startPos -= (width - lastDim);
    width = lastDim;
  }

  for (sd::LongType i = 0; i < input->lengthOf(); i += lastDim) {
    for (sd::LongType k = startPos; k < width && pos < output->lengthOf(); k++) {
      output->p(pos++, input->e<T>(i + k));
    }
  }
  return sd::Status::OK;
}

sd::Status randomCropFunctor(graph::Context& context, NDArray* input, NDArray* shape, NDArray* output, int seed) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return _randomCropFunctor, (context, input, shape, output, seed),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template sd::Status _randomCropFunctor,
                      (graph::Context & context, NDArray* input, NDArray* shape, NDArray* output, int seed),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif