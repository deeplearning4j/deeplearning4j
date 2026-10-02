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

// One implementation for every backend: the crop is a sub-array view of the input, which the
// output's own assign copies on the input's device.
#include <ops/declarable/helpers/random.h>
#include <ops/declarable/helpers/random_crop.h>
#if NOT_EXCLUDED(OP_random_crop)
namespace sd {
namespace ops {
namespace helpers {

Status randomCropFunctor(graph::Context& context, NDArray* input, NDArray* shape, NDArray* output, int seed) {
  if (output->lengthOf() == 0) return Status::OK;
  auto& rng = context.randomGenerator();
  applySeedArgument(rng, seed);

  // Every dimension is cropped at an offset drawn uniformly from those where the crop fits.
  const int rank = input->rankOf();
  std::vector<LongType> ranges(2 * rank);
  for (int d = 0; d < rank; d++) {
    const LongType size = output->sizeAt(d);
    const LongType offsets = input->sizeAt(d) - size + 1;
    const LongType offset = offsets > 1 ? rng.relativeLong(d) % offsets : 0;
    ranges[2 * d] = offset;
    ranges[2 * d + 1] = offset + size;
  }
  rng.rewindH(rank);

  NDArray* crop = (*input)(ranges, true);
  output->assign(crop);
  delete crop;
  return Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
