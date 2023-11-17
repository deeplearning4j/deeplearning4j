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
#if NOT_EXCLUDED(OP_random_normal)

#include <helpers/RandomLauncher.h>
#include <ops/declarable/headers/random.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(random_normal, 1, 1, true, 2, 0) {
  // normal distribution
  auto rng = block.randomGenerator();

  RandomLauncher::fillGaussian(block.launchContext(), rng, OUTPUT_VARIABLE(0), T_ARG(0), T_ARG(1));

  return Status::OK;
}

DECLARE_SHAPE_FN(random_normal) {
  auto in = INPUT_VARIABLE(0);
  auto shape = in->template asVectorT<LongType>();
  if(block.getDArguments()->size() > 0) {
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(D_ARG(0), 'c', shape);
    return SHAPELIST(newShape);
  } else {
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(block.dataType(), 'c', shape);
    return SHAPELIST(newShape);
  }

}

DECLARE_SYN(randomnormal, random_normal);

DECLARE_TYPES(random_normal) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}
}  // namespace ops
}  // namespace sd

#endif
