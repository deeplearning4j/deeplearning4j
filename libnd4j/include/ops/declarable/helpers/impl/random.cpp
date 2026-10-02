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

// Backend-neutral part of the distribution helpers; the samplers run in
//   ops/declarable/helpers/cpu/random.cpp   (CPU build)
//   ops/declarable/helpers/cuda/random.cu   (CUDA build)
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/random.h>
#if NOT_EXCLUDED(OP_random)
namespace sd {
namespace ops {
namespace helpers {

NDArray* randomParameter(NDArray* values, const LongType* shapeInfo, DataType computeType, LaunchContext* context) {
  std::vector<LongType> shape(shape::shapeOf(shapeInfo), shape::shapeOf(shapeInfo) + shape::rank(shapeInfo));
  auto* parameter = new NDArray('c', shape, computeType, context);
  if (shape::equalsSoft(values->shapeInfo(), shapeInfo)) {
    parameter->assign(values);
  } else {
    NDArray* cast = values->cast(computeType);
    parameter->applyTrueBroadcast(BroadcastOpsTuple::Assign(), cast, parameter);
    delete cast;
  }
  return parameter;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
