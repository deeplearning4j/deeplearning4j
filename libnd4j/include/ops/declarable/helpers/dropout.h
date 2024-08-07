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
#ifndef __DROP_OUT_HELPERS__
#define __DROP_OUT_HELPERS__
#include <array/NDArray.h>
#include <graph/Context.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN Status dropOutFunctor(Context& context, NDArray* input, NDArray* output, NDArray* reduceShape,
                                        int seed, double probValue, NDArray* mask);
SD_LIB_HIDDEN Status dropOutFunctorBP(Context& context, NDArray* input, NDArray* gradOut, NDArray* output,
                                          NDArray* reduceShape, int seed, double probValue, NDArray* mask);
SD_LIB_HIDDEN Status alphaDropOutFunctor(Context& context, NDArray* input, NDArray* output,
                                             NDArray* reduceShape, int seed, double probValue, double alpha,
                                             double alpha1, double beta, NDArray* mask);
SD_LIB_HIDDEN Status alphaDropOutFunctorBP(Context& context, NDArray* input, NDArray* gradOut,
                                               NDArray* output, NDArray* reduceShape, int seed, double probValue,
                                               double alpha, double alpha1, double beta, NDArray* mask);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
