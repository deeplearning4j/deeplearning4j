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
// @author Yurii Shyrma, created on 16.04.2017
//

#ifndef LIBND4J_RNN_H
#define LIBND4J_RNN_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void rnnCell(LaunchContext* context, NDArray* xt, NDArray* Wx, NDArray* Wh,
                           NDArray* b, NDArray* ht_1, NDArray* ht);

SD_LIB_HIDDEN void rnnTimeLoop(LaunchContext* context, NDArray* x, NDArray* Wx, NDArray* Wh,
                               NDArray* b, NDArray* h0, NDArray* maxTimeStep, NDArray* h,
                               NDArray* hFinal);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_RNN_H
