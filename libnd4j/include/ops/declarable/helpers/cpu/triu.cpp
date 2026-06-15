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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <helpers/Loops.h>
#include <ops/declarable/helpers/transforms.h>
#if NOT_EXCLUDED(OP_triu)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void triuBP_(sd::LaunchContext* context, NDArray& input, NDArray& gradO, NDArray& gradI,
                    const int diagonal) {
  if (gradO.isScalar()) {
    auto scalar = gradO.e<T>(0);
    gradI.assign(scalar);
  } else {
    gradI.assign(&gradO);
  }
  gradI.fillAsTriangular<T>(0.0f, diagonal, diagonal, gradI, 'l', false);
}

void triuBP(sd::LaunchContext* context, NDArray& input, NDArray& gradO, NDArray& gradI,
            const int diagonal) {
  BUILD_SINGLE_SELECTOR(gradO.dataType(), triuBP_, (context, input, gradO, gradI, diagonal), SD_COMMON_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
