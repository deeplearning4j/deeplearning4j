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
#ifndef __LUP_H_HELPERS__
#define __LUP_H_HELPERS__
#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

    int lup(sd::LaunchContext* context, NDArray* input, NDArray* lu, NDArray* permutation);
    void lu(sd::LaunchContext *context, NDArray* input, NDArray* output, NDArray* permutation);
    int determinant(sd::LaunchContext * context, NDArray* input, NDArray* output);
    int logAbsDeterminant(sd::LaunchContext * context, NDArray* input, NDArray* output);

    int inverse(sd::LaunchContext * context, NDArray* input, NDArray* output);
    int upperInverseFunctor(sd::LaunchContext* context, NDArray* input, NDArray* output);
    int lowerInverseFunctor(sd::LaunchContext* context, NDArray* input, NDArray* output);

    bool checkCholeskyInput(sd::LaunchContext * context, NDArray const* input);
    int cholesky(sd::LaunchContext * context, NDArray* input, NDArray* output, bool inplace = false);
    int logdetFunctor(sd::LaunchContext * context, NDArray* input, NDArray* output);
}
}
}
#endif
