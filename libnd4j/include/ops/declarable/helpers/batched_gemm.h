/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <vector>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

void bgemm(const std::vector<NDArray*>& vA, const std::vector<NDArray*>& vB, std::vector<NDArray*>& vC, const NDArray* alphas, const NDArray* betas, int transA, int transB, int M, int N, int K, const int lda, const int ldb, const int ldc);

}
}
}