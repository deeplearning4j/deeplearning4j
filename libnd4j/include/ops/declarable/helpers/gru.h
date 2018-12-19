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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.02.2018
//

#ifndef LIBND4J_GRU_H
#define LIBND4J_GRU_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {

	void gruCell(const NDArray* x, const NDArray* h0, const NDArray* Wx, const NDArray* Wh, const NDArray* b, NDArray* h);

	void gruTimeLoop(const NDArray* x, const NDArray* h0, const NDArray* Wx, const NDArray* Wh, const NDArray* b, NDArray* h);

	void gruCellBP(const NDArray* x, const NDArray* h0, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* dLdh, const NDArray* dLdWx0, 
                  const NDArray* dLdWh0, const NDArray* dLdb0, NDArray* dLdx, NDArray* dLdh0, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb);

}
}
}


#endif //LIBND4J_GRU_H
