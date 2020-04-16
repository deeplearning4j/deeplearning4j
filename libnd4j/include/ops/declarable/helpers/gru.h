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

namespace sd {
namespace ops {
namespace helpers {

	void gruCell(sd::LaunchContext * context, const NDArray* x, const NDArray* hLast, const NDArray* Wru, const NDArray* Wc,
				 const NDArray* bru, const NDArray* bc,
				 NDArray* r, NDArray* u, NDArray* c, NDArray* h);

	void gruCell(sd::LaunchContext * context, const NDArray* x, const NDArray* hLast, const NDArray* Wru, const NDArray* Wc, const NDArray* b,
				 NDArray* gates, NDArray* h);

	void gruTimeLoop(sd::LaunchContext * context, const NDArray* x, const NDArray* h0, const NDArray* Wx, const NDArray* Wh, const NDArray* b, NDArray* h);

	void gruCellBp(sd::LaunchContext* context,
              		const NDArray* x,    const NDArray* hLast,
              		const NDArray* W,    const NDArray* Wc,        const NDArray* b,    const NDArray* bc,
              		const NDArray* dLdr, const NDArray* dLdu,      const NDArray* dLdc, const NDArray* dLdh,
              		      NDArray* dLdx,       NDArray* dLdhLast,
              		      NDArray* dLdW,       NDArray* dLdWc,
              		      NDArray* dLdb,       NDArray* dLdbc);

	void gruCellBp(sd::LaunchContext* context,
              const NDArray* x, const NDArray* hI, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* dLdh, const NDArray* gates,
              NDArray* dLdx, NDArray* dLdhI, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb);

	void gruTimeLoopBp(sd::LaunchContext * context,
                    const NDArray* x, const NDArray* hI, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* dLdh,
                    NDArray* dLdx, NDArray* dLdhI, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb);
}
}
}


#endif //LIBND4J_GRU_H