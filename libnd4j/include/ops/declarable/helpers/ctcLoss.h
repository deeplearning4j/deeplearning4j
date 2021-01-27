/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// @author AbdelRauf
//

#ifndef LIBND4J_HELPERS_CTCLOSS_H
#define LIBND4J_HELPERS_CTCLOSS_H

#include <ops/declarable/helpers/helpers.h>
#include <graph/Context.h>

namespace sd    {
namespace ops     {
namespace helpers {


	void ctcLoss(graph::Context& block, const NDArray &logInput, const NDArray &targetLabels, const NDArray &logInputLengths, const NDArray &targetLabelLengths, NDArray &logLosses, NDArray &gradients, int blankIndex);

}
}
}


#endif // LIBND4J_ADDBIAS_H
