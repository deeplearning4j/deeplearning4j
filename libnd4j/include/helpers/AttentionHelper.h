/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Paul Dubs
//

#ifndef LIBND4J_ATTENTIONHELPER_H
#define LIBND4J_ATTENTIONHELPER_H

#include "array/NDArray.h"

namespace sd {
    class ND4J_EXPORT AttentionHelper {

    public:
        static sd::NDArray multiHeadProject(const sd::NDArray* input, const sd::NDArray* projectionMatrix, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());
        static void multiHeadProjectBp(const sd::NDArray* input, const sd::NDArray* projectionMatrix, const sd::NDArray* eps, sd::NDArray* dLdInput, sd::NDArray* dLdProjectionMatrix, sd::LaunchContext * context = sd::LaunchContext ::defaultContext());
    };
}


#endif
