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

#include "NDArray.h"

namespace nd4j {
    class AttentionHelper {

    public:
        static nd4j::NDArray* multiHeadProject(const nd4j::NDArray* input, const nd4j::NDArray* projectionMatrix, nd4j::memory::Workspace* workspace = nullptr);
        static void multiHeadProjectBp(const nd4j::NDArray* input, const nd4j::NDArray* projectionMatrix, const nd4j::NDArray* eps, nd4j::NDArray* dLdInput, nd4j::NDArray* dLdProjectionMatrix, nd4j::memory::Workspace* workspace = nullptr);
    };
}


#endif
