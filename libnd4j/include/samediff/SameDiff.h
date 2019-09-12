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
// @author raver119@gmail.com
//

#ifndef SAMEDIFF_SAMEDIFF_H
#define SAMEDIFF_SAMEDIFF_H

#include <NDArray.h>
#include "SDVariable.h"

using namespace nd4j;

namespace samediff {

    class SameDiff {
    public:
        SameDiff() = default;
        ~SameDiff() = default;


        SDVariable variable(const char *name, const nd4j::NDArray &array);
        SDVariable placeholder(const char *name, const nd4j::DataType dataType = nd4j::DataType::FLOAT32, const std::vector<Nd4jLong> shape = {-1});
    };
}

#endif //SAMEDIFF_SAMEDIFF_H
