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

#include <exceptions/allocation_exception.h>
#include <helpers/StringUtils.h>

namespace sd {
    allocation_exception::allocation_exception(std::string message) : std::runtime_error(message){
        //
    }

    allocation_exception allocation_exception::build(std::string message, Nd4jLong numBytes) {
        auto bytes = StringUtils::valueToString<Nd4jLong>(numBytes);
        message += "; Requested bytes: [" + bytes + "]";
        return allocation_exception(message);
    }

    allocation_exception allocation_exception::build(std::string message, Nd4jLong limit, Nd4jLong numBytes) {
        auto bytes = StringUtils::valueToString<Nd4jLong>(numBytes);
        auto lim = StringUtils::valueToString<Nd4jLong>(limit);
        message += "; Limit bytes: [" + lim + "]; Requested bytes: [" + bytes + "]";
        return allocation_exception(message);
    }
}