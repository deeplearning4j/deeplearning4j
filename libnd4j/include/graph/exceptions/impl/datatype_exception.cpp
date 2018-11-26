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
// Created by raver on 11/26/2018.
//

#include <array/DataTypeUtils.h>
#include "../datatype_exception.h"

namespace nd4j {
    datatype_exception::datatype_exception(std::string message) : std::runtime_error(message){
        //
    }

    datatype_exception datatype_exception::build(std::string message, nd4j::DataType expected, nd4j::DataType actual) {
        auto exp = DataTypeUtils::asString(expected);
        auto act = DataTypeUtils::asString(actual);
        message += "; Expected: [" + exp + "]; Actual: [" + act + "]";
        return datatype_exception(message);
    }

    datatype_exception datatype_exception::build(std::string message, nd4j::DataType expected, nd4j::DataType actualX, nd4j::DataType actualY) {
        auto exp = DataTypeUtils::asString(expected);
        auto actX = DataTypeUtils::asString(actualX);
        auto actY = DataTypeUtils::asString(actualY);
        message += "; Expected: [" + exp + "]; Actual: [" + actX + ", " + actY + "]";
        return datatype_exception(message);
    }

    datatype_exception datatype_exception::build(std::string message, nd4j::DataType actual) {
        auto act = DataTypeUtils::asString(actual);
        message += "; Actual: [" + act + "]";
        return datatype_exception(message);
    }
}