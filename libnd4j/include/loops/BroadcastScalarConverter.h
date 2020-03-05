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

/**
 * @author raver119@gmail.com
 */
#ifndef DEV_TESTS_BROADCASTSCALARCONVERTER_H
#define DEV_TESTS_BROADCASTSCALARCONVERTER_H

#include <system/op_boilerplate.h>
#include <stdexcept>

namespace sd {
    inline bool isConvertibleToScalar(broadcast::Ops op) {
        int opNum = (int) op;

        if (opNum <= 17)
            return true;

        return false;
    }

    inline scalar::Ops convertToScalar(broadcast::Ops op) {
        switch (op) {
            case broadcast::Add: return scalar::Add;
            case broadcast::Subtract: return scalar::Subtract;
            case broadcast::Multiply: return scalar::Multiply;
            case broadcast::Divide: return scalar::Divide;
            case broadcast::ReverseDivide: return scalar::ReverseDivide;
            case broadcast::ReverseSubtract: return scalar::ReverseSubtract;
            case broadcast::CopyPws: return scalar::CopyPws;
            case broadcast::Pow: return scalar::Pow;
            case broadcast::MinPairwise: return scalar::MinPairwise;
            case broadcast::MaxPairwise: return scalar::MaxPairwise;
            case broadcast::AMinPairwise: return scalar::AMinPairwise;
            case broadcast::AMaxPairwise: return scalar::AMaxPairwise;
            case broadcast::SquaredSubtract: return scalar::SquaredSubtract;
            default:
                throw std::runtime_error("Not convertible operation");
        }
    }
}

#endif //DEV_TESTS_BROADCASTSCALARCONVERTER_H
