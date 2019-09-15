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

#ifndef SAMEDIFF_SAMEDIFF_CPP_H
#define SAMEDIFF_SAMEDIFF_CPP_H

#include <NDArray.h>
#include <samediff/SameDiff.h>
#include <samediff/Variable.h>
#include <samediff/Tuple.h>

#include <vector>
#include <string>
#include <unordered_map>


namespace samediff {

    // general graph management functions
    SameDiff create();


    // basic arithmetic operations
    namespace arithmetic {
        Variable Add(const Variable &x, const Variable &y, const std::string &name = {});
        Variable Neg(const Variable &x, const std::string &name = {});
    }

    namespace transform {
        Tuple Tear(const Variable &x, const std::vector<int> &axis, const std::string &name = {});
    }

    // math functions
    namespace math {
        //void cos();
        //void sin();
    }

    // nn-related functions
    namespace nn {
        //void convolution2d();
        //void avgpooling2d();
    }
}

#endif //SAMEDIFF_SAMEDIFF_CPP_H
