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
#include <samediff/SDVariable.h>
#include <unordered_map>



namespace samediff {

    // general graph management functions
    SameDiff create();


    // basic arithmetic operations
    namespace arithmetic {
        SDVariable Add(SameDiff &sd, SDVariable &x, SDVariable &y, const char *name = nullptr);
        SDVariable Neg(SameDiff &sd, SDVariable &x, const char *name = nullptr);
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
