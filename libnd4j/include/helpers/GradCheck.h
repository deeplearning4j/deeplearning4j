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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.07.2018
//

#ifndef LIBND4J_GRADCHECK_H
#define LIBND4J_GRADCHECK_H


#include <NDArray.h>
#include <ops/declarable/DeclarableOp.h>

namespace nd4j {

class GradCheck {

    public:        
        enum LossFunc {MEAN = 0, SUM = 1};    
    private:
        static constexpr double EPSILON = 1e-5;
        static constexpr double MAXRELERR = 1e-5;
        static constexpr double MINABSERR = 1e-6;
        static void fillGradArrays(const LossFunc loss, const std::vector<NDArray*>& gradArrs);

    
    public:        
        
        /** 
        *  performs numerical check of gradients in back prop
        * 
        *  opFF - feed forward operation
        *  opBP - back propagation operation
        *  argsHolderFF - argument holder for feed forward operation
        *  argsHolderBP - argument holder for back propagation operation
        *  whatArrsToCheck - specifies what output gradient arrays to check, for example {0, 1, 0} means that only second output gradient array will be checked, default value is empty array which means to check all arrays
        *  IdxRange - specifies indexes range over which array elements will be checked, for example {0.2, 0.7} means range [0.2*array_length, 0.7*array_length), default value is {0., 1.}
        *  loss - type of scalar loss function, it specifies what elements values will be filled into input gradient arrays automatically, default value is SUM
        */
        static bool checkGrad(ops::DeclarableOp& opFF, ops::DeclarableOp& opBP, const OpArgsHolder& argsHolderFF, const OpArgsHolder& argsHolderBP,
                              const std::vector<bool>& whatArrsToCheck = std::vector<bool>(), const std::vector<double>& IdxRange = {0., 1.}, const LossFunc loss = SUM);

};





// //////////////////////////////////////////////////////////////////////////
// ///// IMLEMENTATION OF INLINE METHODS ///// 
// //////////////////////////////////////////////////////////////////////////

// template<typename T>
// FORCEINLINE bool ShapeUtils::isPermutNecessary(const std::vector<int>& permut) {

//     for(int i=0; i<permut.size(); ++i)
//         if(permut[i] != i)
//             return true;

//     return false;
// }



}

#endif //LIBND4J_GRADCHECK_H
