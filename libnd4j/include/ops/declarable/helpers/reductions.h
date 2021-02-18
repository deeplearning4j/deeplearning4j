
/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
 //
 // @author AbdelRauf    (rauf@konduit.ai)
 //

#ifndef LIBND4J_HELPERS_REDUCTIONS_H
#define LIBND4J_HELPERS_REDUCTIONS_H

#include <system/op_boilerplate.h>
#include <math/templatemath.h>
#include <array/NDArray.h>

namespace sd {
    namespace ops {
        namespace helpers {

            void argMax(const NDArray& input, NDArray& output, const std::vector<int>& dimensions);
            void argAbsMax(const NDArray& input, NDArray& output, const std::vector<int>& dimensions);
            void argMin(const NDArray& input, NDArray& output, const std::vector<int>& dimensions);
            void argAbsMin(const NDArray& input, NDArray& output, const std::vector<int>& dimensions);
            
        }
    }
}

#endif