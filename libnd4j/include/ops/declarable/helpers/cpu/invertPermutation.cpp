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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include <ops/declarable/helpers/transforms.h>
#include <helpers/Loops.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

////////////////////////////////////////////////////////////////////////
void invertPermutation(sd::LaunchContext * context, const NDArray& input, NDArray& output) {

    std::set<int> uniqueElems;
    const int length = input.lengthOf();

    for(int i = 0; i < length; ++i) {

        int elem = input.e<int>(i);

        if(!uniqueElems.insert(elem).second)        // this operation forbids us to use #pragma omp
            throw std::runtime_error("helpers::invertPermutation function: input array contains duplicates !");

        if(elem < 0 || elem > length - 1)
            throw  std::runtime_error("helpers::invertPermutation function: element of input array is out of range (0, length-1) !");

        output.p<int>(elem, i);
    }
}

}
}
}
