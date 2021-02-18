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
#include <helpers/ShapeUtils.h>
#include <helpers/Loops.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
void scatterUpdate(sd::LaunchContext * context, NDArray& input, NDArray& updates, const std::vector<int>* intArgs) {

    int opCode = (*intArgs)[0];
    int dimSize = (*intArgs)[1];
    Nd4jLong e;
    Nd4jLong limg = 2 + dimSize;
    std::vector<int> tadDimensions(dimSize);
    for (e = 2; e < limg; e++)
        tadDimensions[e-2] = (*intArgs)[e];

    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(input.rankOf(), tadDimensions);

    // increasing counter to skip numIndices
    e++;
    std::vector<int> indices;
    for (; e < static_cast<Nd4jLong>(intArgs->size()); e++)
        indices.push_back((*intArgs)[e]);

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            auto inSubArr = input(indices[i], dimsToExclude, true);
            auto updSubArr = updates(i, dimsToExclude, true);

            if (inSubArr.lengthOf() != updSubArr.lengthOf())
                continue;

            switch (opCode) {
                case 0:
                    inSubArr.applyPairwiseTransform(pairwise::Add, updSubArr, inSubArr);
                    break;
                case 1:
                    inSubArr.applyPairwiseTransform(pairwise::Subtract, updSubArr, inSubArr);
                    break;
                case 2:
                    inSubArr.applyPairwiseTransform(pairwise::Multiply, updSubArr, inSubArr);
                    break;
                case 3:
                    inSubArr.applyPairwiseTransform(pairwise::Divide, updSubArr, inSubArr);
                    break;
                case 4:
                    inSubArr.applyPairwiseTransform(pairwise::ReverseSubtract, updSubArr, inSubArr);
                    break;
                case 5:
                    inSubArr.applyPairwiseTransform(pairwise::ReverseDivide, updSubArr, inSubArr);
                    break;
                case 6:
                    inSubArr.applyPairwiseTransform(pairwise::CopyPws, updSubArr, inSubArr);
                    break;
                default:
                    continue;
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, indices.size());
}


//////////////////////////////////////////////////////////////////////////
void scatterSimple(sd::LaunchContext * context, const int opId, NDArray& input, const NDArray& updates, const NDArray& indices, const std::vector<int>& dimensions) {

    // updates and indices have same length
    const Nd4jLong len = indices.lengthOf();

    switch (opId) {

        case 6: {   // copy
            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i++) {
                    auto inSubArr = input(i, dimensions);
                    inSubArr.p(indices.t<Nd4jLong>(i), updates.e(i));
                }
            };

            samediff::Threads::parallel_for(func, 0, len);
        }
            break;

        default:
            throw std::invalid_argument("helpers::scatterSimple: operation is not implemented for given id !");
    }
}

}
}
}
