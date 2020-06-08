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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//



#include <ops/declarable/helpers/transforms.h>
#include <helpers/Loops.h>
#include <graph/RandomGenerator.h>
#include <numeric>
#include <helpers/ShapeUtils.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
void randomShuffle_(NDArray& input, NDArray& output, sd::graph::RandomGenerator& rng, const bool isInplace) {

    // check edge cases first
    int temp;
    const int firstDim = input.sizeAt(0);
    if(input.lengthOf() == 1 || firstDim == 1) {

        if(!isInplace)
            output.assign(input);
    }
    else if (input.isVector() || shape::isLikeVector(input.shapeInfo(), temp)) {

        // apply Fisher-Yates shuffle
        if(isInplace) {
            //PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance().tadThreshold())
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.relativeInt(i) % i;
                if(i == r)
                    continue;
                T t0 = input.t<T>(i);
                T t1 = input.t<T>(r);
                //math::nd4j_swap<T>(input(i), input(r));
                input.r<T>(i) = t1;
                input.r<T>(r) = t0;
            }
        }
        else {
            std::vector<int> indices(firstDim);
            std::iota(indices.begin(), indices.end(), 0);
            output.p<T>(Nd4jLong(0), input.e<T>(0));

            // FIXME: parallelism!!
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.relativeInt(i) % i;
                output.r<T>(i) = input.t<T>(indices[r]);
                if(i == r)
                    continue;

                output.r<T>(r) = input.t<T>(indices[i]);
                math::nd4j_swap<int>(indices[i], indices[r]);
            }
            rng.rewindH(firstDim-1);
        }
    }
    else {

        // evaluate sub-arrays list of input array through all dimensions excluding first one
        std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input.rankOf(), {0});
        auto subArrsListIn = input.allTensorsAlongDimension(dimensions);

        // apply Fisher-Yates shuffle
        if(isInplace) {
            //PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance().elementwiseThreshold())
            for(int i = firstDim - 1; i > 0; --i) {
                int r = rng.relativeInt(i) % i;

                if(i == r)
                    continue;
                subArrsListIn.at(i)->swapUnsafe(*subArrsListIn.at(r));
            }
        }
        else {
            // evaluate sub-arrays list of output array through all dimensions excluding first one
            auto subArrsListOut = output.allTensorsAlongDimension(dimensions);
            std::vector<int> indices(firstDim);
            std::iota(indices.begin(), indices.end(), 0);
            bool isZeroShuffled = false;
            //PRAGMA_OMP_PARALLEL_FOR_IF((firstDim-1) > Environment::getInstance().tadThreshold())
            for(int i = firstDim - 1; i > 0; --i) {
                int r = rng.relativeInt(i) % i;
                subArrsListOut.at(i)->assign(subArrsListIn.at(indices[r]));
                if(r == 0)
                    isZeroShuffled = true;
                if(i == r)
                    continue;
                subArrsListOut.at(r)->assign(subArrsListIn.at(indices[i]));
                math::nd4j_swap<int>(indices[i], indices[r]);
            }
            if(!isZeroShuffled)
                subArrsListOut.at(0)->assign(subArrsListIn.at(0));
        }
        rng.rewindH(firstDim-1);
    }

}

    void randomShuffle(sd::LaunchContext * context, NDArray& input, NDArray& output, sd::graph::RandomGenerator& rng, const bool isInplace) {
        BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (input, output, rng, isInplace), LIBND4J_TYPES);
    }
}
}
}
