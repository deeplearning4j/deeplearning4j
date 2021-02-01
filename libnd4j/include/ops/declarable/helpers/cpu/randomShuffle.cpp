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
// implementation is based on following article:
// "MergeShuffle: A Very Fast, Parallel Random Permutation Algorithm", https://arxiv.org/abs/1508.03167



#include <ops/declarable/helpers/transforms.h>
#include <helpers/Loops.h>
#include <graph/RandomGenerator.h>
#include <numeric>
#include <helpers/ShapeUtils.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// Fisher-Yates shuffle
template <typename T>
static void fisherYates(sd::graph::RandomGenerator& rng, T* buff, const Nd4jLong& len, const Nd4jLong& ews, Nd4jLong ind) {

    for(Nd4jLong i = len-1; i > 0; --i) {
        const Nd4jLong j = rng.relativeLong(ind++) % (i + 1);
        if(i != j)
            math::nd4j_swap<T>(buff[i*ews], buff[j*ews]);
    }
}

//////////////////////////////////////////////////////////////////////////
// mutual shuffle of two adjacent already shuffled ranges with length len1 and (totLen - len1) correspondingly
template <typename T>
static void mergeShuffle(sd::graph::RandomGenerator& rng, T* buff, const Nd4jLong& len1, const Nd4jLong& totLen, const Nd4jLong& ews, Nd4jLong ind) {

    Nd4jLong beg = 0;           // beginning
    Nd4jLong mid = len1;        // middle

    while (true) {
        if(rng.relativeLong(ind++) % 2) {
            if(mid == totLen)
                break;
            math::nd4j_swap<T>(buff[ews * beg], buff[ews * mid++]);
        } else {
            if(beg == mid)
                break;
        }
        ++beg;
    }

    // fisherYates
    while (beg < totLen) {
        const Nd4jLong j = rng.relativeLong(ind++) % (beg + 1);
        if(beg != j)
            math::nd4j_swap<T>(buff[ews * beg], buff[ews * j]);
        ++beg;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void randomShuffle_(NDArray& input, NDArray& output, sd::graph::RandomGenerator& rng, const bool isInplace) {

    const int firstDim = input.sizeAt(0);
    int temp;

    if(input.lengthOf() == 1 || firstDim == 1) {

        if(!isInplace)
            output.assign(input);
    }
    else if (shape::isCommonVector(input.shapeInfo(), temp)) {

        NDArray* arr = &input;

        if (!isInplace) {
            output.assign(input);
            arr = &output;
        }

        const Nd4jLong ews = arr->ews();

        const Nd4jLong len       = arr->lengthOf();
        const Nd4jLong threshold = 1<<22;  // this number was deduced from diagram in article

        int power = 0;
        while ((len >> power) > threshold)
            ++power;

        const Nd4jLong numChunks = 1 << power;

        auto funcFisherYates = PRAGMA_THREADS_FOR {

            for (auto i = start; i < stop; ++i) {

                Nd4jLong offset = (len * i) >> power;
                Nd4jLong currLen = ((len * (i + 1)) >> power) - offset;
                fisherYates<T>(rng, arr->bufferAsT<T>() + offset*ews, currLen, ews, offset);
            }
        };

        auto funcMerge = PRAGMA_THREADS_FOR {

            for (int64_t i = start, k = 1; i < stop; i += increment, ++k) {
                Nd4jLong offset = len * i >> power;
                Nd4jLong len1   = (len * (i + increment/2) >> power) - offset;
                Nd4jLong totLen = (len * (i + increment)   >> power) - offset;
                mergeShuffle<T>(rng, arr->bufferAsT<T>() + offset*ews, len1, totLen, ews, len * k + offset);
            }
        };

        samediff::Threads::parallel_for(funcFisherYates, 0, numChunks);

        for (int j = 1; j < numChunks; j += j)
            samediff::Threads::parallel_for(funcMerge, 0, numChunks, 2*j);

        // #pragma omp parallel for
        // for (uint i = 0; i < numChunks; ++i) {

        //     Nd4jLong offset = (len * i) >> power;
        //     Nd4jLong currLen = ((len * (i + 1)) >> power) - offset;
        //     fisherYates<T>(rng, arr->bufferAsT<T>() + offset*ews, currLen, ews, offset);
        // }

        // for (uint j = 1; j < numChunks; j += j) {
        //     #pragma omp parallel for
        //     for (auto i = 0; i < numChunks; i += 2*j) {
        //         Nd4jLong offset = len * i >> power;
        //         Nd4jLong len1   = (len * (i + j) >> power) - offset;
        //         Nd4jLong totLen = (len * (i + 2*j)   >> power) - offset;
        //         mergeShuffle(rng, arr->bufferAsT<T>() + offset*ews, len1, totLen, ews, len * j + offset);
        //     }
        // }

        rng.rewindH((len + 1) * power);
    }
    else {

        auto dimsToExclude = ShapeUtils::evalDimsToExclude(input.rankOf(), {0});

        if(isInplace) {

            auto subArrsList = input.allTensorsAlongDimension(dimsToExclude);

            // Fisher-Yates shuffle
            for(int i = firstDim - 1; i > 0; --i) {
                const int j = rng.relativeInt(i) % (i + 1);
                if(i != j)
                    subArrsList.at(i)->swapUnsafe(*subArrsList.at(j));
            }
        }
        else {

            auto subArrsListIn  = input.allTensorsAlongDimension(dimsToExclude);
            auto subArrsListOut = output.allTensorsAlongDimension(dimsToExclude);

            std::vector<int> indices(firstDim);
            std::iota(indices.begin(), indices.end(), 0);   // 0,1,2,3, ... firstDim-1

            // shuffle indices
            fisherYates<int>(rng, indices.data(), firstDim, 1, 0);

            auto func = PRAGMA_THREADS_FOR {

                for (auto i = start; i < stop; ++i)
                    subArrsListOut.at(i)->assign(subArrsListIn.at(indices[i]));
            };

            samediff::Threads::parallel_for(func, 0, firstDim);
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

