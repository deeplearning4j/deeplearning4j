/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_SCATTERHELPER_H
#define LIBND4J_SCATTERHELPER_H

#include <pointercast.h>
#include <op_boilerplate.h>
#include <NDArray.h>
#include <numeric>


namespace nd4j {
namespace ops {

class ScatterHelper {
    
    public:

////////////////////////////////////////////////////////////////////////
        static FORCEINLINE void scatter(pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

            const int outRank = output.rankOf();
            const int indRank = indices.rankOf();
            const int updRank = updates.rankOf();
            const Nd4jLong indLen = indices.lengthOf();

            if(outRank == 1) {

                PRAGMA_OMP_PARALLEL_FOR_IF(!lock)
                for(Nd4jLong i = 0; i < indLen; ++i) {
                    
                    Nd4jLong idx = indices.e<Nd4jLong>(i);
                    NDArray out = output({idx, idx+1});

                    PRAGMA_OMP_CRITICAL
                    {
                        out.applyPairwiseTransform(op, updates.e(i), nullptr);
                    }
                }
            }
            else {      // outRank > 1

                int sizeOfDims = indRank;
                if(outRank == updRank && indices.isVector())
                    sizeOfDims = 1;

                std::vector<int> dimsToExcludeUpd(sizeOfDims);
                std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);

                PRAGMA_OMP_PARALLEL_FOR_IF(!lock)
                for(Nd4jLong i = 0; i < indLen; ++i) {                                       

                    NDArray outSubArr = output(indices.e<Nd4jLong>(i), std::vector<int>({0}));
                    NDArray updSubArr = updates(i, dimsToExcludeUpd);

                    PRAGMA_OMP_CRITICAL
                    {
                        outSubArr.applyPairwiseTransform(op, updSubArr, nullptr);
                    }
                }
            }
        }


////////////////////////////////////////////////////////////////////////
static FORCEINLINE void scatterND(pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

    const Nd4jLong indLen = indices.lengthOf();
    const int outRank = output.rankOf();
    const int indRank = indices.rankOf();
    const Nd4jLong indLastDim = indices.sizeAt(-1);

    if(outRank == 1) {

        PRAGMA_OMP_PARALLEL_FOR_IF(!lock)
        for(Nd4jLong i = 0; i < indLen; ++i) {

            auto idx = indices.e<Nd4jLong>(i);
            auto out = output({idx, idx+1});
            PRAGMA_OMP_CRITICAL
            {
                out.applyPairwiseTransform(op, updates.e(i), nullptr);
            }
        }
    } 
    else {
        
        ResultSet indSubArrs = indices.allTensorsAlongDims({indRank-1});
        std::vector<int> dimsToExcludeUpd(indRank - 1);
        std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);
        std::vector<Nd4jLong> idxRangeOut(2*outRank, 0);        

        PRAGMA_OMP_PARALLEL_FOR_ARGS(if(!lock) firstprivate(idxRangeOut))
        for(Nd4jLong i = 0; i < indLen/indLastDim; ++i) {

            for(Nd4jLong j = 0; j < indLastDim; ++j) {
                idxRangeOut[2*j] = indSubArrs[i]->e<Nd4jLong>(j);
                idxRangeOut[2*j + 1] = idxRangeOut[2*j] + 1;
            }

            auto outSubArr = output(idxRangeOut);
            auto updSubArr = updates(i, dimsToExcludeUpd);

            PRAGMA_OMP_CRITICAL
            {
                outSubArr.applyPairwiseTransform(op, updSubArr, nullptr);
            }
        }        
    }
}


////////////////////////////////////////////////////////////////////////
static FORCEINLINE void scatterForLoss(const NDArray& indices, const NDArray& updates, NDArray& output, const bool calcGrad) {

    // requirements for arrays
    // shapes of updates and output must be the same
    // shape of indices should be the same as updates shape with last dimension excluded
    // for example if updates is {a,b,c} then indices should be {a,b}

    const Nd4jLong indicesLen = indices.lengthOf();

    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(updates.rankOf(), {-1});

    if(!calcGrad) {
        PRAGMA_OMP_PARALLEL_FOR
        for(Nd4jLong i = 0; i < indicesLen; ++i) {

            auto subArr = updates(i, dimsToExclude);
            output.p(i, subArr.e(indices.e<Nd4jLong>(i)));
        }
    }
    else {
        PRAGMA_OMP_PARALLEL_FOR
        for(Nd4jLong i = 0; i < indicesLen; ++i) {

            auto subArr = updates(i, dimsToExclude);
            auto ind = indices.e<Nd4jLong>(i);
            subArr.p(ind, subArr.e(ind) - 1.);
        }   
    }
}

};


}
}

#endif