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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/scatter.h>
#include <numeric>
#include <helpers/ShapeUtils.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

void scatter(nd4j::LaunchContext  *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

    const int outRank = output.rankOf();
    const int indRank = indices.rankOf();
    const int updRank = updates.rankOf();
    const Nd4jLong indLen = indices.lengthOf();

    if(outRank == 1) {

// PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided))
PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(!lock) schedule(guided))
        for(Nd4jLong i = 0; i < indLen; ++i) {

            Nd4jLong idx = indices.e<Nd4jLong>(i);
            NDArray out = output({idx, idx+1});

            out.applyPairwiseTransform(op, updates.e(i), nullptr);
        }
    }
    else {      // outRank > 1

        int sizeOfDims = indRank;
        if(outRank == updRank && indices.isVector())
            sizeOfDims = 1;

        std::vector<int> dimsToExcludeUpd(sizeOfDims);
        std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);

        shape::printIntArray(dimsToExcludeUpd.data(),dimsToExcludeUpd.size());

// PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)) // causes known openMP asan bug !
PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(!lock) schedule(guided))
        for(Nd4jLong i = 0; i < indLen; ++i) {

            NDArray outSubArr = output(indices.e<Nd4jLong>(i), std::vector<int>({0}));
            NDArray updSubArr = updates(i, dimsToExcludeUpd);

            outSubArr.applyPairwiseTransform(op, updSubArr, nullptr);
        }
    }
}

///////////////////////////////////////////////////////////////////
void scatterND(nd4j::LaunchContext  *context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock) {

    const Nd4jLong indLen = indices.lengthOf();
    const int outRank = output.rankOf();
    const int indRank = indices.rankOf();
    const Nd4jLong indLastDim = indices.sizeAt(-1);

    if(outRank == 1) {

// PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided))
PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(!lock) schedule(guided))
        for(Nd4jLong i = 0; i < indLen; ++i) {

            Nd4jLong idx = indices.e<Nd4jLong>(i);
            NDArray out = output({idx, idx+1});

            out.applyPairwiseTransform(op, updates.e(i), nullptr);
        }
    }
    else {

        std::vector<int> dimsToExcludeInd = ShapeUtils::evalDimsToExclude(indRank, {indRank-1});
        std::vector<int> dimsToExcludeUpd(indRank - 1);
        std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);
        std::vector<Nd4jLong> idxRangeOut(2*outRank, 0);

// PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(indLen/indLastDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided) firstprivate(idxRangeOut))
PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(!lock) schedule(guided) firstprivate(idxRangeOut))
        for(Nd4jLong i = 0; i < indLen/indLastDim; ++i) {

            NDArray indSubArr = indices(i, dimsToExcludeInd);

            for(Nd4jLong j = 0; j < indLastDim; ++j) {
                idxRangeOut[2*j] = indSubArr.e<Nd4jLong>(j);
                idxRangeOut[2*j + 1] = idxRangeOut[2*j] + 1;
            }

            NDArray outSubArr = output(idxRangeOut);
            NDArray updSubArr = updates(i, dimsToExcludeUpd);

            outSubArr.applyPairwiseTransform(op, updSubArr, nullptr);
        }
    }
}

void scatterForLoss(nd4j::LaunchContext  *context, const NDArray& indices, NDArray& updates, NDArray& output, const bool calcGrad) {

    // shapes of indices and output must be the same
    // shape of indices should be the same as updates shape with last dimension excluded
    // for example if updates is {a,b,c} then indices should be {a,b}

    const Nd4jLong indicesLen = indices.lengthOf();

    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(updates.rankOf(), {-1});

    if(!calcGrad) {
PRAGMA_OMP_PARALLEL_FOR_ARGS(schedule(guided))
        for(Nd4jLong i = 0; i < indicesLen; ++i) {

            auto subArr = updates(i, dimsToExclude);
            output.p(i, subArr.e(indices.e<Nd4jLong>(i)));
        }
    } else {
PRAGMA_OMP_PARALLEL_FOR_ARGS(schedule(guided))
		for(Nd4jLong i = 0; i < indicesLen; ++i) {

            auto subArr = updates(i, dimsToExclude);
            auto ind = indices.e<Nd4jLong>(i);
            subArr.p(ind, subArr.e(ind) - 1.);
        }
    }
}

}
}
}
