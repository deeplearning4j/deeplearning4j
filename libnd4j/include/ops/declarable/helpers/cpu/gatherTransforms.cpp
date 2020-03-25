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
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <helpers/Loops.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {


////////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
static void gatherND_(NDArray& input, NDArray& indices, NDArray& output) {

    const X* x = reinterpret_cast<X*>(input.getBuffer());
    const Y* y = reinterpret_cast<Y*>(indices.getBuffer());
          X* z = reinterpret_cast<X*>(output.getBuffer());

    const int xRank    = input.rankOf();
    const int yRank    = indices.rankOf();
    const int zRank    = output.rankOf();
    const int maxRank  = sd::math::nd4j_max<int>(yRank, sd::math::nd4j_max<int>(xRank, zRank));

    const Nd4jLong zLen = output.lengthOf();

    const uint yLastDim = indices.sizeAt(-1);

    const int diff = zRank - xRank;
    const bool bEqual = yLastDim == xRank;

    auto func = PRAGMA_THREADS_FOR {

        int xCoords[MAX_RANK], zCoords[MAX_RANK], temp;

        for (auto i = start; i < stop; i++) {

            shape::index2coordsCPU(start, i, output.getShapeInfo(), zCoords);

            const auto zOffset = shape::getOffset(output.getShapeInfo(), zCoords);

            temp = zCoords[yRank - 1];
            zCoords[yRank - 1] = 0;
            const auto yOffset = shape::getOffset(indices.getShapeInfo(), zCoords);
            zCoords[yRank - 1] = temp;

            if(bEqual)
                memcpy(xCoords, zCoords, zRank * sizeof(int));
            else if(diff >= 0)
                memcpy(xCoords, zCoords + diff, xRank * sizeof(int));
            else
                memcpy(xCoords - diff, zCoords, zRank * sizeof(int));

            for (uint j = 0; j < yLastDim; ++j)
                xCoords[j] = y[yOffset + j * indices.stridesOf()[yRank - 1]];   // last stride

            const auto xOffset = shape::getOffset(input.getShapeInfo(), xCoords);

            z[zOffset] = x[xOffset];
        }
    };

    samediff::Threads::parallel_tad(func, 0, zLen);
}

////////////////////////////////////////////////////////////////////////
void gatherND(sd::LaunchContext * context, NDArray& input, NDArray& indices, NDArray& output) {
    BUILD_DOUBLE_SELECTOR(input.dataType(), indices.dataType(), gatherND_, (input, indices, output), LIBND4J_TYPES, INDEXING_TYPES);
}


////////////////////////////////////////////////////////////////////////
template<typename T>
static void gather_(NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {

    int axis = intArgs.size() > 0 ? intArgs[0] : 0;
    const int inputRank = input->rankOf();
    if(axis < 0)
        axis += inputRank;

    const int numOfIntArgs = intArgs.size();

    if (indices != nullptr) {

        for(Nd4jLong i = 0; i < indices->lengthOf(); ++i)
            if(indices->e<Nd4jLong>(i) >= input->sizeAt(axis))
                throw std::runtime_error("helpers::gather function: indices array contains wrong elements, each element must be smaller than corresponding dimension of input array !");

        // first case: indices consist of only one scalar
        if(indices->isScalar()) {
            if(input->rankOf() <= 1){
                //For scalar indices, rank 0 or 1 input: can't do tensor along dimension 0 as this is whole array... instead, we want to get a scalar
				auto idx = indices->e<Nd4jLong>(0);
				auto scalarNDArray = input->e(idx);
                output->assign(scalarNDArray);
            } else {
                auto dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {axis});
                auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimensions);

                auto tadArr = NDArray(reinterpret_cast<void *>(reinterpret_cast<T*>(input->getBuffer()) + tadPack.primaryOffsets()[indices->e<Nd4jLong>(0)]), tadPack.primaryShapeInfo(), output->getContext());
                output->assign(&tadArr);
			}
        }
        else if (input->rankOf() == 1 && indices->isVector()) {
            // special case
            auto func = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e++)
                    output->p(e, input->e<T>(indices->e<Nd4jLong>(e)));
            };

            samediff::Threads::parallel_for(func, 0, indices->lengthOf());
        }
        else {

            std::vector<int> dimsOut(indices->rankOf());
            std::iota(dimsOut.begin(), dimsOut.end(), axis);   // fill with axis, axis+1, ... indices->rankOf()-1
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(output->getShapeInfo(), dimsOut);

            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i++) {
                    NDArray subArrOut = (*output)(i, dimsOut);
                    NDArray subArrIn = (*input)(indices->e<Nd4jLong>(i), {axis});
                    subArrOut.assign(subArrIn);
                }
            };

            samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
        }
    }
    else {

        for(int i = 1; i < numOfIntArgs; ++i)
            if(intArgs[i] >= input->sizeAt(axis))
                throw std::runtime_error("helpers::gather function: some of input indexes is larger than corresponding shape of input array !");

        // we only allow scalar/vector case here
        if (numOfIntArgs == 2) { // scalar case
            output->assign((*input)(intArgs[1], {axis}));
        }
        else { // vector case
            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(output->getShapeInfo(), {axis});

            auto func = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i++) {
                    NDArray subArrOut = (*output)(i, {axis});
                    NDArray subArrIn = (*input)(intArgs[i + 1], {axis});
                    subArrOut.assign(subArrIn);
                }
            };

            samediff::Threads::parallel_tad(func, 0, numOfSubArrs);
        }
    }
}

    void gather(NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {
        BUILD_SINGLE_SELECTOR(input->dataType(), gather_, (input, indices, output, intArgs), LIBND4J_TYPES);
    }

}
}
}
