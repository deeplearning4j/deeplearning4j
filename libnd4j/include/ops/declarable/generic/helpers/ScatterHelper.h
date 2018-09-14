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
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include <pointercast.h>
#include <op_boilerplate.h>
#include <NDArray.h>
#include <numeric>


namespace nd4j {
namespace ops {

template <typename T>
class ScatterHelper {
    
    public:

        template <typename OpClass>
        static FORCEINLINE Nd4jStatus scatterApply(NDArray<T>* output, NDArray<T>* indices, NDArray<T>* updates) {
            
            NDArray<T>* input = output;
            int indicesLength = (int) indices->lengthOf();

            if ((indices->isVector() && input->isVector() && updates->isVector()) ||
                (input->isScalar() && input->isScalar() && updates->isScalar()) ||
                (input->isVector() && indices->isScalar() && updates->isScalar()) ) {
                
                for (int e = 0; e < indicesLength; e++) {
                    int idx = (int) indices->getScalar(e);
                    
                    T t0 = input->getScalar(idx);
                    T t1 = updates->getScalar(e);
                    
                    output->putScalar(idx, OpClass::op(t0, t1, nullptr));
                }

                return Status::OK();
            } else if (indices->isVector() || indices->isScalar()) {
                std::vector<int> idc;
                std::vector<int> idcU;

                for (int e = 0; e < indicesLength; e++) {
                    idc.push_back((int) indices->getScalar(e));
                    idcU.push_back(e);
                }

                std::vector<int> tadDimension = ShapeUtils::convertAxisToTadTarget(input->rankOf(), {0});
                auto tadsOperand = output->multipleTensorsAlongDimension(idc, tadDimension);
                auto tadsUpdate = updates->multipleTensorsAlongDimension(idcU, tadDimension);

                auto z0 = tadsOperand->at(0);
                auto z1 = tadsUpdate->at(0);

                REQUIRE_TRUE(z0->isSameShape(z1), 0, "scatter_add: updates shapes should match");

                for (int e = 0; e < tadsOperand->size(); e++) {
                    auto t0 = tadsOperand->at(e);
                    auto t1 = tadsUpdate->at(e);
                    
                    t0->template applyPairwiseTransform<OpClass>(t1, nullptr);
                }

                delete tadsOperand;
                delete tadsUpdate;

                return Status::OK();
            }  else if (indices->isMatrix() || indices->rankOf() >= 2) {
                auto _input = input->reshape(input->ordering(), {input->sizeAt(0), -1});
                auto _updates = updates->reshape(updates->ordering(), {indicesLength, (int) updates->lengthOf() / indicesLength});

                auto tadsOperand = _input->allTensorsAlongDimension({1});
                auto tadsUpdates = _updates->allTensorsAlongDimension({1});

                for (int e = 0; e < indicesLength; e++) {
                    int idx = indices->getScalar(e);
                    
                    auto t0 = tadsOperand->at(idx);
                    auto t1 = tadsUpdates->at(e);

                    t0->template applyPairwiseTransform<OpClass>(t1, nullptr);
                }

                delete _input;
                delete _updates;

                delete tadsOperand;
                delete tadsUpdates;
                return Status::OK();
            }

                return Status::THROW("ScatterHelper failed");
        }

////////////////////////////////////////////////////////////////////////
        template <typename OpClass>
        static FORCEINLINE void scatter(const NDArray<T>& indices, const NDArray<T>& updates, NDArray<T>& output) {

            const int outRank = output.rankOf();
            const int indRank = indices.rankOf();
            const int updRank = updates.rankOf();
            const Nd4jLong indLen = indices.lengthOf();

            if(outRank == 1) {

// #pragma omp parallel for if(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
#pragma omp parallel for schedule(guided)
                for(Nd4jLong i = 0; i < indLen; ++i) {
                    T& out = output(indices(i));                    
#pragma omp critical                    
                    out = OpClass::op(out, updates(i), nullptr);
                }
            }
            else {      // outRank > 1

                int sizeOfDims = indRank;
                if(outRank == updRank && indices.isVector())
                    sizeOfDims = 1;
                
                std::vector<int> dimsToExcludeUpd(sizeOfDims);
                std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);

// #pragma omp parallel for if(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) // causes known openMP asan bug !
// #pragma omp parallel for schedule(guided)
                for(Nd4jLong i = 0; i < indLen; ++i) {                                       

                    NDArray<T> outSubArr = output(indices(i), std::vector<int>({0}));
                    NDArray<T> updSubArr = updates(i, dimsToExcludeUpd);

#pragma omp critical
                    outSubArr.template applyPairwiseTransform<OpClass>(&updSubArr, nullptr);
                }
            }
        }

////////////////////////////////////////////////////////////////////////
template <typename OpClass>
static FORCEINLINE void scatterND(const NDArray<T>& indices, const NDArray<T>& updates, NDArray<T>& output) {   

    const Nd4jLong indLen = indices.lengthOf();
    const int outRank = output.rankOf();
    const int indRank = indices.rankOf();
    const Nd4jLong indLastDim = indices.sizeAt(-1);

    if(outRank == 1) {

// #pragma omp parallel for if(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
#pragma omp parallel for schedule(guided)        
        for(Nd4jLong i = 0; i < indLen; ++i) {

            T& elemOut = output(indices(i));                    
#pragma omp critical                    
            elemOut = OpClass::op(elemOut, updates(i), nullptr);
        }
    } 
    else {

        std::vector<int> dimsToExcludeInd = ShapeUtils::evalDimsToExclude(indRank, {indRank-1});
        std::vector<int> dimsToExcludeUpd(indRank - 1);
        std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);
        std::vector<Nd4jLong> idxRangeOut(2*outRank, 0);
 
// #pragma omp parallel for if(indLen/indLastDim > Environment::getInstance()->elementwiseThreshold()) schedule(guided) firstprivate(idxRangeOut)
#pragma omp parallel for schedule(guided) firstprivate(idxRangeOut)
        for(Nd4jLong i = 0; i < indLen/indLastDim; ++i) {
            
            NDArray<T> indSubArr = indices(i, dimsToExcludeInd);

            for(Nd4jLong j = 0; j < indLastDim; ++j) {
                idxRangeOut[2*j] = indSubArr(j);
                idxRangeOut[2*j + 1] = idxRangeOut[2*j] + 1;
            }

            NDArray<T> outSubArr = output(idxRangeOut);
            NDArray<T> updSubArr = updates(i, dimsToExcludeUpd);

#pragma omp critical                             
            outSubArr.template applyPairwiseTransform<OpClass>(&updSubArr, nullptr);
        }        
    }
}


};


}
}
