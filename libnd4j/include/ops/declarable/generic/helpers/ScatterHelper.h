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

                std::vector<int> tadDimension = ShapeUtils<T>::convertAxisToTadTarget(input->rankOf(), {0});
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

#pragma omp parallel for if(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
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

                std::vector<Nd4jLong> idxRangesOut(2 * outRank);
                std::vector<Nd4jLong> idxRangesUpd(2 * updRank);

// #pragma omp parallel for if(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) firstprivate(idxRangesOut, idxRangesUpd)  // causes known openMP asan bug !
#pragma omp parallel for schedule(guided) firstprivate(idxRangesOut, idxRangesUpd)
                for(Nd4jLong i = 0; i < indLen; ++i) {                    

                    ShapeUtils<T>::evalIdxRangesForSubArr(static_cast<Nd4jLong>(indices(i)), output.getShapeInfo(), {0}, idxRangesOut.data());
                    ShapeUtils<T>::evalIdxRangesForSubArr(i, updates.getShapeInfo(), dimsToExcludeUpd, idxRangesUpd.data());

                    NDArray<T> outSubArr = output(idxRangesOut);                
                    NDArray<T> updSubArr = updates(idxRangesUpd);
 #pragma omp critical
                    outSubArr.template applyPairwiseTransform<OpClass>(&updSubArr, nullptr);
                }
            }
        }

////////////////////////////////////////////////////////////////////////
template <typename OpClass>
static FORCEINLINE void scatterND(const NDArray<T>& indices, const NDArray<T>& updates, NDArray<T>& output) {   

    const Nd4jLong indLen = indices.lengthOf();
    const Nd4jLong updLen = updates.lengthOf();
    const int outRank = output.rankOf();
    const int indRank = indices.rankOf();
    const int updRank = updates.rankOf();
    const Nd4jLong indLastDim = indices.sizeAt(-1);

    if(outRank == 1) {

#pragma omp parallel for if(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i = 0; i < indLen; ++i) {
            T& out = output(indices(i));                    
#pragma omp critical                    
            out = OpClass::op(out, updates(i), nullptr);
        }
    } 
    else if(outRank == indLastDim) {

        std::vector<int> dimsToExcludeInd = ShapeUtils<T>::evalDimsToExclude(indRank, {indRank-1});
        std::vector<Nd4jLong> outIdx(outRank);        
        std::vector<Nd4jLong> idxRangesInd(2 * indRank);

#pragma omp parallel for schedule(guided) firstprivate(outIdx, idxRangesInd)
        for(Nd4jLong i = 0; i < updLen; ++i) {            

            ShapeUtils<T>::evalIdxRangesForSubArr(i, indices.getShapeInfo(), dimsToExcludeInd, idxRangesInd.data());
            NDArray<T> indSubArr = indices(idxRangesInd);
#pragma omp simd
            for(Nd4jLong j = 0; j < indLastDim; ++j)
                outIdx[j] = indSubArr(j);
// #pragma omp critical
            // output(outIdx) = updates(i);
        }
    }
    else {

        int sizeExludeOut = std::is_same<OpClass, simdOps::Copy<T>>::value ? 1 : indices.sizeAt(-1) - 1;    

        std::vector<int> dimsToExcludeUpd(outRank - indLastDim);
        std::vector<int> dimsToExcludeOut(sizeExludeOut);
        std::iota(dimsToExcludeUpd.begin(), dimsToExcludeUpd.end(), 0);
        std::iota(dimsToExcludeOut.begin(), dimsToExcludeOut.end(), 0);

        std::vector<Nd4jLong> idxRangesOut(2 * outRank);
        std::vector<Nd4jLong> idxRangesUpd(2 * updRank);
// #pragma omp parallel for if(indLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) firstprivate(idxRangesOut, idxRangesUpd)  // causes known openMP asan bug !
#pragma omp parallel for schedule(guided) firstprivate(idxRangesOut, idxRangesUpd)
        for(Nd4jLong i = 0; i < indLen; ++i) {
        
            ShapeUtils<T>::evalIdxRangesForSubArr(static_cast<Nd4jLong>(indices(i)), output.getShapeInfo(), dimsToExcludeOut, idxRangesOut.data());
            ShapeUtils<T>::evalIdxRangesForSubArr(i, updates.getShapeInfo(), dimsToExcludeUpd, idxRangesUpd.data());

            NDArray<T> outSubArr = output(idxRangesOut);
            NDArray<T> updSubArr = updates(idxRangesUpd);

#pragma omp critical
            outSubArr.template applyPairwiseTransform<OpClass>(&updSubArr, nullptr);
        }
    }
}


};


}
}