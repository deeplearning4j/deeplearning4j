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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 07.03.2019
//


#include <ops/declarable/helpers/gather.h>
#include <numeric>
#include <PointersManager.h>

namespace nd4j    {
namespace ops     {
namespace helpers {         


//////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
__global__ static void gatherCuda(const int numOfSubArrs, 
                                    const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xOffsets,
                                    const void* vy, const Nd4jLong* yShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zOffsets) {

    const Y* y = reinterpret_cast<const Y*>(vy);
    __shared__ const X* x;
    __shared__ Z* z;
    
    const Nd4jLong len = shape::length(xShapeInfo);

    for (int i = blockIdx.x; i < numOfSubArrs; i += gridDim.x) {
        
        if (threadIdx.x == 0) {
                        
            x = reinterpret_cast<const X*>(vx) + xOffsets[ y[shape::getIndexOffset(i, yShapeInfo, numOfSubArrs)] ];
            z = reinterpret_cast<Z*>(vz) + zOffsets[i];
        }
        __syncthreads();

        for (int j = threadIdx.x; j < len; j += blockDim.x) 
            z[shape::getIndexOffset(j, zShapeInfo, len)] = x[shape::getIndexOffset(j, xShapeInfo, len)];

        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
__host__ static void gatherCudaLauncher(const cudaStream_t *stream, const int numOfSubArrs, 
                                    const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xOffsets,
                                    const void* vy, const Nd4jLong* yShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zOffsets) {

    gatherCuda<X,Y,Z><<<numOfSubArrs, MAX_NUM_THREADS, 1024, *stream>>>(numOfSubArrs, vx, xShapeInfo, xOffsets, vy, yShapeInfo, vz, zShapeInfo, zOffsets);
}

//////////////////////////////////////////////////////////////////////
void gather(graph::LaunchContext* context, const NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {

    const int inputRank = input->rankOf();
    int axis = intArgs.size() > 0 ? intArgs[0] : 0;    
    if(axis < 0)
        axis += inputRank;

    const int numOfIntArgs = intArgs.size();

    if (indices == nullptr && numOfIntArgs == 2) { // scalar case
        output->assign((*input)(intArgs[1], {axis}));
    }
    else if (indices != nullptr && indices->isScalar()) {

        if(input->rankOf() <= 1) { //For scalar indices, rank 0 or 1 input: can't do tensor along dimension 0 as this is whole array... instead, we want to get a scalar
            auto idx = indices->e<Nd4jLong>(0);
            auto scalarNDArray = input->e(idx);
            output->assign(scalarNDArray);
        } 
        else {                
            NDArray inSubArr = (*input)(indices->e<Nd4jLong>(0), {axis});            
            output->assign(inSubArr);
        }
    }    
    else {

        NDArray* pIndices = const_cast<NDArray*>(indices);
        if(indices == nullptr)          
            pIndices = new NDArray(input->ordering(), {numOfIntArgs-1}, std::vector<double>(intArgs.begin() + 1, intArgs.end()), DataType::INT64, input->getContext());
        
        std::vector<int> dimsOut(pIndices->rankOf());
        std::iota(dimsOut.begin(), dimsOut.end(), axis);   // fill with axis, axis+1, ... axis+pIndices->rankOf()-1
        
        const Nd4jLong numOfSubArrs = pIndices->lengthOf();

        Nd4jLong *outSubArrShapeInfo(nullptr), *inSubArrShapeInfo(nullptr), *outSubArrOffsets(nullptr), *inSubArrOffsets(nullptr);        
        input-> getSubArrShapeAndOffsets({axis},  inSubArrShapeInfo,  inSubArrOffsets);
        output->getSubArrShapeAndOffsets(dimsOut, outSubArrShapeInfo, outSubArrOffsets);

        PointersManager manager(context, "gather");
        auto xShapeInfo = reinterpret_cast<Nd4jLong*>(manager.replicatePointer(inSubArrShapeInfo,  shape::shapeInfoByteLength(inSubArrShapeInfo)));
        auto zShapeInfo = reinterpret_cast<Nd4jLong*>(manager.replicatePointer(outSubArrShapeInfo, shape::shapeInfoByteLength(outSubArrShapeInfo)));
        auto xOffsets   = reinterpret_cast<Nd4jLong*>(manager.replicatePointer(inSubArrOffsets,    numOfSubArrs * sizeof(Nd4jLong)));
        auto zOffsets   = reinterpret_cast<Nd4jLong*>(manager.replicatePointer(outSubArrOffsets,   numOfSubArrs * sizeof(Nd4jLong)));
                
        NDArray::prepareSpecialUse({output}, {input, pIndices});
        BUILD_TRIPLE_SELECTOR(input->dataType(), pIndices->dataType(), output->dataType(), gatherCudaLauncher, (context->getCudaStream(), numOfSubArrs, input->getSpecialBuffer(), xShapeInfo, xOffsets, pIndices->getSpecialBuffer(), pIndices->getSpecialShapeInfo(), output->getSpecialBuffer(), zShapeInfo, zOffsets ), NUMERIC_TYPES, INTEGER_TYPES, NUMERIC_TYPES);
        NDArray::registerSpecialUse({output}, {input, pIndices});

        manager.synchronize();

        if(indices == nullptr)
            delete pIndices;
    }        
}    


BUILD_TRIPLE_TEMPLATE(template void gatherCudaLauncher, (const cudaStream_t *stream, const int numOfSubArrs, const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xOffsets, const void* vy, const Nd4jLong* yShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zOffsets), NUMERIC_TYPES, INTEGER_TYPES, NUMERIC_TYPES);



}
}
}