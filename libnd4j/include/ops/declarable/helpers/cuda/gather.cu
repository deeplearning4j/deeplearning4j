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
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

namespace sd    {
namespace ops     {
namespace helpers {

    template<typename X, typename Y>
    __global__ static void gatherCudaLinearKernel(const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo,
    void* vz, const Nd4jLong* zShapeInfo) {


    __shared__ const X* x;
    __shared__ const Y* y;
    __shared__ X* z;
    __shared__ Nd4jLong xLen, yLen, zLen;

    if (threadIdx.x == 0) {
        x = reinterpret_cast<const X*>(vx);
        z = reinterpret_cast<X*>(vz);
        y = reinterpret_cast<const Y *>(vy);
        xLen = shape::length(xShapeInfo);
        yLen = shape::length(yShapeInfo);
        zLen = shape::length(zShapeInfo);
    }
    __syncthreads();
    //const Nd4jLong zLen = shape::length(zShapeInfo);
    auto start = blockIdx.x * blockDim.x + threadIdx.x;
    auto step = blockDim.x * gridDim.x;

    for (int j = start; j < zLen; j += step) {
        auto zIndex = shape::getIndexOffset(j, zShapeInfo);
        auto yIndex = shape::getIndexOffset(j, yShapeInfo);
        auto xIndex = shape::getIndexOffset(y[yIndex], xShapeInfo);
        z[zIndex] = x[xIndex];
    }
}

//////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__global__ static void gatherCuda(const int numOfSubArrs,
                                    const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xOffsets,
                                    const void* vy, const Nd4jLong* yShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zOffsets) {

    const Y* y = reinterpret_cast<const Y*>(vy);
    __shared__ const X* x;
    __shared__ X* z;

    const Nd4jLong len = shape::length(xShapeInfo);
    //const Nd4jLong zLen = shape::length(zShapeInfo);
    for (int i = blockIdx.x; i < numOfSubArrs; i += gridDim.x) {

        if (threadIdx.x == 0) {
            x = reinterpret_cast<const X*>(vx) + xOffsets[y[shape::getIndexOffset(i, yShapeInfo)]];
            z = reinterpret_cast<X*>(vz) + zOffsets[i];
        }
        __syncthreads();

        for (int j = threadIdx.x; j < len; j += blockDim.x) {
            auto zIndex = shape::getIndexOffset(j, zShapeInfo);
            auto xIndex = shape::getIndexOffset(j, xShapeInfo);
            z[zIndex] = x[xIndex];
        }
        __syncthreads();
    }
}

template<typename X, typename Y>
__host__ static void gatherCudaLinear(const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, const void* vy, const Nd4jLong* yShapeInfo,
                                            void* vz, const Nd4jLong* zShapeInfo) {
    gatherCudaLinearKernel<X,Y><<<128, 256, 1024, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

//////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__host__ static void gatherCudaLauncher(const cudaStream_t *stream, const int numOfSubArrs,
                                    const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xOffsets,
                                    const void* vy, const Nd4jLong* yShapeInfo,
                                          void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zOffsets) {
    gatherCuda<X,Y><<<numOfSubArrs, MAX_NUM_THREADS, 1024, *stream>>>(numOfSubArrs, vx, xShapeInfo, xOffsets, vy, yShapeInfo, vz, zShapeInfo, zOffsets);
}

//////////////////////////////////////////////////////////////////////
void gather(sd::LaunchContext * context, const NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {

    const int inputRank = input->rankOf();
    const int numOfIntArgs = intArgs.size();

    int axis = numOfIntArgs > 0 ? intArgs[0] : 0;
    if(axis < 0)
        axis += inputRank;

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
        if (output->rankOf() > 1) {
            PointersManager manager(context, "gather");
            auto xShapeInfo = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(inSubArrShapeInfo,
                                                                                    shape::shapeInfoByteLength(
                                                                                            inSubArrShapeInfo)));
            auto zShapeInfo = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(outSubArrShapeInfo,
                                                                                    shape::shapeInfoByteLength(
                                                                                            outSubArrShapeInfo)));
            auto xOffsets = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(inSubArrOffsets, (input->lengthOf() /
                                                                                                    shape::length(
                                                                                                            inSubArrShapeInfo)) *
                                                                                                   sizeof(Nd4jLong)));
            auto zOffsets = reinterpret_cast<Nd4jLong *>(manager.replicatePointer(outSubArrOffsets,
                                                                                  (output->lengthOf() /
                                                                                   shape::length(outSubArrShapeInfo)) *
                                                                                  sizeof(Nd4jLong)));

            NDArray::prepareSpecialUse({output}, {input, pIndices});
            BUILD_DOUBLE_SELECTOR(input->dataType(), pIndices->dataType(), gatherCudaLauncher, (context->getCudaStream(), numOfSubArrs, input->specialBuffer(), xShapeInfo, xOffsets, pIndices->specialBuffer(), pIndices->specialShapeInfo(), output->specialBuffer(), zShapeInfo, zOffsets), LIBND4J_TYPES, INDEXING_TYPES);
            NDArray::registerSpecialUse({output}, {input, pIndices});
            manager.synchronize();
        }
        else {
            NDArray::prepareSpecialUse({output}, {input, pIndices});
            BUILD_DOUBLE_SELECTOR(input->dataType(), pIndices->dataType(), gatherCudaLinear, (context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), pIndices->specialBuffer(), pIndices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo()), LIBND4J_TYPES, INDEXING_TYPES);
            NDArray::registerSpecialUse({output}, {input, pIndices});

        }

        if(indices == nullptr)
            delete pIndices;

    }
}

}
}
}