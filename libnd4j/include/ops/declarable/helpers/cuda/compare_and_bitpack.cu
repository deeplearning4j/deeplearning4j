/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

 //
 // @author AbdelRauf 
 //

#include <system/op_boilerplate.h>
#include <ops/declarable/helpers/imagesHelpers.h>
#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/helpers/adjust_hue.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/transforms.h>
#include <helpers/LoopsCoordsHelper.h>
namespace sd    {
namespace ops     {
namespace helpers {

    template<typename X>
    _CUDA_HD uint8_t pack(const X* buff, const X& threshold){
        uint8_t res;
        res = (buff[0] > threshold) << 7;
        res = res | ((buff[1] > threshold) << 6); 
        res = res | ((buff[2] > threshold) << 5);
        res = res | ((buff[3] > threshold) << 4);
        res = res | ((buff[4] > threshold) << 3);
        res = res | ((buff[5] > threshold) << 2);
        res = res | ((buff[6] > threshold) << 1);
        res = res | (buff[7] > threshold);
        return res;
    }

    template<>
    _CUDA_HD uint8_t pack<bool>(const bool* buff, const bool &threshold){
        //ignore threshold
        uint8_t res;
        res = buff[0] << 7;
        res = res | (buff[1] << 6); 
        res = res | (buff[2] << 5);
        res = res | (buff[3] << 4);
        res = res | (buff[4] << 3);
        res = res | (buff[5] << 2);
        res = res | (buff[6] << 1);
        res = res | buff[7] ;
        return res;
    }

    template<typename X>
    _CUDA_HD uint8_t pack(const X* buff, int stride, const X& threshold){
        uint8_t res;
        res = (buff[0] > threshold) << 7;
        res = res | ((buff[1*stride] > threshold) << 6); 
        res = res | ((buff[2*stride] > threshold) << 5);
        res = res | ((buff[3*stride] > threshold) << 4);
        res = res | ((buff[4*stride] > threshold) << 3);
        res = res | ((buff[5*stride] > threshold) << 2);
        res = res | ((buff[6*stride] > threshold) << 1);
        res = res | (buff[7*stride] > threshold);
        return res;
    }

    template<>
    _CUDA_HD uint8_t pack<bool>(const bool* buff, int stride, const bool &threshold){
        //ignore threshold
        uint8_t res;
        res = buff[0] << 7;
        res = res | (buff[1*stride] << 6); 
        res = res | (buff[2*stride] << 5);
        res = res | (buff[3*stride] << 4);
        res = res | (buff[4*stride] << 3);
        res = res | (buff[5*stride] << 2);
        res = res | (buff[6*stride] << 1);
        res = res | buff[7*stride] ;
        return res;
    }
///////////////////////////////////////////////////////////////////
template <typename T>
static void _CUDA_G cmpBitpack(const void* vx, void* vz,  int rank, int len, const Nd4jLong *xStridesExtended, const Nd4jLong *outPutShapeInfo, T threshold) {

    const T* x = reinterpret_cast<const T*>(vx);
    uint8_t* z = reinterpret_cast<uint8_t*>(vz);

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto shapes = shape::shapeOf(outPutShapeInfo);
    auto zStrides = shape::stride(outPutShapeInfo);
    Nd4jLong coords[MAX_RANK] = {};
    Nd4jLong* ptr_coords = (Nd4jLong*)&coords;
    // its extended as {rank+1} so xStridesExtended[rank] is valid 
    auto inLastStride = xStridesExtended[rank];

    for(auto k=tid; k < len; k+=gridDim.x * blockDim.x){
        sd::index2coords_C(k, rank, shapes, ptr_coords); 
        auto offset = sd::offset_from_coords(xStridesExtended, zStrides, ptr_coords, rank); 
        auto buffPart = &(x[offset.first]);
        auto outBuffPart = &(z[offset.second]);
        *outBuffPart = pack<T>(buffPart, inLastStride, threshold);
    }
}

template <typename T>
static void _CUDA_G cmpBitpackEws(const void* vx, void* vz,  int len, const Nd4jLong xStride,  const Nd4jLong yStride,  T threshold) {

    const T* x = reinterpret_cast<const T*>(vx);
    uint8_t* z = reinterpret_cast<uint8_t*>(vz);

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(xStride==1){
        for(auto k=tid; k < len; k+=gridDim.x * blockDim.x){
            auto buffPart = &(x[k*8]);
            auto outBuffPart = &(z[k*yStride]);
            *outBuffPart = pack<T>(buffPart, threshold); 
        }
    }else{
        for(auto k=tid; k < len; k+=gridDim.x * blockDim.x){
            auto buffPart = &(x[k*8*xStride]);
            auto outBuffPart = &(z[k*yStride]);
            *outBuffPart =  pack<T>(buffPart, xStride, threshold); 
        }
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static _CUDA_H void cmpBitpackCudaLauncher(sd::graph::Context& block, const NDArray& input, const NDArray& thresholdScalar, NDArray& output) {
    T threshold = thresholdScalar.e<T>(0);


    auto inStrides = input.stridesOf();
    auto rank = output.rankOf();

    //threadblock size
    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    //grid size
    const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    auto stream = block.launchContext()->getCudaStream();
    //nd4j_printf("n %i g %i th %i \n", output.lengthOf(), blocksPerGrid, threadsPerBlock);
    PointersManager manager(block.launchContext(), "compare_and_bitpack");
    NDArray::prepareSpecialUse({&output}, {&input});
    if(input.ews()>0 && output.ews()>0 && input.ordering()=='c' && output.ordering()=='c'){
        cmpBitpackEws<T><<<blocksPerGrid, threadsPerBlock >>>(input.specialBuffer(), output.specialBuffer(), output.lengthOf(), inStrides[rank-1], output.stridesOf()[rank-1] , threshold);
    }else{
        //if output shape is {n1, n2, n3} then input shape is { n1. n2, n3 * 8}
        //therefore we can split input shape  {n1, n2, n3 , 8} and correct its stride
        //as we do not need last shape info. lets just extend and correct its stride
        Nd4jLong extendedStrides[MAX_RANK];
        for(int i=0;i<rank; i++){
            extendedStrides[i] = inStrides[i];
        }
        //lets correct new stride
        extendedStrides[rank-1] = 8*inStrides[rank-1];
        extendedStrides[rank] = inStrides[rank-1];

        auto strideSize = (rank+1)*sizeof(Nd4jLong); 
        Nd4jLong* extendedStridesDevPtr = reinterpret_cast<Nd4jLong*>(manager.replicatePointer(extendedStrides, strideSize));
        cmpBitpack<T><<<blocksPerGrid, threadsPerBlock >>>(input.specialBuffer(), output.specialBuffer(), rank, output.lengthOf(), extendedStridesDevPtr, output.specialShapeInfo(), threshold);
    }

    NDArray::registerSpecialUse({&output}, {&input});
    manager.synchronize(); 

}


void compareAndBitpack(sd::graph::Context& block, const NDArray& input, const NDArray& threshold, NDArray& output)  {

    BUILD_SINGLE_SELECTOR(input.dataType(), cmpBitpackCudaLauncher, (block, input, threshold, output), LIBND4J_TYPES);
}



}
}
}

