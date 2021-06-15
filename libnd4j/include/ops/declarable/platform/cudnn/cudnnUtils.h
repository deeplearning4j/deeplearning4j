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
// @author raver119@gmail.com
//

#ifndef SD_CUDNNUTILS_H
#define SD_CUDNNUTILS_H

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <system/dll.h>
#include <vector>
#include <cudnn.h>

#define CUDNN_NEW_RNN_API_VER 8001
#define CUDNN_CLIPPING_API_VER 7201

namespace sd      {
namespace ops       {
namespace platforms {

    DECLARE_PLATFORM(conv2d, ENGINE_CUDA);
    DECLARE_PLATFORM(conv2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(conv3dnew, ENGINE_CUDA);
    DECLARE_PLATFORM(conv3dnew_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CUDA);
    DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(batchnorm, ENGINE_CUDA);
    DECLARE_PLATFORM(batchnorm_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(avgpool2d, ENGINE_CUDA);
    DECLARE_PLATFORM(avgpool2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(maxpool2d, ENGINE_CUDA);
    DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(avgpool3dnew, ENGINE_CUDA);
    DECLARE_PLATFORM(avgpool3dnew_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(maxpool3dnew, ENGINE_CUDA);
    DECLARE_PLATFORM(maxpool3dnew_bp, ENGINE_CUDA);

    DECLARE_PLATFORM(lstmLayer, ENGINE_CUDA);

    DECLARE_PLATFORM(ctc_loss, ENGINE_CUDA);
    DECLARE_PLATFORM(ctc_loss_grad, ENGINE_CUDA);


//////////////////////////////////////////////////////////////////////////
template<typename Op, typename ...Args>
FORCEINLINE void callCudnnIfNoErr(cudnnStatus_t &err, Op op, Args&&... args){
        if(err==CUDNN_STATUS_SUCCESS){
            err = op(std::forward<Args>(args)...);
            if(err){
                nd4j_printf("Cudnn error code %s\n", cudnnGetErrorString(err));
            }
        }
}

template <typename T>
FORCEINLINE const T* bufferInHost( const NDArray &array)  {
    array.syncToHost();
    return reinterpret_cast<const T*>(array.buffer());
}

 

struct CudnnTensor{

    CudnnTensor(const CudnnTensor& s) = delete; 

    CudnnTensor& operator=(const CudnnTensor& other) = delete;

    CudnnTensor(){ 
       create();
    }

    CudnnTensor(bool createDesc){ 
       if(createDesc)  create();
       else { desc= nullptr; }
    }

    void create(){
      desc_status = cudnnCreateTensorDescriptor(&desc); 
    }

    template<typename ...Args>
    void set(cudnnStatus_t &last_err, Args&&... args){ 
        if(desc_status==CUDNN_STATUS_SUCCESS && last_err==CUDNN_STATUS_SUCCESS){
            last_err = cudnnSetTensorNdDescriptor(desc, std::forward<Args>(args)...);
            if(last_err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(last_err));
            }
        }
    }

    cudnnStatus_t descStatus(){ return desc_status;}

    operator cudnnTensorDescriptor_t() const { return desc; }

    ~CudnnTensor(){
      if(desc_status==CUDNN_STATUS_SUCCESS)  cudnnDestroyTensorDescriptor(desc);
    }

    cudnnStatus_t desc_status;
    cudnnTensorDescriptor_t desc;
};

struct CudnnTensorList{

    CudnnTensorList(const CudnnTensorList& s) = delete; 

    CudnnTensorList& operator=(const CudnnTensorList& other) = delete;

    CudnnTensorList(int size){ 
       descList.resize(size);
       desc_status = CUDNN_STATUS_SUCCESS;
       for(int i=0;i<size;i++){ 
           desc_status = cudnnCreateTensorDescriptor(&descList[i]);
           if(desc_status != CUDNN_STATUS_SUCCESS) break;
       } 
    }

    template<typename ...Args>
    void set(cudnnStatus_t &last_err, int index, Args&&... args){ 
        if(desc_status==CUDNN_STATUS_SUCCESS && last_err==CUDNN_STATUS_SUCCESS && index<descList.size()){
            last_err = cudnnSetTensorNdDescriptor(descList[index], std::forward<Args>(args)...);
            if(last_err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(last_err));
            }
        }
    }


    cudnnStatus_t descStatus(){ return desc_status;}

    cudnnTensorDescriptor_t get(int i) const { 
        if(i<descList.size()) return descList[i];
        return nullptr;
     }

    const cudnnTensorDescriptor_t* getDescriptors() const { return descList.data(); }

    ~CudnnTensorList(){
      for(auto x: descList){
          cudnnDestroyTensorDescriptor(x);
      }
    }

    cudnnStatus_t desc_status;
    std::vector<cudnnTensorDescriptor_t> descList;
};

struct FilterDesc{

    FilterDesc(const FilterDesc& s) = delete; 

    FilterDesc& operator=(const FilterDesc& other) = delete;

    FilterDesc(){ 
       desc_status = cudnnCreateFilterDescriptor(&desc);
    }

    template<typename ...Args>
    void set(cudnnStatus_t &last_err, Args&&... args){ 
        if(desc_status==CUDNN_STATUS_SUCCESS && last_err==CUDNN_STATUS_SUCCESS){
            last_err = cudnnSetFilterNdDescriptor(desc, std::forward<Args>(args)...);
            if(last_err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(last_err));
            }
        }
    }

    cudnnStatus_t descStatus(){ return desc_status;}

    operator cudnnFilterDescriptor_t() const { return desc; }

    ~FilterDesc(){
      if(desc_status==CUDNN_STATUS_SUCCESS)  cudnnDestroyFilterDescriptor(desc); ;
    }

    cudnnStatus_t desc_status;
    cudnnFilterDescriptor_t desc;
};

struct DropoutDesc{

    DropoutDesc(const DropoutDesc& s) = delete; 

    DropoutDesc& operator=(const DropoutDesc& other) = delete;

    DropoutDesc(){ 
       create();
    }

    DropoutDesc(bool createDesc){ 
       if(createDesc)  create();
       else { desc= nullptr; }
    }

    void create(){
      desc_status = cudnnCreateDropoutDescriptor(&desc); 
    }

    template<typename ...Args>
    void set(cudnnStatus_t &last_err, Args&&... args){ 
        if(desc_status==CUDNN_STATUS_SUCCESS && last_err==CUDNN_STATUS_SUCCESS){
            last_err = cudnnSetDropoutDescriptor(desc, std::forward<Args>(args)...);
            if(last_err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(last_err));
            }
        }
    }

    cudnnStatus_t descStatus(){ return desc_status;}

    operator cudnnDropoutDescriptor_t() const { return desc; }

    ~DropoutDesc(){
      if(desc_status==CUDNN_STATUS_SUCCESS)  cudnnDestroyDropoutDescriptor(desc); ;
    }

    cudnnStatus_t desc_status;
    cudnnDropoutDescriptor_t desc;
};

#if CUDNN_VERSION > CUDNN_NEW_RNN_API_VER
struct RnnDataDesc{

    RnnDataDesc(const RnnDataDesc& s) = delete; 

    RnnDataDesc& operator=(const RnnDataDesc& other) = delete;

    RnnDataDesc(){ 
       desc_status = cudnnCreateRNNDataDescriptor(&desc);
    }

    template<typename ...Args>
    void set(cudnnStatus_t &last_err, Args&&... args){ 
        if(desc_status==CUDNN_STATUS_SUCCESS && last_err==CUDNN_STATUS_SUCCESS){
            last_err = cudnnSetRNNDataDescriptor(desc, std::forward<Args>(args)...);
            if(last_err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(last_err));
            }
        }
    }

    cudnnStatus_t descStatus(){ return desc_status;}

    operator cudnnRNNDataDescriptor_t() const { return desc; }

    ~RnnDataDesc(){
      if(desc_status==CUDNN_STATUS_SUCCESS)  cudnnDestroyRNNDataDescriptor(desc); ;
    }

    cudnnStatus_t desc_status;
    cudnnRNNDataDescriptor_t desc;
};
#endif

FORCEINLINE cudnnStatus_t setRnnDescriptorOldApi(cudnnRNNDescriptor_t rnnDesc, 
                            cudnnHandle_t handle,
                            cudnnRNNInputMode_t inputMode,
                            cudnnDirectionMode_t dirMode,
                            cudnnRNNMode_t cellMode,
                            cudnnRNNAlgo_t algo,
                            cudnnDataType_t mathPrec,
                            int32_t hiddenSize,
                            int32_t numLayers,
                            cudnnDropoutDescriptor_t dropoutDesc,  bool use_tensor_op=false){

    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    callCudnnIfNoErr( err, cudnnSetRNNDescriptor_v6, handle, rnnDesc, hiddenSize, numLayers,
        dropoutDesc, inputMode, dirMode, cellMode, algo, mathPrec);
#if CUDNN_VERSION >= 7001
    if(cudnnGetVersion()>=7001){
        cudnnMathType_t mathType =  use_tensor_op ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
        callCudnnIfNoErr( err, cudnnSetRNNMatrixMathType, rnnDesc,  mathType); 
    }
#endif
    return err;
}


struct RnnDesc{

    RnnDesc(const RnnDesc& s) = delete; 

    RnnDesc& operator=(const RnnDesc& other) = delete;

    RnnDesc(){ 
       desc_status = cudnnCreateRNNDescriptor(&desc);
    }

    template<typename ...Args>
    void setUsingOldAPI(cudnnStatus_t &last_err, Args&&... args){ 
        if(desc_status==CUDNN_STATUS_SUCCESS && last_err==CUDNN_STATUS_SUCCESS){
            last_err = setRnnDescriptorOldApi(desc, std::forward<Args>(args)...);
            if(last_err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(last_err));
            }
        }
    }

#if CUDNN_VERSION>=CUDNN_NEW_RNN_API_VER
    template<typename ...Args>
    void set(cudnnStatus_t &last_err, Args&&... args){ 
        if(desc_status==CUDNN_STATUS_SUCCESS && last_err==CUDNN_STATUS_SUCCESS){
            last_err = cudnnSetRNNDescriptor_v8(desc, std::forward<Args>(args)...);
            if(last_err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(last_err));
            }
        }
    }
#endif

    cudnnStatus_t descStatus(){ return desc_status;}

    operator cudnnRNNDescriptor_t() const { return desc; }

    ~RnnDesc(){
      if(desc_status==CUDNN_STATUS_SUCCESS)  cudnnDestroyRNNDescriptor(desc); ;
    }

    cudnnStatus_t desc_status;
    cudnnRNNDescriptor_t desc;
};



struct CTCLossDesc{

    CTCLossDesc(const CTCLossDesc& s) = delete; 

    CTCLossDesc& operator=(const CTCLossDesc& other) = delete;

    CTCLossDesc(){ 
       desc_status = cudnnCreateCTCLossDescriptor(&desc);
    }

    template<typename ...Args>
    void set(cudnnStatus_t &last_err, Args&&... args){ 
        if(desc_status==CUDNN_STATUS_SUCCESS && last_err==CUDNN_STATUS_SUCCESS){
            last_err = cudnnSetCTCLossDescriptorEx(desc, std::forward<Args>(args)...);
            if(last_err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(last_err));
            }
        }
    }

    cudnnStatus_t descStatus(){ return desc_status;}

    operator cudnnCTCLossDescriptor_t() const { return desc; }

    ~CTCLossDesc(){
      if(desc_status==CUDNN_STATUS_SUCCESS)  cudnnDestroyCTCLossDescriptor(desc); ;
    }

    cudnnStatus_t desc_status;
    cudnnCTCLossDescriptor_t desc;
};
//////////////////////////////////////////////////////////////////////////
FORCEINLINE cudnnDataType_t cudnnDataType(sd::DataType dataType) {
    switch (dataType) {
        case sd::DataType::FLOAT32:
            return CUDNN_DATA_FLOAT;
        case sd::DataType::DOUBLE:
            return CUDNN_DATA_DOUBLE;
        case sd::DataType::HALF:
            return CUDNN_DATA_HALF;
        case sd::DataType::INT32:
            return CUDNN_DATA_INT32;
        case sd::DataType::INT8:
            return CUDNN_DATA_INT8;
        default:
            throw datatype_exception::build("Unsupported data type", dataType);
    }
}

//////////////////////////////////////////////////////////////////////////
void checkConv2dCUDNNPadAsymmetric(NDArray* &input, NDArray* &gradI,
                                    const int iH, const int iW,
                                    const int oH, const int oW,
                                    const int kH, const int kW,
                                    const int sH, const int sW,
                                    const int pH, const int pW,
                                    const int dH, const int dW,
                                    const bool isNCHW);

//////////////////////////////////////////////////////////////////////////
void checkConv3dCUDNNPadAsymmetric(NDArray* &input, NDArray* &gradI,
                                    const int iD, const int iH, const int iW,
                                    const int oD, const int oH, const int oW,
                                    const int kD, const int kH, const int kW,
                                    const int sD, const int sH, const int sW,
                                    const int pD, const int pH, const int pW,
                                    const int dD, const int dH, const int dW,
                                    const bool isNCDHW);

//////////////////////////////////////////////////////////////////////////
void pooling2dCUDNN(const LaunchContext* context,
                    const NDArray* input, NDArray* output,
                    const int kH, const int kW,
                    const int sH, const int sW,
                    const int pH, const int pW,
                    const int dH, const int dW,
                    const bool isNCHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling2dBpCUDNN(const LaunchContext* context,
                      const NDArray* input, const NDArray* gradO,
                            NDArray* gradI,
                      const int kH, const int kW,
                      const int sH, const int sW,
                      const int pH, const int pW,
                      const int dH, const int dW,
                      const bool isNCHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling3dCUDNN(const LaunchContext* context,
                    const NDArray* input, NDArray* output,
                    const int kD, const int kH, const int kW,
                    const int sD, const int sH, const int sW,
                    const int pD, const int pH, const int pW,
                    const int dD, const int dH, const int dW,
                    const bool isNCDHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling3dBpCUDNN(const LaunchContext* context,
                    const NDArray* input, const NDArray* gradO,
                          NDArray* gradI,
                    const int kD, const int kH, const int kW,
                    const int sD, const int sH, const int sW,
                    const int pD, const int pH, const int pW,
                    const int dD, const int dH, const int dW,
                    const bool isNCDHW, const cudnnPoolingMode_t mode);

}
}
}

#endif //SD_CUDNNUTILS_H
