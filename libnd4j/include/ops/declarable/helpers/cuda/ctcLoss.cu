/*******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
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
 *******************************************************************************/
 //
 // @author AbdelRauf
 //

#include <ops/declarable/platform/cudnn/cudnnUtils.h>
#include <array/NDArrayFactory.h>
#include <vector>


namespace sd   {
namespace ops     {
namespace platforms {



    template<typename Op, typename ...Args>
    void callCudnnIfNoErr(cudnnStatus_t &err, Op op, Args&&... args){
        if(err==CUDNN_STATUS_SUCCESS){
            err = op(std::forward<Args>(args)...);
            if(err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(err));
            }
        }
    }

    template <typename T>
    const T* bufferInHost( const NDArray &array)  {
        array.syncToHost();
        return reinterpret_cast<const T*>(array.buffer());
    }

    std::vector<int> getConcatTargets(const NDArray &targetLabels, const NDArray &targetLabelLengths){
                //concatenate target labels
                const int32_t *tlabels = bufferInHost<int32_t>(targetLabels);
                const int32_t *tlens =bufferInHost<int32_t>(targetLabelLengths);
                int32_t nextOffset = targetLabels.strideAt(0);
                int32_t elStride = targetLabels.strideAt(1);
                int32_t batchCount = targetLabelLengths.lengthOf();
                std::vector<int> labels;
                labels.resize(targetLabels.lengthOf());
                int j=0;
                if(targetLabels.ews()){
                    for(int i=0; i<batchCount;i++){
                        int count = tlens[i];
                        for( int k=0;k<count;k++){
                            labels[j] = tlabels[k];
                            j++;
                        }
                        tlabels+=nextOffset;
                    }
                }else{
                    for(int i=0; i<batchCount;i++){
                        int count = tlens[i];
                        for( int k=0;k<count;k++){
                            labels[j] = tlabels[k*elStride];
                            j++;
                        }
                        tlabels+=nextOffset;
                    }
                }
                return labels;
    }

    cudnnStatus_t cudnnCtcLoss(const LaunchContext  &context, const  NDArray &probs, const int32_t* targetLabelsPtr, const NDArray&  probInputLengthes,
                               const NDArray &targetLabelLengths, NDArray &ctcLosses,  NDArray &grads){
        const int dims[] = {(int)probs.sizeAt(0), (int)probs.sizeAt(1), (int)probs.sizeAt(2)};
        const int strides[] = {(int)probs.strideAt(0), (int)probs.strideAt(1), (int)probs.strideAt(2)};
        auto handle = reinterpret_cast<cudnnHandle_t *>(context.getCuDnnHandle());
        cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
        callCudnnIfNoErr(err, cudnnSetStream, *handle, *context.getCudaStream());

        cudnnCTCLossDescriptor_t  ctcLossDesc;
        cudnnTensorDescriptor_t probsDesc = nullptr;
        cudnnTensorDescriptor_t gradsDesc = nullptr;
        callCudnnIfNoErr(err, cudnnCreateCTCLossDescriptor, &ctcLossDesc);
        callCudnnIfNoErr(err, cudnnSetCTCLossDescriptorEx, ctcLossDesc, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN);
        callCudnnIfNoErr(err, cudnnCreateTensorDescriptor, &probsDesc);
        callCudnnIfNoErr(err, cudnnSetTensorNdDescriptor, probsDesc, cudnnDataType(probs.dataType()), probs.rankOf() , dims, strides);
        if(!grads.isEmpty()){
            const int gradStrides[] = {(int)grads.strideAt(0), (int)grads.strideAt(1), (int)grads.strideAt(2)};
            callCudnnIfNoErr(err, cudnnCreateTensorDescriptor, &gradsDesc);
            callCudnnIfNoErr(err, cudnnSetTensorNdDescriptor, gradsDesc, cudnnDataType(grads.dataType()), grads.rankOf() , dims, gradStrides);
        }

        size_t tempWorkSpaceSize=0;
        callCudnnIfNoErr(err,cudnnGetCTCLossWorkspaceSize, *handle,  probsDesc, gradsDesc,
            targetLabelsPtr,
            bufferInHost<int32_t>(targetLabelLengths),
            bufferInHost<int32_t>(probInputLengthes),
            CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
            ctcLossDesc, &tempWorkSpaceSize);

        // Allocate temp tempWorkspace buffer
        void *tempWorkSpace = nullptr;
        cudaMalloc(&tempWorkSpace, tempWorkSpaceSize);

        NDArray::prepareSpecialUse({&ctcLosses, &grads}, {&probs});
        callCudnnIfNoErr(err, cudnnCTCLoss,*handle,
            probsDesc,
            probs.specialBuffer(),
            targetLabelsPtr,
            bufferInHost<int32_t>(targetLabelLengths),
            bufferInHost<int32_t>(probInputLengthes),
            ctcLosses.specialBuffer(),
            gradsDesc,
            grads.specialBuffer(),
            CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
            ctcLossDesc,
            tempWorkSpace,
            tempWorkSpaceSize);

        NDArray::registerSpecialUse({&ctcLosses, &grads}, {&probs});

        cudaFree(tempWorkSpace);
        callCudnnIfNoErr(err, cudnnDestroyTensorDescriptor,probsDesc);
        if(gradsDesc) callCudnnIfNoErr(err, cudnnDestroyTensorDescriptor,gradsDesc);
        callCudnnIfNoErr(err, cudnnDestroyCTCLossDescriptor,ctcLossDesc);
        return err;
     }

    PLATFORM_IMPL(ctc_loss, ENGINE_CUDA) {
        auto targetLabels = INPUT_VARIABLE(0);
        auto logitInput = INPUT_VARIABLE(1);
        auto targetLabelLengths = INPUT_VARIABLE(2);
        auto logitInputLengths = INPUT_VARIABLE(3);
        auto outputLosses = OUTPUT_VARIABLE(0);
        auto context = block.launchContext();
        //in Cudnn Batch is in the middle dimension
        logitInput->permutei({1,0,2});
        //in Cudnn targets are concantenated instead of batched as matrix
        auto labels = getConcatTargets(*targetLabels, *targetLabelLengths);
        const int32_t *ldata= labels.data();
        auto emptyGrads= NDArrayFactory::empty<float>();
        auto err = cudnnCtcLoss(*context, *logitInput, ldata, *logitInputLengths, *targetLabelLengths, *outputLosses, emptyGrads);
        if(err!=CUDNN_STATUS_SUCCESS) throw sd::cuda_exception::build("ctc_loss CUDNN call failure ", err);
        return Status::OK();
    }

    template<typename T>
    bool checkLabelLength(const NDArray &labelLengthArr){
            //check label lengthes
            auto lenBatch = labelLengthArr.lengthOf();
            for(int i=0; i < lenBatch; i++){
                // The labelLengths is greater than 256.
                if(labelLengthArr.e<int32_t>(i)>256) return false;
            }
            return true;
    }

    PLATFORM_CHECK(ctc_loss, ENGINE_CUDA) {
        auto targetLabels = INPUT_VARIABLE(0);
        auto logitInput = INPUT_VARIABLE(1);
        auto targetLabelLengths = INPUT_VARIABLE(2);
        auto logitInputLengths = INPUT_VARIABLE(3);
        auto outputLosses = OUTPUT_VARIABLE(0);
        int blankIndex = INT_ARG(0);

        auto dTypeInput = logitInput->dataType();
        auto intType = targetLabelLengths->dataType();
        auto dTypeOutput = outputLosses->dataType();

        bool is_supported = blankIndex==0 && intType == DataType::INT32  && dTypeInput == DataType::FLOAT32;
        is_supported = is_supported && outputLosses->ews() && targetLabelLengths->ews() && targetLabels->ews() && logitInputLengths->ews();
        is_supported = is_supported && checkLabelLength<int32_t>(*targetLabelLengths);
        return  is_supported;
    }

    PLATFORM_IMPL(ctc_loss_grad, ENGINE_CUDA) {
        auto targetLabels = INPUT_VARIABLE(0);
        auto logitInput = INPUT_VARIABLE(1);
        auto targetLabelLengths = INPUT_VARIABLE(2);
        auto logitInputLengths = INPUT_VARIABLE(3);
        auto outputGradients = OUTPUT_VARIABLE(0);
        auto context = block.launchContext();
        //in Cudnn Batch is in the middle dimension
        logitInput->permutei({1,0,2});
        outputGradients->permutei({1,0,2});
        //in Cudnn targets are concantenated instead of batched as matrix
        auto labels = getConcatTargets(*targetLabels, *targetLabelLengths);
        const int32_t * ldata= labels.data();
        auto tempLosses = NDArrayFactory::create<float>('c', {logitInputLengths->sizeAt(0)});
        auto err = cudnnCtcLoss(*context, *logitInput, ldata, *logitInputLengths, *targetLabelLengths, tempLosses, *outputGradients);
        if(err!=CUDNN_STATUS_SUCCESS) throw sd::cuda_exception::build("ctc_loss CUDNN call failure ", err);
        //restore grads shape from {T, BATCH, C} -> {BATCHS, T, C}
        outputGradients->permutei({1,0,2});
        //tempLosses.printIndexedBuffer("tempLosses");
        return Status::OK();
    }

    PLATFORM_CHECK(ctc_loss_grad, ENGINE_CUDA) {
        auto targetLabels = INPUT_VARIABLE(0);
        auto logitInput = INPUT_VARIABLE(1);
        auto targetLabelLengths = INPUT_VARIABLE(2);
        auto logitInputLengths = INPUT_VARIABLE(3);
        auto outputGrads = OUTPUT_VARIABLE(0);
        int blankIndex = INT_ARG(0);

        auto dTypeInput = logitInput->dataType();
        auto intType = targetLabelLengths->dataType();
        auto dTypeOutput = outputGrads->dataType();

        bool is_supported = blankIndex==0 && intType == DataType::INT32  && dTypeInput == DataType::FLOAT32;
        is_supported = is_supported && outputGrads->ews() && targetLabelLengths->ews() && targetLabels->ews() && logitInputLengths->ews();
        is_supported = is_supported && checkLabelLength<int32_t>(*targetLabelLengths);
        return  is_supported;
    }

}
}
}