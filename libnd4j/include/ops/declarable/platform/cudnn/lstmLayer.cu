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
 // @author AbdelRauf
 //

 #include <ops/declarable/OpRegistrator.h>
 #include "cudnnUtils.h"
 #include <array/NDArrayFactory.h>

 namespace sd     {
 namespace ops      {
 namespace platforms {

//our implementation designed for 1 physical layer
constexpr int numLayers = 1;

//we will copy without using cudnnGetRNNLinLayerMatrixParams : 1 pseudo layer , isBidirectional : 2 pseudo layer
void copyWeights(const cudaStream_t &stream , bool isBidirectional, uint8_t *weightsSpace, size_t weightsSize, uint8_t *inputWeightsData, uint8_t *recurrentWeightsData, uint8_t *biasesData, int inputSize, int hiddenSize, int dataTypeSize){
    int pseudo_layer_count = isBidirectional ? 2 :1 ;
    uint8_t * wptr = weightsSpace;
    auto wEnd = wptr + weightsSize;

    //copy size for 1 full pseudo layer
    //in bidirectional 1 layer consist of 2 pseduo layers
    auto input_pseudo_size = 4 * inputSize * hiddenSize * dataTypeSize;
    auto hidden_pseudo_size = 4 * hiddenSize * hiddenSize * dataTypeSize;
    for(int i=0; i< pseudo_layer_count ; i++){
        if(wptr + input_pseudo_size + hidden_pseudo_size > wEnd) return;
        //copy input weights
        if(inputWeightsData){
            cudaMemcpyAsync(wptr, inputWeightsData, input_pseudo_size, cudaMemcpyDeviceToDevice, stream);
            inputWeightsData +=  input_pseudo_size;
        }
        wptr += input_pseudo_size;
        //copy recurrent weights
        if(recurrentWeightsData){
            cudaMemcpyAsync(wptr, recurrentWeightsData, hidden_pseudo_size, cudaMemcpyDeviceToDevice, stream);
            recurrentWeightsData += hidden_pseudo_size;
        }
        wptr += hidden_pseudo_size;
    }

    //copy bias first 4
    auto bias_size =  4 * hiddenSize * dataTypeSize;
    for(int i=0; i< pseudo_layer_count ; i++){
        //refill first 4 biases
        if(biasesData && wptr + bias_size < wEnd){
            cudaMemcpyAsync(wptr, biasesData, bias_size, cudaMemcpyDeviceToDevice, stream);
            biasesData += bias_size;
        }
        wptr += bias_size; 
        //refill next 4 with zeros
        if(wptr + bias_size < wEnd){
            cudaMemsetAsync(wptr, 0, bias_size, stream);
            wptr += bias_size; 
        } 
   }
       //metset the rest
    if( wEnd-wptr ) cudaMemsetAsync(wptr, 0 , wEnd-wptr, stream);
}

cudnnStatus_t cudnn_rnn_old(LaunchContext *contextPtr, int dataFormat, NDArray *input, NDArray *inputWeights,  NDArray *recurrentWeights,
    NDArray *biases, NDArray *prevAct, NDArray *prevMemCell, NDArray *outputActivations, NDArray *finalTimeStepActivations, NDArray *finalMemCellState,
    int maxSeqLength, int batchSize, int inputSize, int hiddenSize, double cellClip, bool isBidirectional){

    nd4j_debug("cudnn rnn api %s \n", "v6");

    bool training = false;
    cudnnHandle_t handle = *(reinterpret_cast<cudnnHandle_t *>(contextPtr->getCuDnnHandle()));
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    auto stream = *(contextPtr->getCudaStream());
    callCudnnIfNoErr(err, cudnnSetStream, handle, stream); 

    CudnnTensorList xDescList (maxSeqLength);
    CudnnTensorList yDescList (maxSeqLength);

    auto cudnnType = cudnnDataType(input->dataType());
    auto dataTypeSize = input->sizeOfT(); 

    CudnnTensor hxDesc, cxDesc, hyDesc, cyDesc;  

    constexpr int rankOf = 3;
    const int numDirections     = isBidirectional ? 2 : 1;

    const int dimsX[rankOf]    = {batchSize,  inputSize, 1};
    const int stridesX[rankOf] = {inputSize, 1, 1};

    const int dimsY[rankOf]    = {batchSize, hiddenSize * numDirections , 1};
    const int stridesY[rankOf] = {hiddenSize * numDirections, 1, 1};

    const int dimC[rankOf]    = {numLayers * numDirections,  batchSize,  hiddenSize};
    const int strideC[rankOf]  = {batchSize * hiddenSize, hiddenSize, 1}; 

    for(int i=0; i<maxSeqLength; i++){
        xDescList.set(err, i, cudnnType, rankOf, dimsX, stridesX);
        // dxDescList.set(err, i, cudnnType, rankOf, dimsX, stridesX);
        yDescList.set(err, i, cudnnType, rankOf, dimsY, stridesY);
        // dyDescList.set(err, i, cudnnType, rankOf, dimsY, stridesY);
        if(err!=CUDNN_STATUS_SUCCESS)  break;
    }

    auto xDesc0 = xDescList.get(0);

    hxDesc.set(err, cudnnType, rankOf, dimC, strideC);
    cxDesc.set(err, cudnnType, rankOf, dimC, strideC);
    hyDesc.set(err, cudnnType, rankOf, dimC, strideC);
    cyDesc.set(err, cudnnType, rankOf, dimC, strideC);

    //dropout section
    DropoutDesc dropoutDesc; 
    //dropout
    float dropout = 0;
    size_t sizeInBytes=0;
    void *droupoutMem = nullptr;
    uint64_t seed =1; //seed
    if(dropout!=0){
       // dropoutDesc.create();
        callCudnnIfNoErr(err, cudnnDropoutGetStatesSize, handle, &sizeInBytes);
        //allocate and set
        cudaMalloc(&droupoutMem, sizeInBytes);
        dropoutDesc.set(err, handle, dropout, droupoutMem, sizeInBytes, seed );
    }

    //RNN
    RnnDesc rnnDesc;
    cudnnRNNMode_t rnnCellMode = CUDNN_LSTM;
    cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;

    auto direction = isBidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    auto mathPrec = cudnnType;

    //Note: We will set some parameters manually
    constexpr auto inputMode = CUDNN_LINEAR_INPUT;
    rnnDesc.setUsingOldAPI(err, handle, inputMode, direction, rnnCellMode, algo, mathPrec, hiddenSize, numLayers, dropoutDesc);
#if CUDNN_VERSION >= CUDNN_CLIPPING_API_VER
    if(cellClip>0 && cudnnGetVersion()>=CUDNN_CLIPPING_API_VER){
        callCudnnIfNoErr(err, cudnnRNNSetClip, handle, rnnDesc, CUDNN_RNN_CLIP_MINMAX, CUDNN_PROPAGATE_NAN, -cellClip, cellClip);
    }
#endif
    //set up parameters
    size_t weightsSize=0;
    callCudnnIfNoErr(err, cudnnGetRNNParamsSize, handle, rnnDesc, xDesc0, &weightsSize, cudnnType);

    FilterDesc wDesc; 
    int dimW[] = {(int) weightsSize / dataTypeSize, 1, 1};

    wDesc.set(err, cudnnType, CUDNN_TENSOR_NCHW, 3, dimW); 
    //allocation
    uint8_t *weightsSpace =nullptr;
    cudaMalloc(&weightsSpace,  weightsSize); 

    // Set up work space and reserved memory
    void *workSpace = nullptr;
    void *reserveSpace = nullptr;

    size_t workSpaceSizeInBytes = 0 ;
    size_t reserveSpaceSizeInBytes = 0;

    callCudnnIfNoErr(err, cudnnGetRNNWorkspaceSize, handle, rnnDesc, maxSeqLength, xDescList.getDescriptors(), &workSpaceSizeInBytes);
    cudaMalloc(&workSpace, workSpaceSizeInBytes);

    // training
    if(training) {
        callCudnnIfNoErr(err, cudnnGetRNNTrainingReserveSize, handle, rnnDesc, maxSeqLength, xDescList.getDescriptors(), &reserveSpaceSizeInBytes);
        cudaMalloc(&reserveSpace, reserveSpaceSizeInBytes);
    }

    NDArray::prepareSpecialUse({outputActivations, finalTimeStepActivations, finalMemCellState}, {input, inputWeights, recurrentWeights, biases, prevAct, prevMemCell});

    uint8_t *biasesData = biases ? (uint8_t*)biases->specialBuffer() : nullptr;
    auto prevActData = prevAct ? prevAct->specialBuffer() : nullptr;
    auto prevMemCellData = prevMemCell ? prevMemCell->specialBuffer() : nullptr;
    auto finalTimeStepActivationsData = finalTimeStepActivations ? finalTimeStepActivations->specialBuffer() : nullptr;
    auto finalMemCellStateData = finalMemCellState ? finalMemCellState->specialBuffer() : nullptr;

    // dimension 4*nOut implies order it, ft, c't, ot
    // input gate, forget gate, new gate, output gate, input gate, forget gate, new gate, output gate
    // Note: our weights should be transposed and duplicated with C order to match cudnn ones

    NDArray inputWeightsT, recurrentWeightsT;
    uint8_t *inputWeightsData = nullptr;
    uint8_t *recurrentWeightsData = nullptr;
    if(inputWeights){
      inputWeightsT= inputWeights->rankOf()==3?inputWeights->permute({0, 2, 1}).dup('c'):inputWeights->transpose().dup('c');
      inputWeightsData = (uint8_t*)inputWeightsT.specialBuffer();
    }
    if(recurrentWeights){
      recurrentWeightsT = recurrentWeights->rankOf()==3?recurrentWeights->permute({0, 2, 1}).dup('c'):recurrentWeights->transpose().dup('c');
      recurrentWeightsData =  (uint8_t*)recurrentWeightsT.specialBuffer();
    }

    //copy without cudnnGetRNNLinLayerMatrixParams
    copyWeights(stream,isBidirectional, weightsSpace, weightsSize, inputWeightsData, recurrentWeightsData, biasesData, inputSize, hiddenSize, dataTypeSize);

    //permute based on dataformat
    NDArray *argX = input;
    NDArray *argOutput= outputActivations;
    NDArray permutedX, outputH;

    if(outputActivations!=nullptr && (dataFormat != 0 ||  outputActivations->ordering()!='c')){
        outputH = NDArray('c',  std::vector<Nd4jLong>{maxSeqLength, batchSize, (numDirections * hiddenSize)}, outputActivations->dataType(), contextPtr);
        argOutput = &outputH;
    }

    if(dataFormat == 1){
        permutedX = input->permute({1, 0, 2}).dup('c');
        argX = &permutedX; 
    }

    auto xData = argX->specialBuffer();
    auto yData = argOutput ? argOutput->specialBuffer() : nullptr;

    if (training) {
          callCudnnIfNoErr(err, cudnnRNNForwardTraining, handle,  rnnDesc, (int) maxSeqLength, xDescList.getDescriptors(), xData,
                        hxDesc, prevActData, cxDesc, prevMemCellData, wDesc,
                       weightsSpace, yDescList.getDescriptors(), yData, hyDesc,
                       finalTimeStepActivationsData,  cyDesc, finalMemCellStateData, workSpace,
                       workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
    } else {
          callCudnnIfNoErr(err, cudnnRNNForwardInference, handle,  rnnDesc, (int) maxSeqLength, xDescList.getDescriptors(), xData,
                        hxDesc, prevActData, cxDesc, prevMemCellData,  wDesc,
                       weightsSpace, yDescList.getDescriptors(), yData,  hyDesc,
                       finalTimeStepActivationsData, cyDesc, finalMemCellStateData, workSpace,
                       workSpaceSizeInBytes);
    }

    //remap output
    if(outputActivations!=nullptr && argOutput!=outputActivations){
        //refill output
        if(dataFormat == 1){
            outputActivations->assign( argOutput->permute({1, 0, 2}));
        }
    }
    NDArray::registerSpecialUse({outputActivations, finalTimeStepActivations, finalMemCellState}, {input, inputWeights, recurrentWeights, biases, prevAct, prevMemCell});

    cudaFree(droupoutMem);
    cudaFree(weightsSpace);
    cudaFree(workSpace);
    cudaFree(reserveSpace);

    return err;

}

#if CUDNN_VERSION >= CUDNN_NEW_RNN_API_VER

cudnnStatus_t cudnn_rnn_v8(LaunchContext  *contextPtr, int dataFormat,  NDArray *input, NDArray *seqLengthArray, NDArray *inputWeights,  NDArray *recurrentWeights,
    NDArray *biases, NDArray *prevAct, NDArray *prevMemCell, NDArray *outputActivations, NDArray *finalTimeStepActivations, NDArray *finalMemCellState,
    int maxSeqLength, int batchSize, int inputSize, int hiddenSize, double cellClip, bool isBidirectional){
    nd4j_debug("cudnn rnn api %s \n", "v8");
    //seqLengthArray should be int
    NDArray *argSeqNdArray = nullptr;
    NDArray seqArrIntData;
    if(seqLengthArray){
        if(seqLengthArray->ews()==1 && seqLengthArray->dataType()==DataType::INT32){
            argSeqNdArray = seqLengthArray;
        }else{
            if(seqLengthArray->dataType()!=DataType::INT32){
                seqArrIntData = seqLengthArray->cast(DataType::INT32);
                if(seqArrIntData.ews()!=1) seqArrIntData=seqArrIntData.dup('c');
            }else{
                seqArrIntData = seqLengthArray->dup('c');
            }
            argSeqNdArray = &seqArrIntData;
        }
    }else{
        seqArrIntData = NDArray('c', std::vector<Nd4jLong>{batchSize}, DataType::INT32, contextPtr);
        seqArrIntData.assign(maxSeqLength);
        argSeqNdArray = &seqArrIntData;
    }

    bool training = false;
    cudnnHandle_t handle = *(reinterpret_cast<cudnnHandle_t *>(contextPtr->getCuDnnHandle()));
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    auto stream = *(contextPtr->getCudaStream());
    callCudnnIfNoErr(err, cudnnSetStream, handle, stream); 

    auto cudnnType = cudnnDataType(input->dataType());
    auto dataTypeSize = input->sizeOfT(); 

    CudnnTensor hDesc, cDesc;  

    constexpr int rankOf = 3;
    const int numDirections     = isBidirectional ? 2 : 1;

    const int dimC[rankOf]    = {numLayers * numDirections,  batchSize,  hiddenSize};
    const int strideC[rankOf]  = {batchSize * hiddenSize, hiddenSize, 1}; 

    hDesc.set(err, cudnnType, rankOf, dimC, strideC);
    cDesc.set(err, cudnnType, rankOf, dimC, strideC);

    //dropout section
    DropoutDesc dropoutDesc; 
    //dropout
    float dropout = 0;
    size_t sizeInBytes=0;
    void *droupoutMem = nullptr;
    uint64_t seed = 1; //seed
    if(dropout!=0){
       // dropoutDesc.create();
        callCudnnIfNoErr(err, cudnnDropoutGetStatesSize, handle, &sizeInBytes);
        //allocate and set
        cudaMalloc(&droupoutMem, sizeInBytes);
        dropoutDesc.set(err, handle, dropout, droupoutMem, sizeInBytes, seed );
    }

    //RNN
    RnnDesc rnnDesc;
    cudnnRNNMode_t rnnCellMode = CUDNN_LSTM;
    cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
    auto direction = isBidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    auto mathPrec = cudnnType;

    //Note: We will set some parameters manually. Some of them could be parameter in future
    constexpr auto inputMode = CUDNN_LINEAR_INPUT;
    bool use_tensor_ops = false ; // could be parameter in future
#if CUDNN_VERSION >= CUDNN_NEW_RNN_API_VER
    cudnnMathType_t mathType = use_tensor_ops ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
#else
    cudnnMathType_t mathType = use_tensor_ops ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
#endif
    //disable projection
    int projSize = hiddenSize;
    cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_DOUBLE_BIAS;
    uint32_t aux_flags = CUDNN_RNN_PADDED_IO_ENABLED;

    rnnDesc.set(err, algo, rnnCellMode, bias_mode, direction, inputMode, cudnnType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, aux_flags);
    if(cellClip>0){
        callCudnnIfNoErr(err, cudnnRNNSetClip, handle, rnnDesc, CUDNN_RNN_CLIP_MINMAX, CUDNN_PROPAGATE_NAN, -cellClip, cellClip);
    }
    //set Data desc
    RnnDataDesc xDataDesc, yDataDesc;
    bool time_major = false;
    float padding_fill = 0.0f;
    auto hostSeqArr = bufferInHost<int>(*argSeqNdArray);
    cudnnRNNDataLayout_t layout = dataFormat==0 ? CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED : CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
    xDataDesc.set(err, cudnnType, layout, maxSeqLength, batchSize, inputSize, hostSeqArr, (void*)&padding_fill);
    yDataDesc.set(err, cudnnType, layout, maxSeqLength, batchSize, hiddenSize * numDirections, hostSeqArr, (void*)&padding_fill);
    //set up parameters
    size_t weightsSize=0;
    callCudnnIfNoErr(err, cudnnGetRNNWeightSpaceSize, handle, rnnDesc, &weightsSize);

    //allocation
    uint8_t *weightsSpace =nullptr;
    cudaMalloc(&weightsSpace,  weightsSize); 

    // Set up work space and reserved memory
    void *workSpace = nullptr;
    void *reserveSpace = nullptr;

    size_t workSpaceSizeInBytes = 0 ;
    size_t reserveSpaceSizeInBytes = 0;

    cudnnForwardMode_t fwdMode = training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE; 
    callCudnnIfNoErr(err, cudnnGetRNNTempSpaceSizes, handle, rnnDesc, fwdMode, xDataDesc, &workSpaceSizeInBytes, &reserveSpaceSizeInBytes);
    cudaMalloc(&workSpace, workSpaceSizeInBytes);
    // training
    if(training) {
        cudaMalloc(&reserveSpace, reserveSpaceSizeInBytes);
    }

    NDArray::prepareSpecialUse({outputActivations, finalTimeStepActivations, finalMemCellState}, {input, inputWeights, recurrentWeights, biases, prevAct, prevMemCell, argSeqNdArray});

    auto xData = input->specialBuffer();
    uint8_t *biasesData = biases ? (uint8_t*)biases->specialBuffer() : nullptr;
    auto prevActData = prevAct ? prevAct->specialBuffer() : nullptr;
    auto prevMemCellData = prevMemCell ? prevMemCell->specialBuffer() : nullptr;
    auto yData = outputActivations ? outputActivations->specialBuffer() : nullptr;
    auto finalTimeStepActivationsData = finalTimeStepActivations ? finalTimeStepActivations->specialBuffer() : nullptr;
    auto finalMemCellStateData = finalMemCellState ? finalMemCellState->specialBuffer() : nullptr;

    // dimension 4*nOut implies order it, ft, c't, ot
    // input gate, forget gate, new gate, output gate, input gate, forget gate, new gate, output gate
    // Note: our weights should be transposed and duplicated with C order to match cudnn ones

    NDArray inputWeightsT, recurrentWeightsT;
    uint8_t *inputWeightsData = nullptr;
    uint8_t *recurrentWeightsData = nullptr;
    if(inputWeights){
      inputWeightsT= inputWeights->rankOf()==3?inputWeights->permute({0, 2, 1}).dup('c'):inputWeights->transpose().dup('c');
      inputWeightsData = (uint8_t*)inputWeightsT.specialBuffer();
    }
    if(recurrentWeights){
      recurrentWeightsT = recurrentWeights->rankOf()==3?recurrentWeights->permute({0, 2, 1}).dup('c'):recurrentWeights->transpose().dup('c');
      recurrentWeightsData =  (uint8_t*)recurrentWeightsT.specialBuffer();
    }

    //copy without cudnnGetRNNLinLayerMatrixParams
    copyWeights(stream, isBidirectional, weightsSpace, weightsSize, inputWeightsData, recurrentWeightsData, biasesData, inputSize, hiddenSize, dataTypeSize);

    callCudnnIfNoErr(err, cudnnRNNForward, handle, rnnDesc, fwdMode, (const int32_t *)argSeqNdArray->specialBuffer(),  xDataDesc, xData,
        yDataDesc, yData, hDesc, prevActData,finalTimeStepActivationsData,  cDesc, prevMemCellData, finalMemCellStateData,
        weightsSize, weightsSpace, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

    NDArray::registerSpecialUse({outputActivations, finalTimeStepActivations, finalMemCellState}, {input, inputWeights, recurrentWeights, biases, prevAct, prevMemCell});

    cudaFree(droupoutMem);
    cudaFree(weightsSpace);
    cudaFree(workSpace);
    cudaFree(reserveSpace);

    return err;

}

#endif

 //////////////////////////////////////////////////////////////////////////
 PLATFORM_IMPL(lstmLayer, ENGINE_CUDA) {
 
    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nOut] (for ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)
 
    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLenArray  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH     = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
 
    const auto cellClip = T_ARG(0);    // cell clipping value, if it = 0 then do not apply clipping
 
    const auto x  = INPUT_VARIABLE(0);        // input
    const auto Wx = INPUT_VARIABLE(1);        // input weights
    const auto Wr = INPUT_VARIABLE(2);        // recurrent weights
 
    int count = 3;
    const auto b     = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto seqLengthArray = hasSeqLenArray ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector
    const auto hI    = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI    = hasInitC  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state
    const auto Wp    = hasPH    ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights

    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;         // output
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;         // output at last step
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;         // cell state at last step

    REQUIRE_TRUE(cellClip >= 0 , 0, "LSTM_LAYER operation: cell clipping value should be nonnegative (>=0) !");
    REQUIRE_TRUE(retFullSeq || retLastH || retLastC, 0, "LSTM_LAYER operation: please specify what output arrays to produce !");
    // evaluate dimensions
    const Nd4jLong seqLength = dataFormat == 3 ?  x->sizeAt(0) : x->sizeAt(dataFormat);
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(1);
    const Nd4jLong nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(2);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;
    const Nd4jLong hiddenSize = nOut;

    auto contextPtr = block.launchContext();
    bool isBidirectional =  directionMode >= 2;

    if(!isBidirectional) {     // no bidirectional
        // Wx validation
        if(Wx->rankOf() != 2 || Wx->sizeAt(0) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 1 || b->sizeAt(0) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 2 || hI->sizeAt(0) != bS || hI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 2 || cI->sizeAt(0) != bS || cI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
    }
    else {                  // bidirectional
         // Wx validation
        if(Wx->rankOf() != 3 || Wx->sizeAt(0) != 2 || Wx->sizeAt(1) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 3 || Wr->sizeAt(0) != 2 || Wr->sizeAt(1) != nOut || Wr->sizeAt(2) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 2 || b->sizeAt(0) != 2 || b->sizeAt(1) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 3 || hI->sizeAt(0) != 2 || hI->sizeAt(1) != bS || hI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 3 || cI->sizeAt(0) != 2 || cI->sizeAt(1) != bS || cI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
    }

    cudnnStatus_t err;
#if CUDNN_VERSION < CUDNN_NEW_RNN_API_VER
    err = cudnn_rnn_old( contextPtr, dataFormat, x, Wx, Wr, b, hI, cI, h, hL, cL,  seqLength, bS, nIn, hiddenSize, (double)cellClip, isBidirectional);
#else
    if(cudnnGetVersion() >= CUDNN_NEW_RNN_API_VER){
        err = cudnn_rnn_v8( contextPtr, dataFormat, x, seqLengthArray, Wx, Wr, b, hI, cI, h, hL, cL,  seqLength, bS, nIn, hiddenSize, (double)cellClip, isBidirectional);
    }else{
        err = cudnn_rnn_old( contextPtr, dataFormat, x, Wx, Wr, b, hI, cI, h, hL, cL,  seqLength, bS, nIn, hiddenSize, (double)cellClip, isBidirectional);
    }
#endif
    if(err!=CUDNN_STATUS_SUCCESS) throw sd::cuda_exception::build("lstmLayer CUDNN call failure ", err);
    return Status::OK();
 }

// Cudnn Lstm:
// Forward inference implemented using v6, and v8 (when version > 8.0.1) api calls.
// As our Cuda Lstm implementation has 1 layer. Cudnn implementation was implemented for 1 physical layer
// Cudnn helper restrictions:
//  - all NDArrays should be the same type
//  - dataFormat should be 0 or 1
//  - only unidirectional (directionMode == 0) and bidirectional concat (directionMode == 3)
//  - no peephole connection
//  - Clipping is allowed for cudnn version >= 7.2.1
//  - SeqLen array is allowed for cudnn version >= 8.0.1
//  - gateActivation: sigmoid, cellActivation and outputActivation: tanh
//  - NDArrays (excluding the weight arrays, as we have to transpose or permute it) should follow 'c' order and ews()==1
 PLATFORM_CHECK(lstmLayer, ENGINE_CUDA) {
 
    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nOut] (for ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)
    // integer numbers corresponding to activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
    const auto gateAct      = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates
    const auto cellAct      = INT_ARG(3);    // activation for cell state (c)
    const auto outAct       = INT_ARG(4);    // activation for output (h)

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLenArray  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH     = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
 
    const auto cellClip = T_ARG(0);  // cell clipping value, if it = 0 then do not apply clipping
 
    const auto x  = INPUT_VARIABLE(0);        // input
    const auto Wx = INPUT_VARIABLE(1);        // input weights
    const auto Wr = INPUT_VARIABLE(2);        // recurrent weights
 
    int count = 3;
    const auto b     = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto hI    = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI    = hasInitC  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state
 
    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;         // output
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;         // output at last step
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;         // cell state at last step
 
    DataType xType  = x->dataType();
    DataType WxType = Wx->dataType();
    DataType WrType = Wr->dataType();
    DataType bType  = b  != nullptr ? b->dataType() : xType;
    DataType hIType = hI != nullptr ? hI->dataType() : xType;
    DataType cIType = cI != nullptr ? cI->dataType() : xType;
    DataType hType  = h  != nullptr ? h->dataType()  : xType;
    DataType hLType = hL != nullptr ? hL->dataType() : xType;
    DataType cLType = cL != nullptr ? cL->dataType() : xType;

    //cudnn related restrictions    //gateAct: sigmoid, cellAct: tanh adn et cetera
    bool implRestrictions = gateAct ==2 && cellAct == 0 && outAct == 0 && 
                       !hasPH && (directionMode==0 || directionMode==3);

    //cudnn api version related restrictions in our helpers
    size_t cudnn_version = cudnnGetVersion();
    //though seqlengthArray was added in earlier versions we do not handle it below 8.0.0.1
#if CUDNN_VERSION < CUDNN_NEW_RNN_API_VER
    implRestrictions = implRestrictions && !hasSeqLenArray;
#else
    implRestrictions = implRestrictions && (cudnn_version >= CUDNN_NEW_RNN_API_VER || !hasSeqLenArray);
#endif
    implRestrictions = implRestrictions && (cudnn_version >= CUDNN_CLIPPING_API_VER || cellClip==0);

    //restriction that comes either from not setting Descriptor or not handling manipulation:
    //restrict0: the same types
    bool inputRestrictions = WxType == xType && xType == WrType && bType == xType && 
                    xType == hIType && cIType == xType && xType == hType &&
                    hLType == xType && xType == cLType; 
    //restrict1: format and some input output shapes
    inputRestrictions = inputRestrictions && (dataFormat ==0 || dataFormat ==1) &&
                (!x  || (x && x->ordering() == 'c'   && x->ews() == 1))  &&
                (!h  || (h && h->ordering() == 'c'   && h->ews() == 1))  &&
                (!hI || (hI && hI->ordering() == 'c' && hI->ews() == 1)) &&
                (!cI || (cI && cI->ordering() == 'c' && cI->ews() == 1)) &&
                (!hL || (hL && hL->ordering() == 'c' && hL->ews() == 1)) &&
                (!cL || (cL && cL->ordering() == 'c' && cL->ews() == 1));

    return implRestrictions && inputRestrictions;
 }
 
 
 
 }
 }
 }
