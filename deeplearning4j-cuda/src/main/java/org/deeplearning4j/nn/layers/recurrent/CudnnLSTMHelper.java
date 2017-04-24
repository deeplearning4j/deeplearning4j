/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.nn.layers.recurrent;

import java.util.Arrays;
import java.util.Map;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.cuda.*;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * cuDNN-based helper for the recurrent LSTM layer (no peephole connections).
 *
 * @author saudet
 */
public class CudnnLSTMHelper implements LSTMHelper {
    protected static final Logger log = LoggerFactory.getLogger(CudnnLSTMHelper.class);

    static void checkCuda(int error) {
        if (error != cudaSuccess) {
            throw new RuntimeException("CUDA error = " + error + ": " + cudaGetErrorString(error).getString());
        }
    }

    static void checkCudnn(int status) {
        if (status != CUDNN_STATUS_SUCCESS) {
            throw new RuntimeException("cuDNN status = " + status + ": " + cudnnGetErrorString(status).getString());
        }
    }

    static class CudnnContext extends cudnnContext {

        static class Deallocator extends CudnnContext implements Pointer.Deallocator {
            Deallocator(CudnnContext c) {
                super(c);
            }

            @Override
            public void deallocate() {
                destroyHandles();
            }
        }

        cudnnTensorStruct hxDesc = new cudnnTensorStruct(), cxDesc = new cudnnTensorStruct();
        cudnnTensorStruct hyDesc = new cudnnTensorStruct(), cyDesc = new cudnnTensorStruct();
        cudnnTensorStruct dhxDesc = new cudnnTensorStruct(), dcxDesc = new cudnnTensorStruct();
        cudnnTensorStruct dhyDesc = new cudnnTensorStruct(), dcyDesc = new cudnnTensorStruct();

        cudnnFilterStruct wDesc = new cudnnFilterStruct(), dwDesc = new cudnnFilterStruct();
        cudnnFilterStruct linLayerMatDesc = new cudnnFilterStruct(), linLayerBiasDesc = new cudnnFilterStruct();

        cudnnRNNStruct rnnDesc = new cudnnRNNStruct();
        cudnnDropoutStruct dropoutDesc = new cudnnDropoutStruct();
        cudnnActivationStruct activationDesc = new cudnnActivationStruct();

        CudnnContext() {
            // insure that cuDNN initializes on the same device as ND4J for this thread
            Nd4j.create(1);
            AtomicAllocator.getInstance();
            createHandles();
            deallocator(new Deallocator(this));
        }

        CudnnContext(CudnnContext c) {
            super(c);
            hxDesc = new cudnnTensorStruct(c.hxDesc);
            cxDesc = new cudnnTensorStruct(c.cxDesc);
            hyDesc = new cudnnTensorStruct(c.hyDesc);
            cyDesc = new cudnnTensorStruct(c.cyDesc);
            dhxDesc = new cudnnTensorStruct(c.dhxDesc);
            dcxDesc = new cudnnTensorStruct(c.dcxDesc);
            dhyDesc = new cudnnTensorStruct(c.dhyDesc);
            dcyDesc = new cudnnTensorStruct(c.dcyDesc);

            wDesc = new cudnnFilterStruct(c.wDesc);
            dwDesc = new cudnnFilterStruct(c.dwDesc);
            linLayerMatDesc = new cudnnFilterStruct(c.linLayerMatDesc);
            linLayerBiasDesc = new cudnnFilterStruct(c.linLayerBiasDesc);

            rnnDesc = new cudnnRNNStruct(c.rnnDesc);
            dropoutDesc = new cudnnDropoutStruct(c.dropoutDesc);
            activationDesc = new cudnnActivationStruct(c.activationDesc);
        }

        void createHandles() {
            checkCudnn(cudnnCreate(this));

            checkCudnn(cudnnCreateTensorDescriptor(hxDesc));
            checkCudnn(cudnnCreateTensorDescriptor(cxDesc));
            checkCudnn(cudnnCreateTensorDescriptor(hyDesc));
            checkCudnn(cudnnCreateTensorDescriptor(cyDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dhxDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dcxDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dhyDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dcyDesc));

            checkCudnn(cudnnCreateFilterDescriptor(wDesc));
            checkCudnn(cudnnCreateFilterDescriptor(dwDesc));
            checkCudnn(cudnnCreateFilterDescriptor(linLayerMatDesc));
            checkCudnn(cudnnCreateFilterDescriptor(linLayerBiasDesc));

            checkCudnn(cudnnCreateRNNDescriptor(rnnDesc));
            checkCudnn(cudnnCreateDropoutDescriptor(dropoutDesc));
            checkCudnn(cudnnCreateActivationDescriptor(activationDesc));
        }

        void destroyHandles() {
            checkCudnn(cudnnDestroyActivationDescriptor(activationDesc));
            checkCudnn(cudnnDestroyDropoutDescriptor(dropoutDesc));
            checkCudnn(cudnnDestroyRNNDescriptor(rnnDesc));

            checkCudnn(cudnnDestroyFilterDescriptor(wDesc));
            checkCudnn(cudnnDestroyFilterDescriptor(dwDesc));
            checkCudnn(cudnnDestroyFilterDescriptor(linLayerMatDesc));
            checkCudnn(cudnnDestroyFilterDescriptor(linLayerBiasDesc));

            checkCudnn(cudnnDestroyTensorDescriptor(hxDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cxDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(hyDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(cyDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dhxDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dcxDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dhyDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dcyDesc));

            checkCudnn(cudnnDestroy(this));
        }
    }

    static class DataCache extends Pointer {

        static class Deallocator extends DataCache implements Pointer.Deallocator {
            Deallocator(DataCache c) {
                super(c);
            }

            @Override
            public void deallocate() {
                checkCuda(cudaFree(this));
                setNull();
            }
        }

        static class HostDeallocator extends DataCache implements Pointer.Deallocator {
            HostDeallocator(DataCache c) {
                super(c);
            }

            @Override
            public void deallocate() {
                checkCuda(cudaFreeHost(this));
                setNull();
            }
        }

        DataCache() {}

        DataCache(long size) {
            position = 0;
            limit = capacity = size;
            int error = cudaMalloc(this, size);
            if (error != cudaSuccess) {
                log.warn("Cannot allocate " + size + " bytes of device memory (CUDA error = " + error
                                + "), proceeding with host memory");
                checkCuda(cudaMallocHost(this, size));
                deallocator(new HostDeallocator(this));
            } else {
                deallocator(new Deallocator(this));
            }
        }

        DataCache(DataCache c) {
            super(c);
        }
    }

    static class TensorArray extends PointerPointer<cudnnTensorStruct> {

        static class Deallocator extends TensorArray implements Pointer.Deallocator {
            Pointer owner;

            Deallocator(TensorArray a, Pointer owner) {
                this.address = a.address;
                this.capacity = a.capacity;
                this.owner = owner;
            }

            @Override
            public void deallocate() {
                for (int i = 0; i < capacity; i++) {
                    cudnnTensorStruct t = this.get(cudnnTensorStruct.class, i);
                    checkCudnn(cudnnDestroyTensorDescriptor(t));
                }
                owner.deallocate();
                owner = null;
                setNull();
            }
        }

        TensorArray() {}

        TensorArray(long size) {
            PointerPointer p = new PointerPointer(size);
            p.deallocate(false);
            this.address = p.address();
            this.limit = p.limit();
            this.capacity = p.capacity();

            cudnnTensorStruct t = new cudnnTensorStruct();
            for (int i = 0; i < capacity; i++) {
                checkCudnn(cudnnCreateTensorDescriptor(t));
                this.put(i, t);
            }
            deallocator(new Deallocator(this, p));
        }

        TensorArray(TensorArray a) {
            super(a);
        }
    }

    int numLayers = 1;
    float dropout = 0;
    boolean bidirectional = false;
    int RNNMode = CUDNN_LSTM;
    int numLinearLayers = 8; // CUDNN_LSTM

    CudnnContext cudnnContext = new CudnnContext();
    TensorArray xDesc = new TensorArray();
    TensorArray yDesc = new TensorArray();
    TensorArray dxDesc = new TensorArray();
    TensorArray dyDesc = new TensorArray();
    DataCache stateSpace = new DataCache();
    DataCache workSpace = new DataCache();
    DataCache reserveSpace = new DataCache();
    DataCache weightsSpace = new DataCache();
    int dataType = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? CUDNN_DATA_DOUBLE
                 : Nd4j.dataType() == DataBuffer.Type.FLOAT ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
    int dataTypeSize = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? 8
                     : Nd4j.dataType() == DataBuffer.Type.FLOAT ? 4 : 2;
    int tensorFormat = CUDNN_TENSOR_NCHW;
    Pointer alpha = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? new DoublePointer(1.0)
                  : Nd4j.dataType() == DataBuffer.Type.FLOAT ? new FloatPointer(1.0f)
                                    : new ShortPointer(new short[] {(short) HalfIndexer.fromFloat(1.0f)});
    Pointer beta = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? new DoublePointer(0.0)
                 : Nd4j.dataType() == DataBuffer.Type.FLOAT ? new FloatPointer(0.0f)
                                    : new ShortPointer(new short[] {(short) HalfIndexer.fromFloat(0.0f)});;
    SizeTPointer sizeInBytes = new SizeTPointer(1);

    @Override
    public Pair<Gradient, INDArray> backpropGradient(final NeuralNetConfiguration conf,
                    final IActivation gateActivationFn, final INDArray input, final INDArray recurrentWeights, //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                    final INDArray inputWeights, //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                    final INDArray epsilon, final boolean truncatedBPTT, final int tbpttBackwardLength,
                    final FwdPassReturn fwdPass, final boolean forwards, final String inputWeightKey,
                    final String recurrentWeightKey, final String biasWeightKey,
                    final Map<String, INDArray> gradientViews, INDArray maskArray, //Input mask: should only be used with bidirectional RNNs + variable length
                    final boolean hasPeepholeConnections) {            //True for GravesLSTM, false for LSTM

        //Expect errors to have shape: [miniBatchSize,n^(L+1),timeSeriesLength]
        int hiddenLayerSize = recurrentWeights.size(0); //i.e., n^L
        int prevLayerSize = inputWeights.size(0); //n^(L-1)
        int inputLayerSize = input.size(1);
        int miniBatchSize = epsilon.size(0);
        boolean is2dInput = epsilon.rank() < 3; //Edge case: T=1 may have shape [miniBatchSize,n^(L+1)], equiv. to [miniBatchSize,n^(L+1),1]
        int timeSeriesLength = (is2dInput ? 1 : epsilon.size(2));

        INDArray x = input.permute(2, 0, 1).dup();
        INDArray dx = Nd4j.create(new int[] {timeSeriesLength, miniBatchSize, prevLayerSize});
        INDArray dy = epsilon.permute(2, 0, 1).dup();

        INDArray iwGradientsOut = Nd4j.create(new int[] {4, hiddenLayerSize, inputLayerSize});
        INDArray rwGradientsOut = Nd4j.create(new int[] {4, hiddenLayerSize, hiddenLayerSize});
        INDArray bGradientsOut  = Nd4j.create(new int[] {4, hiddenLayerSize});
        INDArray bGradientsOut2 = bGradientsOut.dup();

        INDArray fwdPassOutput = fwdPass.fwdPassOutputAsArrays[0];
        INDArray memCellState = fwdPass.memCellState[0];
        INDArray memCellActivations = fwdPass.memCellActivations[0];

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(x, recurrentWeights, inputWeights, dy, dx,
                fwdPass.fwdPassOutputAsArrays[0], fwdPass.memCellState[0], fwdPass.memCellActivations[0],
                iwGradientsOut, rwGradientsOut, bGradientsOut, bGradientsOut2);
        Pointer xData = allocator.getPointer(x, context);
        Pointer recurrentWeightsData = allocator.getPointer(recurrentWeights, context);
        Pointer inputWeightsData = allocator.getPointer(inputWeights, context);
        Pointer dyData = allocator.getPointer(dy, context);
        Pointer dxData = allocator.getPointer(dx, context);
        Pointer fwdPassOutputData = allocator.getPointer(fwdPass.fwdPassOutputAsArrays[0], context);
        Pointer memCellStateData = allocator.getPointer(fwdPass.memCellState[0], context);
        Pointer memCellActivationsData = allocator.getPointer(fwdPass.memCellActivations[0], context);
        Pointer iwGradientsOutData = allocator.getPointer(iwGradientsOut, context);
        Pointer rwGradientsOutData = allocator.getPointer(rwGradientsOut, context);
        Pointer bGradientsOutData = allocator.getPointer(bGradientsOut, context);
        Pointer bGradientsOut2Data = allocator.getPointer(bGradientsOut2, context);

        CUstream_st stream = new CUstream_st(context.getOldStream());
        checkCudnn(cudnnSetStream(cudnnContext, stream));

        cudnnTensorStruct xDesc0 = xDesc.get(cudnnTensorStruct.class, 0);

        checkCudnn(cudnnRNNBackwardData(cudnnContext, cudnnContext.rnnDesc, timeSeriesLength, yDesc, fwdPassOutputData, dyDesc, dyData,
                cudnnContext.dhyDesc, null, cudnnContext.dcyDesc, null, cudnnContext.wDesc, weightsSpace, cudnnContext.hxDesc, memCellActivationsData,
                cudnnContext.cxDesc, memCellStateData, dxDesc, dxData, cudnnContext.dhxDesc, null, cudnnContext.dcxDesc, null,
                workSpace, workSpace.limit(), reserveSpace, reserveSpace.limit()));

        // cudnnRNNBackwardWeights adds to the data in dw.
        checkCuda(cudaMemsetAsync(weightsSpace, 0, weightsSpace.limit(), stream));

        checkCudnn(cudnnRNNBackwardWeights(cudnnContext, cudnnContext.rnnDesc, timeSeriesLength, xDesc, xData, cudnnContext.hxDesc, memCellActivationsData,
                yDesc, fwdPassOutputData, workSpace, workSpace.limit(), cudnnContext.dwDesc, weightsSpace, reserveSpace, reserveSpace.limit()));

        for (int layer = 0; layer < numLayers * (bidirectional ? 2 : 1); layer++) {
            for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
                int[] dataType = new int[1];
                int[] format = new int[1];
                int[] nbDims = new int[1];
                int[] filterDimA = new int[3];

                Pointer linLayerMat = new Pointer();

                checkCudnn(cudnnGetRNNLinLayerMatrixParams(cudnnContext, cudnnContext.rnnDesc, layer, xDesc0,
                        cudnnContext.wDesc, weightsSpace, linLayerID, cudnnContext.linLayerMatDesc, linLayerMat));

                checkCudnn(cudnnGetFilterNdDescriptor(cudnnContext.linLayerMatDesc, 3, dataType, format, nbDims, filterDimA));

                Pointer linLayerBias = new Pointer();

                checkCudnn(cudnnGetRNNLinLayerBiasParams(cudnnContext, cudnnContext.rnnDesc, layer, xDesc0,
                        cudnnContext.wDesc, weightsSpace, linLayerID, cudnnContext.linLayerBiasDesc, linLayerBias));

                checkCudnn(cudnnGetFilterNdDescriptor(cudnnContext.linLayerBiasDesc, 3, dataType, format, nbDims, filterDimA));

                // our data is in "new, forget, output, and input gates" order (aka IFOG), each kind of weight packed together
                int position = 0, size = 0;
                Pointer mat = null, bias = null;
                switch (linLayerID) {
                    case 0: mat = iwGradientsOutData; bias = bGradientsOutData;  position = 3; size = inputLayerSize; break; // input gate
                    case 1: mat = iwGradientsOutData; bias = bGradientsOutData;  position = 1; size = inputLayerSize; break; // forget gate
                    case 2: mat = iwGradientsOutData; bias = bGradientsOutData;  position = 0; size = inputLayerSize; break; // memory gate
                    case 3: mat = iwGradientsOutData; bias = bGradientsOutData;  position = 2; size = inputLayerSize; break; // output gate
                    case 4: mat = rwGradientsOutData; bias = bGradientsOut2Data; position = 3; size = hiddenLayerSize; break; // input gate
                    case 5: mat = rwGradientsOutData; bias = bGradientsOut2Data; position = 1; size = hiddenLayerSize; break; // forget gate
                    case 6: mat = rwGradientsOutData; bias = bGradientsOut2Data; position = 0; size = hiddenLayerSize; break; // memory gate
                    case 7: mat = rwGradientsOutData; bias = bGradientsOut2Data; position = 2; size = hiddenLayerSize; break; // output gate
                    default: assert false;
                }
                checkCuda(cudaMemcpyAsync(mat.position(position * size * hiddenLayerSize * dataTypeSize), linLayerMat, size * hiddenLayerSize * dataTypeSize, cudaMemcpyDeviceToDevice, stream));
                checkCuda(cudaMemcpyAsync(bias.position(position * hiddenLayerSize * dataTypeSize), linLayerBias, hiddenLayerSize * dataTypeSize, cudaMemcpyDeviceToDevice, stream));
            }
        }

        allocator.getFlowController().registerActionAllWrite(context, x, recurrentWeights, inputWeights, dy, dx,
                fwdPass.fwdPassOutputAsArrays[0], fwdPass.memCellState[0], fwdPass.memCellActivations[0],
                iwGradientsOut, rwGradientsOut, bGradientsOut, bGradientsOut2);

        bGradientsOut = bGradientsOut.addi(bGradientsOut2);

        Gradient retGradient = new DefaultGradient();
        retGradient.gradientForVariable().put(inputWeightKey, iwGradientsOut/*.permute(0, 1, 2)*/.reshape('f', inputLayerSize, 4 * hiddenLayerSize));
        retGradient.gradientForVariable().put(recurrentWeightKey, rwGradientsOut/*.permute(0, 1, 2)*/.reshape('f', hiddenLayerSize, 4 * hiddenLayerSize));
        retGradient.gradientForVariable().put(biasWeightKey, bGradientsOut.reshape(1, 4 * hiddenLayerSize));

        INDArray epsilonNext = dx.permute(1, 2, 0); //i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]

        return new Pair<>(retGradient, epsilonNext);
    }

    @Override
    public FwdPassReturn activate(final Layer layer, final NeuralNetConfiguration conf,
                    final IActivation gateActivationFn, //Activation function for the gates - sigmoid or hard sigmoid (must be found in range 0 to 1)
                    final INDArray input, final INDArray recurrentWeights, //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                    final INDArray inputWeights, //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                    final INDArray biases, //Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T
                    final boolean training, final INDArray prevOutputActivations,
                    final INDArray prevMemCellState, boolean forBackprop, boolean forwards,
                    final String inputWeightKey, INDArray maskArray, //Input mask: should only be used with bidirectional RNNs + variable length
                    final boolean hasPeepholeConnections) {            //True for GravesLSTM, false for LSTM

        boolean is2dInput = input.rank() < 3; //Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
        int timeSeriesLength = (is2dInput ? 1 : input.size(2));
        int hiddenLayerSize = recurrentWeights.size(0);
        int miniBatchSize = input.size(0);
        int inputLayerSize = input.size(1);

        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        if (timeSeriesLength > xDesc.capacity()) {
            xDesc.deallocate();
            xDesc = new TensorArray(timeSeriesLength);
        }
        if (timeSeriesLength > yDesc.capacity()) {
            yDesc.deallocate();
            yDesc = new TensorArray(timeSeriesLength);
        }
        if (timeSeriesLength > dxDesc.capacity()) {
            dxDesc.deallocate();
            dxDesc = new TensorArray(timeSeriesLength);
        }
        if (timeSeriesLength > dyDesc.capacity()) {
            dyDesc.deallocate();
            dyDesc = new TensorArray(timeSeriesLength);
        }

        for (int i = 0; i < timeSeriesLength; i++) {
            int[] dimA = {miniBatchSize, inputLayerSize, 1};
            int[] strideA = {dimA[2] * dimA[1], dimA[2], 1};

            checkCudnn(cudnnSetTensorNdDescriptor(xDesc.get(cudnnTensorStruct.class, i), dataType, 3, dimA, strideA));
            checkCudnn(cudnnSetTensorNdDescriptor(dxDesc.get(cudnnTensorStruct.class, i), dataType, 3, dimA, strideA));

            int[] dimB = {miniBatchSize, bidirectional ? hiddenLayerSize * 2 : hiddenLayerSize, 1};
            int[] strideB = {dimB[2] * dimB[1], dimB[2], 1};

            checkCudnn(cudnnSetTensorNdDescriptor(yDesc.get(cudnnTensorStruct.class, i), dataType, 3, dimB, strideB));
            checkCudnn(cudnnSetTensorNdDescriptor(dyDesc.get(cudnnTensorStruct.class, i), dataType, 3, dimB, strideB));
        }

        int[] dimC = {numLayers * (bidirectional ? 2 : 1), miniBatchSize, hiddenLayerSize};
        int[] strideC = {dimC[2] * dimC[1], dimC[2], 1};

        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.hxDesc, dataType, 3, dimC, strideC));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.cxDesc, dataType, 3, dimC, strideC));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.hyDesc, dataType, 3, dimC, strideC));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.cyDesc, dataType, 3, dimC, strideC));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.dhxDesc, dataType, 3, dimC, strideC));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.dcxDesc, dataType, 3, dimC, strideC));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.dhyDesc, dataType, 3, dimC, strideC));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.dcyDesc, dataType, 3, dimC, strideC));

        checkCudnn(cudnnDropoutGetStatesSize(cudnnContext, sizeInBytes));
        long stateSize = sizeInBytes.get(0);
        if (stateSize > stateSpace.capacity()) {
            stateSpace.deallocate();
            stateSpace = new DataCache(stateSize);
        }
        stateSpace.limit(stateSize);

        checkCudnn(cudnnSetDropoutDescriptor(cudnnContext.dropoutDesc, cudnnContext,
                        dropout, stateSpace, stateSize, Nd4j.getRandom().getSeed()));

        checkCudnn(cudnnSetRNNDescriptor(cudnnContext.rnnDesc, hiddenLayerSize, numLayers, cudnnContext.dropoutDesc,
                        CUDNN_LINEAR_INPUT, bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, RNNMode, dataType));

        cudnnTensorStruct xDesc0 = xDesc.get(cudnnTensorStruct.class, 0);
        checkCudnn(cudnnGetRNNParamsSize(cudnnContext, cudnnContext.rnnDesc, xDesc0, sizeInBytes, dataType));
        long weightsSize = sizeInBytes.get(0);
        if (weightsSize > weightsSpace.capacity()) {
            weightsSpace.deallocate();
            weightsSpace = new DataCache(weightsSize);
        }
        weightsSpace.limit(weightsSize);

        int[] dimW = {(int)weightsSize / dataTypeSize, 1, 1};

        checkCudnn(cudnnSetFilterNdDescriptor(cudnnContext.wDesc, dataType, CUDNN_TENSOR_NCHW, 3, dimW));
        checkCudnn(cudnnSetFilterNdDescriptor(cudnnContext.dwDesc, dataType, CUDNN_TENSOR_NCHW, 3, dimW));

        checkCudnn(cudnnGetRNNWorkspaceSize(cudnnContext, cudnnContext.rnnDesc, timeSeriesLength, xDesc, sizeInBytes));
        long workSize = sizeInBytes.get(0);
        if (workSize > workSpace.capacity()) {
            workSpace.deallocate();
            workSpace = new DataCache(workSize);
        }
        workSpace.limit(workSize);

        checkCudnn(cudnnGetRNNTrainingReserveSize(cudnnContext, cudnnContext.rnnDesc, timeSeriesLength, xDesc, sizeInBytes));
        long reserveSize = sizeInBytes.get(0);
        if (reserveSize > reserveSpace.capacity()) {
            reserveSpace.deallocate();
            reserveSpace = new DataCache(reserveSize);
        }
        reserveSpace.limit(reserveSize);

        INDArray x = input.permute(2, 0, 1).dup();
        INDArray linInputWeights = inputWeights.reshape(4, hiddenLayerSize, inputLayerSize);//.permute(0, 1, 2).dup();
        INDArray linRecurrentWeights = recurrentWeights.reshape(4, hiddenLayerSize, hiddenLayerSize);//.permute(0, 1, 2).dup();
        INDArray linHalfBiases = biases.reshape(4, hiddenLayerSize).mul(0.5);
        INDArray prevAct = prevOutputActivations.permute(1, 2, 0).dup();
        INDArray prevMemCell = prevMemCellState.permute(1, 2, 0).dup();
//        INDArray outputActivations = null;

        FwdPassReturn toReturn = new FwdPassReturn();
//        if (forBackprop) {
            toReturn.fwdPassOutputAsArrays = new INDArray[] {Nd4j.create(new int[] {timeSeriesLength, miniBatchSize, bidirectional ? hiddenLayerSize * 2 : hiddenLayerSize})};
            toReturn.memCellState = new INDArray[] {Nd4j.create(new int[] {numLayers * (bidirectional ? 2 : 1), miniBatchSize, hiddenLayerSize})};
            toReturn.memCellActivations = new INDArray[] {Nd4j.create(new int[] {numLayers * (bidirectional ? 2 : 1), miniBatchSize, hiddenLayerSize})};
//        } else {
//            outputActivations = Nd4j.create(new int[] {miniBatchSize, bidirectional ? hiddenLayerSize * 2 : hiddenLayerSize, timeSeriesLength}, 'f');
//            toReturn.fwdPassOutput = outputActivations;
//        }

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(x,
                        linInputWeights, linRecurrentWeights, linHalfBiases, prevAct, prevMemCell,
                        toReturn.fwdPassOutputAsArrays[0], toReturn.memCellState[0], toReturn.memCellActivations[0]/*, outputActivations*/);
        Pointer xData = allocator.getPointer(x, context);
        Pointer linInputWeightsData = allocator.getPointer(linInputWeights, context);
        Pointer linRecurrentWeightsData = allocator.getPointer(linRecurrentWeights, context);
        Pointer linHalfBiasesData = allocator.getPointer(linHalfBiases, context);
        Pointer prevActData = allocator.getPointer(prevAct, context);
        Pointer prevMemCellData = allocator.getPointer(prevMemCell, context);
        Pointer fwdPassOutputData = allocator.getPointer(toReturn.fwdPassOutputAsArrays[0], context);
        Pointer memCellStateData = allocator.getPointer(toReturn.memCellState[0], context);
        Pointer memCellActivationsData = allocator.getPointer(toReturn.memCellActivations[0], context);
//        Pointer outputActivationsData = allocator.getPointer(outputActivations, context);

        CUstream_st stream = new CUstream_st(context.getOldStream());
        checkCudnn(cudnnSetStream(cudnnContext, stream));

        checkCuda(cudaMemsetAsync(weightsSpace, 0, weightsSpace.limit(), stream));

        for (int layerNum = 0; layerNum < numLayers * (bidirectional ? 2 : 1); layerNum++) {
            for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
                int[] dataType = new int[1];
                int[] format = new int[1];
                int[] nbDims = new int[1];
                int[] filterDimA = new int[3];

                Pointer linLayerMat = new Pointer();

                checkCudnn(cudnnGetRNNLinLayerMatrixParams(cudnnContext, cudnnContext.rnnDesc, layerNum, xDesc0,
                        cudnnContext.wDesc, weightsSpace, linLayerID, cudnnContext.linLayerMatDesc, linLayerMat));

                checkCudnn(cudnnGetFilterNdDescriptor(cudnnContext.linLayerMatDesc, 3, dataType, format, nbDims, filterDimA));

                Pointer linLayerBias = new Pointer();

                checkCudnn(cudnnGetRNNLinLayerBiasParams(cudnnContext, cudnnContext.rnnDesc, layerNum, xDesc0,
                        cudnnContext.wDesc, weightsSpace, linLayerID, cudnnContext.linLayerBiasDesc, linLayerBias));

                checkCudnn(cudnnGetFilterNdDescriptor(cudnnContext.linLayerBiasDesc, 3, dataType, format, nbDims, filterDimA));

                // our data is in "new, forget, output, and input gates" order (aka IFOG), each kind of weight packed together
                int position = 0, size = 0;
                Pointer data = null;
                switch (linLayerID) {
                    case 0: data = linInputWeightsData;     position = 3; size = inputLayerSize;  break; // input gate
                    case 1: data = linInputWeightsData;     position = 1; size = inputLayerSize;  break; // forget gate
                    case 2: data = linInputWeightsData;     position = 0; size = inputLayerSize;  break; // new gate
                    case 3: data = linInputWeightsData;     position = 2; size = inputLayerSize;  break; // output gate
                    case 4: data = linRecurrentWeightsData; position = 3; size = hiddenLayerSize; break; // input gate
                    case 5: data = linRecurrentWeightsData; position = 1; size = hiddenLayerSize; break; // forget gate
                    case 6: data = linRecurrentWeightsData; position = 0; size = hiddenLayerSize; break; // new gate
                    case 7: data = linRecurrentWeightsData; position = 2; size = hiddenLayerSize; break; // output gate
                    default: assert false;
                }
                checkCuda(cudaMemcpyAsync(linLayerMat, data.position(position * size * hiddenLayerSize * dataTypeSize), size * hiddenLayerSize * dataTypeSize, cudaMemcpyDeviceToDevice, stream));
                checkCuda(cudaMemcpyAsync(linLayerBias, linHalfBiasesData.position(position * hiddenLayerSize * dataTypeSize), hiddenLayerSize * dataTypeSize, cudaMemcpyDeviceToDevice, stream));
            }
        }

        if (training) {
            checkCudnn(cudnnRNNForwardTraining(cudnnContext, cudnnContext.rnnDesc, timeSeriesLength,
                            xDesc, xData, cudnnContext.hxDesc, prevActData, cudnnContext.cxDesc, prevMemCellData,
                            cudnnContext.wDesc, weightsSpace, yDesc, fwdPassOutputData, cudnnContext.hyDesc, memCellActivationsData,
                            cudnnContext.cyDesc, memCellStateData, workSpace, workSpace.limit(), reserveSpace, reserveSpace.limit()));
        } else {
            checkCudnn(cudnnRNNForwardInference(cudnnContext, cudnnContext.rnnDesc, timeSeriesLength,
                            xDesc, xData, cudnnContext.hxDesc, prevActData, cudnnContext.cxDesc, prevMemCellData,
                            cudnnContext.wDesc, weightsSpace, yDesc, fwdPassOutputData, cudnnContext.hyDesc, memCellActivationsData,
                            cudnnContext.cyDesc, memCellStateData, workSpace, workSpace.limit()));
        }

        allocator.getFlowController().registerActionAllWrite(context, x,
                        linInputWeights, linRecurrentWeights, linHalfBiases, prevAct, prevMemCell,
                        toReturn.fwdPassOutputAsArrays[0], toReturn.memCellState[0], toReturn.memCellActivations[0]/*, outputActivations*/);

        if (!forBackprop) {
            toReturn.fwdPassOutput = toReturn.fwdPassOutputAsArrays[0].permute(1, 2, 0);
            toReturn.lastAct = toReturn.memCellActivations[0].permute(1, 2, 0);
            toReturn.lastMemCell = toReturn.memCellState[0].permute(1, 2, 0);
        }

        return toReturn;
    }

}
