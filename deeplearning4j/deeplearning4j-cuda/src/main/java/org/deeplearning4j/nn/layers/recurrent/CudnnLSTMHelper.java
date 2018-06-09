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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseCudnnHelper;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.util.StringUtils;

import java.util.Map;

import static org.bytedeco.javacpp.cuda.*;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * cuDNN-based helper for the recurrent LSTM layer (no peephole connections).
 *
 * @author saudet
 */
@Slf4j
public class CudnnLSTMHelper extends BaseCudnnHelper implements LSTMHelper {

    private static class CudnnLSTMContext extends CudnnContext {

        private static class Deallocator extends CudnnLSTMContext implements Pointer.Deallocator {
            Deallocator(CudnnLSTMContext c) {
                super(c);
            }

            @Override
            public void deallocate() {
                destroyHandles();
            }
        }

        private cudnnTensorStruct hxDesc = new cudnnTensorStruct(), cxDesc = new cudnnTensorStruct();
        private cudnnTensorStruct hyDesc = new cudnnTensorStruct(), cyDesc = new cudnnTensorStruct();
        private cudnnTensorStruct dhxDesc = new cudnnTensorStruct(), dcxDesc = new cudnnTensorStruct();
        private cudnnTensorStruct dhyDesc = new cudnnTensorStruct(), dcyDesc = new cudnnTensorStruct();

        private cudnnFilterStruct wDesc = new cudnnFilterStruct(), dwDesc = new cudnnFilterStruct();
        private cudnnFilterStruct linLayerMatDesc = new cudnnFilterStruct(), linLayerBiasDesc = new cudnnFilterStruct();

        private cudnnRNNStruct rnnDesc = new cudnnRNNStruct();
        private cudnnDropoutStruct dropoutDesc = new cudnnDropoutStruct();
        private cudnnActivationStruct activationDesc = new cudnnActivationStruct();

        public CudnnLSTMContext() {
            createHandles();
            deallocator(new Deallocator(this));
        }

        public CudnnLSTMContext(CudnnLSTMContext c) {
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

        @Override
        protected void createHandles() {
            super.createHandles();

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

        @Override
        protected void destroyHandles() {
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

            super.destroyHandles();
        }
    }

    // These constants might eventually become variable parameters...
    protected static final int NUM_LAYERS = 1;
    protected static final float DROPOUT = 0;
    protected static final boolean BIDIRECTIONAL = false;
    protected static final int RNN_MODE = CUDNN_LSTM;
    protected static final int NUM_LINEAR_LAYERS = 8; // CUDNN_LSTM

    private CudnnLSTMContext cudnnContext = new CudnnLSTMContext();
    private TensorArray xDesc = new TensorArray();
    private TensorArray yDesc = new TensorArray();
    private TensorArray dxDesc = new TensorArray();
    private TensorArray dyDesc = new TensorArray();
    private DataCache stateSpace = new DataCache();
//    private DataCache workSpace = new DataCache();
    private DataCache reserveSpace = new DataCache();
    private DataCache weightsSpace = new DataCache();

    private static INDArray toCOrder(INDArray arr) {
        if (arr.isView() || arr.ordering() != 'c' || !Shape.strideDescendingCAscendingF(arr)) {
            arr = arr.dup('c');
        }
        return arr;
    }

    @Override
    public boolean checkSupported(IActivation gateActivationFn, IActivation activationFn,
                    boolean hasPeepholeConnections) {
        boolean supported = checkSupported();
        if (!(gateActivationFn instanceof ActivationSigmoid)) {
            supported = false;
            log.warn("Not supported: Gate activation functions != ActivationSigmoid");
        }
        if (!(activationFn instanceof ActivationTanH)) {
            supported = false;
            log.warn("Not supported: Layer activation functions != ActivationTanH");
        }
        if (hasPeepholeConnections) {
            supported = false;
            log.warn("Not supported: LSTM layers with peephole connections");
        }
        return supported;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(final NeuralNetConfiguration conf,
                                                     final IActivation gateActivationFn, final INDArray input, final INDArray recurrentWeights, //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                                                     final INDArray inputWeights, //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                                                     final INDArray epsilon, final boolean truncatedBPTT, final int tbpttBackwardLength,
                                                     final FwdPassReturn fwdPass, final boolean forwards, final String inputWeightKey,
                                                     final String recurrentWeightKey, final String biasWeightKey,
                                                     final Map<String, INDArray> gradientViews, INDArray maskArray, //Input mask: should only be used with bidirectional RNNs + variable length
                                                     final boolean hasPeepholeConnections, //True for GravesLSTM, false for LSTM
                                                     final LayerWorkspaceMgr workspaceMgr) {

        //Expect errors to have shape: [miniBatchSize,n^(L+1),timeSeriesLength]
        val hiddenLayerSize = recurrentWeights.size(0); //i.e., n^L
        val prevLayerSize = inputWeights.size(0); //n^(L-1)
        val inputLayerSize = input.size(1);
        val miniBatchSize = epsilon.size(0);
        boolean is2dInput = epsilon.rank() < 3; //Edge case: T=1 may have shape [miniBatchSize,n^(L+1)], equiv. to [miniBatchSize,n^(L+1),1]
        long timeSeriesLength = (is2dInput ? 1 : epsilon.size(2));

        INDArray x = toCOrder(input.permute(2, 0, 1));
        INDArray dy = toCOrder(epsilon.permute(2, 0, 1));
        INDArray dx = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, new long[] {timeSeriesLength, miniBatchSize, prevLayerSize}, 'c');

        INDArray iwGradientsOut = gradientViews.get(inputWeightKey);
        INDArray rwGradientsOut = gradientViews.get(recurrentWeightKey); //Order: {I,F,O,G}
        INDArray bGradientsOut = gradientViews.get(biasWeightKey);

        INDArray outputActivations = toCOrder(fwdPass.fwdPassOutput.permute(2, 0, 1));
        INDArray prevStepMemCellState = toCOrder(fwdPass.prevMemCell);
        INDArray prevStepActivations = toCOrder(fwdPass.prevAct);

        Nd4j.getExecutioner().commit();

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(x, dy, dx, outputActivations,
                        prevStepMemCellState, prevStepActivations, iwGradientsOut, rwGradientsOut, bGradientsOut);
        Pointer xData = allocator.getPointer(x, context);
        Pointer dyData = allocator.getPointer(dy, context);
        Pointer dxData = allocator.getPointer(dx, context);
        Pointer outputActivationsData = allocator.getPointer(outputActivations, context);
        Pointer prevMemCellStateData = allocator.getPointer(prevStepMemCellState, context);
        Pointer prevStepActivationsData = allocator.getPointer(prevStepActivations, context);
        Pointer iwGradientsOutData = allocator.getPointer(iwGradientsOut, context);
        Pointer rwGradientsOutData = allocator.getPointer(rwGradientsOut, context);
        Pointer bGradientsOutData = allocator.getPointer(bGradientsOut, context);

        CUstream_st stream = new CUstream_st(context.getOldStream());
        checkCudnn(cudnnSetStream(cudnnContext, stream));

        if (truncatedBPTT) {
            val endIdx = Math.max(0, timeSeriesLength - tbpttBackwardLength) * miniBatchSize * hiddenLayerSize;
            xData.position(endIdx * dataTypeSize);
            dyData.position(endIdx * (BIDIRECTIONAL ? 2 : 1) * dataTypeSize);
            outputActivationsData.position(endIdx * (BIDIRECTIONAL ? 2 : 1) * dataTypeSize);
            timeSeriesLength = (int) Math.min(timeSeriesLength, tbpttBackwardLength);
        }

        cudnnTensorStruct xDesc0 = xDesc.get(cudnnTensorStruct.class, 0);

        DataCache workSpace = workspaceMgr.getHelperWorkspace(LayerWorkspaceMgr.CUDNN_WORKSPACE_KEY);
        checkCudnn(cudnnRNNBackwardData(cudnnContext, cudnnContext.rnnDesc, (int) timeSeriesLength, yDesc,
                        outputActivationsData, dyDesc, dyData, cudnnContext.dhyDesc, null, cudnnContext.dcyDesc, null,
                        cudnnContext.wDesc, weightsSpace, cudnnContext.hxDesc, prevStepActivationsData, //hx: initial hidden state of RNN
                        cudnnContext.cxDesc, prevMemCellStateData, //cx: initial cell state of RNN
                        dxDesc, dxData, //dx: gradient at input of each time step
                        cudnnContext.dhxDesc, null, //dhx: gradient at initial hidden state of RNN
                        cudnnContext.dcxDesc, null, //dcx: Gradient at initial cell state
                        workSpace, workSpace.limit(), reserveSpace, reserveSpace.limit()));

        // cudnnRNNBackwardWeights adds to the data in dw.
        checkCuda(cudaMemsetAsync(weightsSpace, 0, weightsSpace.limit(), stream));

        checkCudnn(cudnnRNNBackwardWeights(cudnnContext, cudnnContext.rnnDesc, (int) timeSeriesLength, xDesc, xData, //Input data
                        cudnnContext.hxDesc, prevStepActivationsData, //Initial hidden state
                        yDesc, outputActivationsData, //Output data
                        workSpace, workSpace.limit(), cudnnContext.dwDesc, weightsSpace, reserveSpace,
                        reserveSpace.limit()));

        int[] dataType = new int[1];
        int[] format = new int[1];
        int[] nbDims = new int[1];
        int[] filterDimA = new int[3];
        Pointer linLayerMat = new Pointer();
        Pointer linLayerBias = new Pointer();

        for (int layer = 0; layer < NUM_LAYERS * (BIDIRECTIONAL ? 2 : 1); layer++) {
            for (int linLayerID = 0; linLayerID < NUM_LINEAR_LAYERS; linLayerID++) {
                checkCudnn(cudnnGetRNNLinLayerMatrixParams(cudnnContext, cudnnContext.rnnDesc, layer, xDesc0,
                                cudnnContext.wDesc, weightsSpace, linLayerID, cudnnContext.linLayerMatDesc,
                                linLayerMat));

                checkCudnn(cudnnGetFilterNdDescriptor(cudnnContext.linLayerMatDesc, 3, dataType, format, nbDims,
                                filterDimA));

                checkCudnn(cudnnGetRNNLinLayerBiasParams(cudnnContext, cudnnContext.rnnDesc, layer, xDesc0,
                                cudnnContext.wDesc, weightsSpace, linLayerID, cudnnContext.linLayerBiasDesc,
                                linLayerBias));

                checkCudnn(cudnnGetFilterNdDescriptor(cudnnContext.linLayerBiasDesc, 3, dataType, format, nbDims,
                                filterDimA));

                // our data is in "new, forget, output, and input gates" order (aka IFOG), each kind of weight packed together
                int position = 0;
                long size = 0;
                Pointer data = null;
                switch (linLayerID) {
                    case 0:
                        data = iwGradientsOutData;
                        position = 3;
                        size = inputLayerSize;
                        break; // input gate
                    case 1:
                        data = iwGradientsOutData;
                        position = 1;
                        size = inputLayerSize;
                        break; // forget gate
                    case 2:
                        data = iwGradientsOutData;
                        position = 0;
                        size = inputLayerSize;
                        break; // new gate (input modulation gate)
                    case 3:
                        data = iwGradientsOutData;
                        position = 2;
                        size = inputLayerSize;
                        break; // output gate
                    case 4:
                        data = rwGradientsOutData;
                        position = 3;
                        size = hiddenLayerSize;
                        break; // input gate
                    case 5:
                        data = rwGradientsOutData;
                        position = 1;
                        size = hiddenLayerSize;
                        break; // forget gate
                    case 6:
                        data = rwGradientsOutData;
                        position = 0;
                        size = hiddenLayerSize;
                        break; // new gate (input modulation gate)
                    case 7:
                        data = rwGradientsOutData;
                        position = 2;
                        size = hiddenLayerSize;
                        break; // output gate
                    default:
                        throw new RuntimeException();
                }
                checkCuda(cudaMemcpyAsync(data.position(position * size * hiddenLayerSize * dataTypeSize), linLayerMat,
                                size * hiddenLayerSize * dataTypeSize, cudaMemcpyDeviceToDevice, stream));
                if (linLayerID < 4) {
                    checkCuda(cudaMemcpyAsync(bGradientsOutData.position(position * hiddenLayerSize * dataTypeSize),
                                    linLayerBias, hiddenLayerSize * dataTypeSize, cudaMemcpyDeviceToDevice, stream));
                }
            }
        }

        allocator.getFlowController().registerActionAllWrite(context, x, dy, dx, outputActivations,
                        prevStepMemCellState, prevStepActivations, iwGradientsOut, rwGradientsOut, bGradientsOut);

        Gradient retGradient = new DefaultGradient();
        retGradient.gradientForVariable().put(inputWeightKey, iwGradientsOut);
        retGradient.gradientForVariable().put(recurrentWeightKey, rwGradientsOut);
        retGradient.gradientForVariable().put(biasWeightKey, bGradientsOut);

        INDArray epsilonNext = dx.permute(1, 2, 0); //i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]

        return new Pair<>(retGradient, epsilonNext);
    }

    @Override
    public FwdPassReturn activate(final Layer layer, final NeuralNetConfiguration conf,
                    final IActivation gateActivationFn, //Activation function for the gates - sigmoid or hard sigmoid (must be found in range 0 to 1)
                    INDArray input, final INDArray recurrentWeights, //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                    final INDArray inputWeights, //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                    final INDArray biases, //Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T
                    final boolean training, final INDArray prevOutputActivations, final INDArray prevMemCellState,
                    boolean forBackprop, boolean forwards, final String inputWeightKey, INDArray maskArray, //Input mask: should only be used with bidirectional RNNs + variable length
                    final boolean hasPeepholeConnections,   //True for GravesLSTM, false for LSTM
                    final LayerWorkspaceMgr workspaceMgr) {

        boolean is2dInput = input.rank() < 3; //Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
        val timeSeriesLength = (is2dInput ? 1 : input.size(2));
        val hiddenLayerSize = recurrentWeights.size(0);
        val miniBatchSize = input.size(0);
        val inputLayerSize = input.size(1);

        INDArray x = toCOrder(input.permute(2, 0, 1));
        INDArray linInputWeights = inputWeights;
        INDArray linRecurrentWeights = recurrentWeights;
        INDArray linBiases = biases;

        INDArray prevAct = toCOrder(prevOutputActivations);
        INDArray prevMemCell = toCOrder(prevMemCellState);

        INDArray outputActivations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS,
                        new long[] {timeSeriesLength, miniBatchSize, hiddenLayerSize * (BIDIRECTIONAL ? 2 : 1)}, 'c');
        INDArray finalMemCellState = Nd4j.createUninitialized(
                        new long[] {/*numLayers * (bidirectional ? 2 : 1),*/ miniBatchSize, hiddenLayerSize}, 'c');
        INDArray finalStepActivations = Nd4j.createUninitialized(
                        new long[] {/*numLayers * (bidirectional ? 2 : 1),*/ miniBatchSize, hiddenLayerSize}, 'c');

        FwdPassReturn toReturn = new FwdPassReturn();
        toReturn.prevAct = prevAct;
        toReturn.prevMemCell = prevMemCell;

        Nd4j.getExecutioner().commit();



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
            int[] dimA = {(int) miniBatchSize, (int) inputLayerSize, 1};
            int[] strideA = {(int) dimA[2] * dimA[1], dimA[2], 1};

            checkCudnn(cudnnSetTensorNdDescriptor(xDesc.get(cudnnTensorStruct.class, i), dataType, 3, dimA, strideA));
            checkCudnn(cudnnSetTensorNdDescriptor(dxDesc.get(cudnnTensorStruct.class, i), dataType, 3, dimA, strideA));

            int[] dimB = {(int) miniBatchSize, (int) hiddenLayerSize * (BIDIRECTIONAL ? 2 : 1), 1};
            int[] strideB = {dimB[2] * dimB[1], dimB[2], 1};

            checkCudnn(cudnnSetTensorNdDescriptor(yDesc.get(cudnnTensorStruct.class, i), dataType, 3, dimB, strideB));
            checkCudnn(cudnnSetTensorNdDescriptor(dyDesc.get(cudnnTensorStruct.class, i), dataType, 3, dimB, strideB));
        }

        int[] dimC = {NUM_LAYERS * (BIDIRECTIONAL ? 2 : 1), (int) miniBatchSize, (int) hiddenLayerSize};
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

        checkCudnn(cudnnSetDropoutDescriptor(cudnnContext.dropoutDesc, cudnnContext, DROPOUT, stateSpace, stateSize,
                        Nd4j.getRandom().getSeed()));

        checkCudnn(cudnnSetRNNDescriptor_v6(cudnnContext, cudnnContext.rnnDesc, (int) hiddenLayerSize, NUM_LAYERS, cudnnContext.dropoutDesc,
                         CUDNN_LINEAR_INPUT, BIDIRECTIONAL ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, RNN_MODE,
                        CUDNN_RNN_ALGO_STANDARD, dataType));

        cudnnTensorStruct xDesc0 = xDesc.get(cudnnTensorStruct.class, 0);
        checkCudnn(cudnnGetRNNParamsSize(cudnnContext, cudnnContext.rnnDesc, xDesc0, sizeInBytes, dataType));
        long weightsSize = sizeInBytes.get(0);
        if (weightsSize > weightsSpace.capacity()) {
            weightsSpace.deallocate();
            weightsSpace = new DataCache(weightsSize);
        }
        weightsSpace.limit(weightsSize);

        int[] dimW = {(int) weightsSize / dataTypeSize, 1, 1};

        checkCudnn(cudnnSetFilterNdDescriptor(cudnnContext.wDesc, dataType, CUDNN_TENSOR_NCHW, 3, dimW));
        checkCudnn(cudnnSetFilterNdDescriptor(cudnnContext.dwDesc, dataType, CUDNN_TENSOR_NCHW, 3, dimW));

        checkCudnn(cudnnGetRNNWorkspaceSize(cudnnContext, cudnnContext.rnnDesc, (int) timeSeriesLength, xDesc, sizeInBytes));
        long workSize = sizeInBytes.get(0);
        DataCache workSpace = workspaceMgr.getHelperWorkspace(LayerWorkspaceMgr.CUDNN_WORKSPACE_KEY);
        if (workSpace == null || workSize > workSpace.capacity()) {
            if(log.isTraceEnabled()){
                if(workSpace == null){
                    log.trace("CudnnLSTMHelper activate: Allocating initial workspace of size {} ({})", workSize,
                            StringUtils.TraditionalBinaryPrefix.long2String(workSize, "B", 2));
                } else {
                    log.trace("CudnnLSTMHelper activate: Deallocating workspace of size {} ({}), allocating new workspace of size {} ({})",
                            workSpace.capacity(), StringUtils.TraditionalBinaryPrefix.long2String(workSpace.capacity(), "B", 2),
                            workSize, StringUtils.TraditionalBinaryPrefix.long2String(workSize, "B", 2));
                }
            }
            if(workSpace != null)
                workSpace.deallocate();
            workSpace = new DataCache(workSize);
            workspaceMgr.setHelperWorkspace(LayerWorkspaceMgr.CUDNN_WORKSPACE_KEY, workSpace);
        }
        workSpace.limit(workSize);

        checkCudnn(cudnnGetRNNTrainingReserveSize(cudnnContext, cudnnContext.rnnDesc, (int) timeSeriesLength, xDesc,
                        sizeInBytes));
        long reserveSize = sizeInBytes.get(0);
        if (reserveSize > reserveSpace.capacity()) {
            reserveSpace.deallocate();
            reserveSpace = new DataCache(reserveSize);
        }
        reserveSpace.limit(reserveSize);

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareActionAllWrite(x, linInputWeights,
                        linRecurrentWeights, linBiases, prevAct, prevMemCell, outputActivations, finalMemCellState,
                        finalStepActivations);
        Pointer xData = allocator.getPointer(x, context);
        Pointer linInputWeightsData = allocator.getPointer(linInputWeights, context);
        Pointer linRecurrentWeightsData = allocator.getPointer(linRecurrentWeights, context);
        Pointer linBiasesData = allocator.getPointer(linBiases, context);
        Pointer prevActData = allocator.getPointer(prevAct, context);
        Pointer prevMemCellData = allocator.getPointer(prevMemCell, context);
        Pointer outputActivationsData = allocator.getPointer(outputActivations, context);
        Pointer finalMemCellStateData = allocator.getPointer(finalMemCellState, context);
        Pointer finalTimeStepActivationsData = allocator.getPointer(finalStepActivations, context);

        CUstream_st stream = new CUstream_st(context.getOldStream());
        checkCudnn(cudnnSetStream(cudnnContext, stream));

        checkCuda(cudaMemsetAsync(weightsSpace, 0, weightsSpace.limit(), stream));

        int[] dataType = new int[1];
        int[] format = new int[1];
        int[] nbDims = new int[1];
        int[] filterDimA = new int[3];
        Pointer linLayerMat = new Pointer();
        Pointer linLayerBias = new Pointer();

        for (int layerNum = 0; layerNum < NUM_LAYERS * (BIDIRECTIONAL ? 2 : 1); layerNum++) {
            for (int linLayerID = 0; linLayerID < NUM_LINEAR_LAYERS; linLayerID++) {
                checkCudnn(cudnnGetRNNLinLayerMatrixParams(cudnnContext, cudnnContext.rnnDesc, layerNum, xDesc0,
                                cudnnContext.wDesc, weightsSpace, linLayerID, cudnnContext.linLayerMatDesc,
                                linLayerMat));

                checkCudnn(cudnnGetFilterNdDescriptor(cudnnContext.linLayerMatDesc, 3, dataType, format, nbDims,
                                filterDimA));

                checkCudnn(cudnnGetRNNLinLayerBiasParams(cudnnContext, cudnnContext.rnnDesc, layerNum, xDesc0,
                                cudnnContext.wDesc, weightsSpace, linLayerID, cudnnContext.linLayerBiasDesc,
                                linLayerBias));

                checkCudnn(cudnnGetFilterNdDescriptor(cudnnContext.linLayerBiasDesc, 3, dataType, format, nbDims,
                                filterDimA));

                // our data is in "new, forget, output, and input gates" order (aka IFOG), each kind of weight packed together
                int position = 0;
                long size = 0;
                Pointer data = null;
                switch (linLayerID) {
                    case 0:
                        data = linInputWeightsData;
                        position = 3;
                        size = inputLayerSize;
                        break; // input gate
                    case 1:
                        data = linInputWeightsData;
                        position = 1;
                        size = inputLayerSize;
                        break; // forget gate
                    case 2:
                        data = linInputWeightsData;
                        position = 0;
                        size = inputLayerSize;
                        break; // new gate
                    case 3:
                        data = linInputWeightsData;
                        position = 2;
                        size = inputLayerSize;
                        break; // output gate
                    case 4:
                        data = linRecurrentWeightsData;
                        position = 3;
                        size = hiddenLayerSize;
                        break; // input gate
                    case 5:
                        data = linRecurrentWeightsData;
                        position = 1;
                        size = hiddenLayerSize;
                        break; // forget gate
                    case 6:
                        data = linRecurrentWeightsData;
                        position = 0;
                        size = hiddenLayerSize;
                        break; // new gate
                    case 7:
                        data = linRecurrentWeightsData;
                        position = 2;
                        size = hiddenLayerSize;
                        break; // output gate
                    default:
                        throw new RuntimeException();
                }
                checkCuda(cudaMemcpyAsync(linLayerMat, data.position(position * size * hiddenLayerSize * dataTypeSize),
                                size * hiddenLayerSize * dataTypeSize, cudaMemcpyDeviceToDevice, stream));
                if (linLayerID < 4) {
                    checkCuda(cudaMemcpyAsync(linLayerBias,
                                    linBiasesData.position(position * hiddenLayerSize * dataTypeSize),
                                    hiddenLayerSize * dataTypeSize, cudaMemcpyDeviceToDevice, stream));
                }
            }
        }

        if (training) {
            checkCudnn(cudnnRNNForwardTraining(cudnnContext, cudnnContext.rnnDesc, (int) timeSeriesLength, xDesc, xData,
                            cudnnContext.hxDesc, prevActData, cudnnContext.cxDesc, prevMemCellData, cudnnContext.wDesc,
                            weightsSpace, yDesc, outputActivationsData, cudnnContext.hyDesc,
                            finalTimeStepActivationsData, cudnnContext.cyDesc, finalMemCellStateData, workSpace,
                            workSpace.limit(), reserveSpace, reserveSpace.limit()));
        } else {
            checkCudnn(cudnnRNNForwardInference(cudnnContext, cudnnContext.rnnDesc, (int) timeSeriesLength, xDesc, xData,
                            cudnnContext.hxDesc, prevActData, cudnnContext.cxDesc, prevMemCellData, cudnnContext.wDesc,
                            weightsSpace, yDesc, outputActivationsData, cudnnContext.hyDesc,
                            finalTimeStepActivationsData, cudnnContext.cyDesc, finalMemCellStateData, workSpace,
                            workSpace.limit()));
        }

        allocator.getFlowController().registerActionAllWrite(context, x, linInputWeights, linRecurrentWeights,
                        linBiases, prevAct, prevMemCell, outputActivations, finalMemCellState, finalStepActivations);

        toReturn.fwdPassOutput = outputActivations.permute(1, 2, 0);
        toReturn.lastAct = finalStepActivations;
        toReturn.lastMemCell = finalMemCellState;
        toReturn.prevAct = prevAct;
        toReturn.prevMemCell = prevMemCell;

        return toReturn;
    }
}
