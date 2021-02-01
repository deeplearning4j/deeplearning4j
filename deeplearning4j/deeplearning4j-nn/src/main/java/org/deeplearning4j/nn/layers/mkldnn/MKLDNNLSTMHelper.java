/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.layers.mkldnn;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.FwdPassReturn;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelper;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class MKLDNNLSTMHelper implements LSTMHelper {
    @Override
    public boolean checkSupported(IActivation gateActivationFn, IActivation activationFn, boolean hasPeepholeConnections) {
        //TODO check other activation functions for MKLDNN
        return gateActivationFn instanceof ActivationSigmoid && activationFn instanceof ActivationTanH && BaseMKLDNNHelper.mklDnnEnabled();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(NeuralNetConfiguration conf, IActivation gateActivationFn, INDArray input,
                                                     INDArray recurrentWeights, INDArray inputWeights, INDArray epsilon, boolean truncatedBPTT,
                                                     int tbpttBackwardLength, FwdPassReturn fwdPass, boolean forwards, String inputWeightKey,
                                                     String recurrentWeightKey, String biasWeightKey, Map<String, INDArray> gradientViews,
                                                     INDArray maskArray, boolean hasPeepholeConnections, LayerWorkspaceMgr workspaceMgr) {
        //Not yet implemented/supported
        return null;
    }

    @Override
    public FwdPassReturn activate(Layer layer, NeuralNetConfiguration conf, IActivation gateActivationFn, INDArray input,
                                  INDArray recurrentWeights, INDArray inputWeights, INDArray biases, boolean training,
                                  INDArray prevOutputActivations, INDArray prevMemCellState, boolean forBackprop, boolean forwards,
                                  String inputWeightKey, INDArray maskArray, boolean hasPeepholeConnections, LayerWorkspaceMgr workspaceMgr) {

        /*
        DL4J data format: [bS, nIn, sL] - dataFormat == 2, directionMode == 0 (forward)
        Inputs:
        x = [bS, nIn, sL]
        Wx = [nIn, 4*nOut]
        Wr = [nOut, 4*nOut]
        Wp = [3*nOut]               Optional peephole weights
        b = [4*nOut]
        seqLen = [bS]
        initialOut = [bs, nOut]
        initialCell = [bs, nOut]

        Outputs:
        out = [bS, nOut, sL]
        outLast = [bs, nOut]
        cellLast = [bs,nOut]

        Gates order: input, forget, input modulation, output


        const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
        const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
        const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
        const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
        const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
        const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
        const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
        const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
         */

        INDArray b1d = biases.reshape(biases.length());
        INDArray seqLen = null;
        if(maskArray != null){
            seqLen = BooleanIndexing.firstIndex(maskArray, Conditions.equals(0), 1);    //First 0 along dimension 1 (for [mb, seqLen])
        }

        List<INDArray> args = new ArrayList<>();
        args.add(input);
        args.add(inputWeights);
        args.add(recurrentWeights);
        if(hasPeepholeConnections){
            throw new IllegalStateException("Not yet implemented");
        }
        args.add(b1d);
        if(seqLen != null)
            args.add(seqLen);
        if(prevOutputActivations != null)
            args.add(prevOutputActivations);
        if(prevMemCellState != null)
            args.add(prevMemCellState);

        IActivation a = ((LSTM)conf.getLayer()).getActivationFn();

        DynamicCustomOp op = DynamicCustomOp.builder("lstmLayer")
                .addInputs(args.toArray(new INDArray[0]))
                .addBooleanArguments(
                        true,                               //hasBiases
                        seqLen != null,                     //hasSeqLen
                        prevOutputActivations != null,      //hasInitH
                        prevMemCellState != null,           //hasInitC
                        hasPeepholeConnections,             //hasPh
                        true,                               //retFullSeq
                        true,                               //retLastH
                        true                                //retLastC
                )
                .addIntegerArguments(
                        2,                                  //data format: 2 = [bS, nIn, sL]
                        0,                                  //direction: 0 = forward
                        activationToArg(gateActivationFn),  //Gate activation
                        activationToArg(a),                 //Cell state activation
                        activationToArg(a)                  //Output activation (same as cell in DL4J)
                )
                .build();

        List<LongShapeDescriptor> outShapes = op.calculateOutputShape();

        for(LongShapeDescriptor lsd : outShapes){
            INDArray arr = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, lsd.dataType(), lsd.getShape(), lsd.getOrder());
            op.addOutputArgument(arr);
        }

        FwdPassReturn f = new FwdPassReturn();
        f.fwdPassOutput = op.getOutputArgument(0);
        f.lastAct = op.getOutputArgument(1);
        f.lastMemCell = op.getOutputArgument(2);

        return f;
    }

    @Override
    public Map<String, Long> helperMemoryUse() {
        return Collections.emptyMap();
    }

    private int activationToArg(IActivation a){
        //0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
        if(a instanceof ActivationTanH)
            return 0;
        if(a instanceof ActivationReLU)
            return 1;
        if(a instanceof ActivationSigmoid)
            return 2;
        if(a instanceof ActivationIdentity)
            return 3;
        if(a instanceof ActivationLReLU)
            return 4;
        if(a instanceof ActivationThresholdedReLU)
            return 5;
        if(a instanceof ActivationHardSigmoid)
            return 7;
        if(a instanceof ActivationELU)
            return 8;
        if(a instanceof ActivationSoftSign)
            return 9;
        if(a instanceof ActivationSoftPlus)
            return 10;
        throw new IllegalStateException("Unknown or not supported activation function: " + a);
    }
}
