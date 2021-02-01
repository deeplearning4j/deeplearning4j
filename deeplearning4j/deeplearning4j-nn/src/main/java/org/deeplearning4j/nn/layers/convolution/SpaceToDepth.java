/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.layers.convolution;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;


/**
 * Space to channels utility layer for convolutional input types.
 * <p>
 * This operation takes 4D array in, in either NCHW or NHWC format, and moves data from spatial dimensions (HW)
 * to channels (C) for given blockSize
 * <p></p>
 * Example:
 * blockSize = 4
 * dataFormat = "NCHW"
 * input shape =  [128, 16, 16, 3]
 * output shape = [128, 16/4, 16/4, 3*4*4]
 *
 *
 *
 * @author Max Pumperla
 */
@Slf4j
public class SpaceToDepth extends AbstractLayer<org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer> {

    public SpaceToDepth(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    private int getBlockSize() {
        return layerConf().getBlockSize();
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        INDArray input = this.input.castTo(epsilon.dataType());

        boolean nchw = layerConf().getDataFormat() == CNN2DFormat.NCHW;
        long miniBatch = input.size(0);
        long inDepth = input.size(nchw ? 1 : 3);
        long inH = input.size(nchw ? 2 : 1);
        long inW = input.size(nchw ? 3 : 2);

        long[] epsShape = nchw ?  new long[]{miniBatch, inDepth, inH, inW} : new long[]{miniBatch, inH, inW, inDepth};
        INDArray outEpsilon = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, input.dataType(), epsShape, 'c');

        Gradient gradient = new DefaultGradient();

        int blockSize = getBlockSize();

        //Workaround for issue: https://github.com/eclipse/deeplearning4j/issues/8859
        if(!Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('c');

        CustomOp op = DynamicCustomOp.builder("depth_to_space")
                .addInputs(epsilon)
                .addIntegerArguments(blockSize, nchw ? 0 : 1)       //nchw = 0, nhwc = 1
                .addOutputs(outEpsilon)
                .build();
        Nd4j.getExecutioner().exec(op);

        return new Pair<>(gradient, outEpsilon);
    }

    protected INDArray preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, workspaceMgr);

        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to space to channels with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape " + layerConf().getDataFormat().dimensionNames() + ". "
                    + layerId());
        }

        if (preOutput != null && forBackprop) {
            return preOutput;
        }

        boolean nchw = layerConf().getDataFormat() == CNN2DFormat.NCHW;

        long miniBatch = input.size(0);
        long depth = input.size(nchw ? 1 : 3);
        long inH = input.size(nchw ? 2 : 1);
        long inW = input.size(nchw ? 3 : 2);

        int blockSize = getBlockSize();

        long outH = inH / blockSize;
        long outW = inW / blockSize;
        long outDepth = depth * blockSize * blockSize;

        long[] outShape = nchw ? new long[]{miniBatch, outDepth, outH, outW} : new long[]{miniBatch, outH, outW,  outDepth};
        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, input.dataType(), outShape, 'c');

        //Workaround for issue: https://github.com/eclipse/deeplearning4j/issues/8859
        INDArray input = this.input;
        if(!Shape.hasDefaultStridesForShape(input))
            input = input.dup('c');

        CustomOp op = DynamicCustomOp.builder("space_to_depth")
                .addInputs(input)
                .addIntegerArguments(blockSize, nchw ? 0 : 1)       //nchw = 0, nhwc = 1
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(op);

        return out;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return preOutput(training, false, workspaceMgr);
    }


    @Override
    public double calcRegularizationScore(boolean backpropParamsOnly){
        return 0;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException("Not supported - no parameters");
    }

    @Override
    public long numParams() {
        return 0;
    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void update(INDArray gradient, String paramType) {

    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return params();
    }

    @Override
    public void setParams(INDArray params) {
    }

}