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

package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;

/**
 * TimeDistributed wrapper layer.<br>
 * Note: only the "Feed forward layer time distributed in an RNN" is currently supported.
 * For example, a time distributed dense layer.<br>
 * Usage: {@code .layer(new TimeDistributed(new DenseLayer.Builder()....build(), timeAxis))}<br>
 * Note that for DL4J RNNs, time axis is always 2 - i.e., RNN activations have shape [minibatch, size, sequenceLength]
 *
 * @author Alex Black
 */
public class TimeDistributedLayer extends BaseWrapperLayer {

    private RNNFormat rnnDataFormat;

    public TimeDistributedLayer(Layer underlying, RNNFormat rnnDataFormat) {
        super(underlying);
        this.rnnDataFormat = rnnDataFormat;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        INDArray reshapedEps = reshape(epsilon);
        Pair<Gradient, INDArray> p = underlying.backpropGradient(reshapedEps, workspaceMgr);
        INDArray reverted = revertReshape(p.getSecond(), epsilon.size(0));
        reverted = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, reverted);
        p.setSecond(reverted);
        return p;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return activate(input(), training, workspaceMgr);
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray reshaped = reshape(input);
        INDArray out = underlying.activate(reshaped, training, workspaceMgr);
        INDArray ret = revertReshape(out, input.size(0));
        return workspaceMgr.dup(ArrayType.ACTIVATIONS, ret);
    }

    protected INDArray reshape(INDArray array){
        //Reshape the time axis to the minibatch axis
        //For example, for RNN -> FF (dense time distributed): [mb, size, seqLen] -> [mb x seqLen, size]
        int axis = (rnnDataFormat == RNNFormat.NCW) ? 2 : 1;
        if(axis < 0)
            axis += array.rank();

        int[] permuteAxis = permuteAxes(array.rank(), axis);
        INDArray permute = array.permute(permuteAxis);

        long[] newShape = new long[array.rank()-1];
        newShape[0] = array.size(0) * array.size(axis);
        int j=1;
        for( int i=1; i<array.rank(); i++ ){
            if(axis == i)
                continue;
            newShape[j++] = array.size(i);
        }

        INDArray reshape = permute.dup().reshape('c', newShape);
        return reshape;
    }

    protected int[] permuteAxes(int rank, int timeAxis){
        int[] permuteAxis = new int[rank];
        permuteAxis[0] = 0;
        permuteAxis[1] = timeAxis;
        int j=2;
        for( int i=1; i<rank; i++ ){
            if(timeAxis == i)
                continue;
            permuteAxis[j++] = i;
        }
        return permuteAxis;
    }

    protected INDArray revertReshape(INDArray toRevert, long minibatch){

        int axis = (rnnDataFormat == RNNFormat.NCW)? 2 : 1;
        if(axis < 0)
            axis += (toRevert.rank()+1);

        long[] newShape = new long[toRevert.rank()+1];
        newShape[0] = minibatch;
        newShape[1] = toRevert.size(0)/minibatch;
        for( int i=1; i<toRevert.rank(); i++ ){
            newShape[i+1] = toRevert.size(i);
        }

        INDArray reshaped = toRevert.reshape('c', newShape);

        int[] permute = ArrayUtil.invertPermutation(permuteAxes(toRevert.rank() + 1, axis));

        INDArray permuted = reshaped.permute(permute);
        return permuted;
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        if(maskArray == null){
            underlying.setMaskArray(null);
        } else {
            INDArray reshaped = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray, LayerWorkspaceMgr.noWorkspaces(), ArrayType.ACTIVATIONS);
            underlying.setMaskArray(reshaped);
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        if(maskArray == null){
            return underlying.feedForwardMaskArray(null, currentMaskState, minibatchSize);
        } else {
            INDArray reshaped = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray, LayerWorkspaceMgr.noWorkspaces(), ArrayType.ACTIVATIONS);
            Pair<INDArray, MaskState> p = underlying.feedForwardMaskArray(reshaped, currentMaskState, minibatchSize);
            if(p == null || p.getFirst() == null){
                return p;
            }
            INDArray reshaped2 = TimeSeriesUtils.reshapeVectorToTimeSeriesMask(p.getFirst(), (int)maskArray.size(0));
            p.setFirst(reshaped2);
            return p;
        }
    }
}
