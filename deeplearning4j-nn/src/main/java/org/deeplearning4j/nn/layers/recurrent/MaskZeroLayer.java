package org.deeplearning4j.nn.layers.recurrent;

import java.util.Arrays;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import lombok.NonNull;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**
 *
 * Masks timesteps with 0 activation. Assumes that the input shape is [batch_size, input_size, timesteps].
   @author Martin Boyanov mboyanov@gmail.com
 */
public class MaskZeroLayer extends BaseWrapperLayer {

    /**
     *
     */
    private static final long serialVersionUID = -7369482676002469854L;

    public MaskZeroLayer(@NonNull Layer underlying){
        super(underlying);
    }


    @Override
    public Type type() {
        return Type.RECURRENT;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        return underlying.backpropGradient(epsilon, workspaceMgr);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray input = input();
        setMaskFromInput(input);
        return underlying.activate(training, workspaceMgr);
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
        setMaskFromInput(input);
        return underlying.activate(input, training, workspaceMgr);
    }

    private void setMaskFromInput(INDArray input) {
        if (input.rank() != 3) {
            throw new IllegalArgumentException("Expected input of shape [batch_size, timestep_input_size, timestep], got shape "+Arrays.toString(input.shape()) + " instead");
        }
        INDArray mask = input.eq(0).sum(1).neq(input.shape()[1]);
        underlying.setMaskArray(mask);
    }

    @Override
    public int numParams() {
        return underlying.numParams();
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        underlying.feedForwardMaskArray(maskArray, currentMaskState, minibatchSize);

        //Input: 2d mask array, for masking a time series. After extracting out the last time step, we no longer need the mask array
        return new Pair<>(null, currentMaskState);
    }


}
