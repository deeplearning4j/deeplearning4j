package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import lombok.NonNull;

/**
 *
 * Masks timesteps with 0 activation. Assumes that the input shape is [batch_size, input_size, timesteps].
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
    public void migrateInput() {
        underlying.migrateInput();
    }


    @Override
    public Type type() {
        return Type.RECURRENT;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        return underlying.backpropGradient(epsilon);
    }


    @Override
    public INDArray preOutput(INDArray x) {
        return underlying.preOutput(x);
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        return underlying.preOutput(x, training);
    }

    @Override
    public INDArray activate(TrainingMode training) {
        INDArray input = input();
        INDArray mask = input.eq(0).sum(1).neq(input.shape()[1]);
        underlying.setMaskArray(mask);
        return underlying.activate(training);
    }

    @Override
    public INDArray activate(INDArray input, TrainingMode training) {
        INDArray mask = input.eq(0).sum(1).neq(input.shape()[1]);
        underlying.setMaskArray(mask);
        return underlying.activate(input, training);
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        return underlying.activate(x, training);
    }

    @Override
    public INDArray activate(boolean training) {
        INDArray input = input();
        INDArray mask = input.eq(0).sum(1).neq(input.shape()[1]);
        underlying.setMaskArray(mask);
        return underlying.activate(training);
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        INDArray mask = input.eq(0).sum(1).neq(input.shape()[1]);
        underlying.setMaskArray(mask);
        return underlying.activate(input, training);
    }

    @Override
    public INDArray activate() {
        INDArray input = input();
        INDArray mask = input.eq(0).sum(1).neq(input.shape()[1]);
        underlying.setMaskArray(mask);
        return underlying.activate();
    }

    @Override
    public INDArray activate(INDArray input) {
        INDArray mask = input.eq(0).sum(1).neq(input.shape()[1]);
        underlying.setMaskArray(mask);
        return underlying.activate(input);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        underlying.feedForwardMaskArray(maskArray, currentMaskState, minibatchSize);

        //Input: 2d mask array, for masking a time series. After extracting out the last time step, we no longer need the mask array
        return new Pair<>(null, currentMaskState);
    }


}
