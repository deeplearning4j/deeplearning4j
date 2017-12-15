package org.deeplearning4j.nn.conf.layers.recurrent;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.layers.recurrent.LastTimeStepLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * LastTimeStep is a "wrapper" layer: it wraps any RNN layer, and extracts out the last time step during forward pass,
 * and returns it as a row vector. That is, for 3d (time series) input, we take the last time step and
 *
 */
public class LastTimeStep extends BaseWrapperLayer {

    public LastTimeStep(Layer underlying){
        super(underlying);
    }


    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        Layer temp = conf.getLayer();
        conf.setLayer(((LastTimeStep)conf.getLayer()).getUnderlying());
        LastTimeStepLayer l = new LastTimeStepLayer(underlying.instantiate(conf, iterationListeners, layerIndex, layerParamsView, initializeParams));
        conf.setLayer(temp);
        return l;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if(inputType.getType() != InputType.Type.RNN){
            throw new IllegalArgumentException("Require RNN input type - got " + inputType);
        }
        InputType outType = underlying.getOutputType(layerIndex, inputType);
        InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent)outType;
        return InputType.feedForward(r.getSize());
    }
}
