package org.deeplearning4j.nn.conf.layers.recurrent;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * LastTimeStep is a "wrapper" layer: it wraps any RNN layer, and extracts out the last time step during forward pass,
 * and returns it as a row vector. That is, for 3d (time series) input, we take the last time step and
 *
 */
public class LastTimeStep extends Layer {

    private Layer underlying;

    public LastTimeStep(Layer underlying){
        this.underlying = underlying;
    }


    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        return null;
    }

    @Override
    public ParamInitializer initializer() {
        return null;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return null;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {

    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }
}
