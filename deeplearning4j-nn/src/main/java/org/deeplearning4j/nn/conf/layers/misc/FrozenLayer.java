package org.deeplearning4j.nn.conf.layers.misc;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.Collection;

/**
 * Created by Alex on 10/07/2017.
 */
public class FrozenLayer extends Layer {


    @Override
    public Layer clone() {
        return null;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
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
    public double getLearningRateByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public Updater getUpdaterByParam(String paramName) {
        return null;
    }

    @Override
    public IUpdater getIUpdaterByParam(String paramName) {
        return null;
    }
}
