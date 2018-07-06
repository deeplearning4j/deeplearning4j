package org.deeplearning4j.nn.conf.layers.util;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.NoParamLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * MaskLayer applies the mask array to the forward pass activations, and backward pass gradients, passing through
 * this layer. It can be used with 2d (feed-forward), 3d (time series) or 4d (CNN) activations.
 *
 * @author Alex Black
 */
@NoArgsConstructor
public class MaskLayer extends NoParamLayer {
    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.util.MaskLayer ret = new org.deeplearning4j.nn.layers.util.MaskLayer(conf);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return inputType;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;    //No op
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;   //No params
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;   //No params
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return new LayerMemoryReport();
    }

    @Override
    public String toString(){
        return "MaskLayer()";
    }
}
