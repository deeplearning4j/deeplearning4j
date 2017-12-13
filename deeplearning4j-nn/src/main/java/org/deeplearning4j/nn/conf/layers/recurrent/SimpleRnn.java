package org.deeplearning4j.nn.conf.layers.recurrent;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.SimpleRnnParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public class SimpleRnn extends BaseRecurrentLayer {

    protected SimpleRnn(Builder builder){
        super(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("SimpleRnn", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.recurrent.SimpleRnn ret =
                new org.deeplearning4j.nn.layers.recurrent.SimpleRnn(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return SimpleRnnParamInitializer.getInstance();
    }

    @Override
    public double getL1ByParam(String paramName){
        switch (paramName){
            case SimpleRnnParamInitializer.WEIGHT_KEY:
            case SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY:
                return l1;
            case SimpleRnnParamInitializer.BIAS_KEY:
                return l1Bias;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName){
        switch (paramName){
            case SimpleRnnParamInitializer.WEIGHT_KEY:
            case SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY:
                return l2;
            case SimpleRnnParamInitializer.BIAS_KEY:
                return l2Bias;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }

    public static class Builder extends BaseRecurrentLayer.Builder<Builder>{


        @Override
        public SimpleRnn build() {
            return new SimpleRnn(this);
        }
    }
}
