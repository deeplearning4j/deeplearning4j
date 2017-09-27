package org.deeplearning4j.nn.graph.multioutput.testlayers;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public class SplitOutputLayerConf extends OutputLayer {


    public SplitOutputLayerConf(Builder builder){
        super(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf,
                             Collection<IterationListener> iterationListeners,
                             String name, int layerIndex, int numInputs, INDArray layerParamsView,
                             boolean initializeParams) {
        LayerValidation.assertNInNOutSet("SplitOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        SplitOutputLayer ret = new SplitOutputLayer(conf);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public int numOutputs(){
        return 2;
    }

    @NoArgsConstructor
    public static class Builder extends OutputLayer.Builder {

        @Override
        @SuppressWarnings("unchecked")
        public SplitOutputLayerConf build(){
            return new SplitOutputLayerConf(this);
        }
    }


}
