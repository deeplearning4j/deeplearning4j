package org.deeplearning4j.nn.conf.layers.samediff;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.samediff.SameDiffLossLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public abstract class BaseSameDiffLossLayer extends NoParamSameDiffLayer {

    protected BaseSameDiffLossLayer(Builder builder){
        super(builder);
    }

    protected BaseSameDiffLossLayer(){
        //No arg for Jackson/JSON
    }

    public abstract void defineLayer(SameDiff sameDiff, SDVariable layerInput, SDVariable label);

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        SameDiffLossLayer ret = new SameDiffLossLayer(conf);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    public static abstract class Builder<T extends Builder<T>> extends NoParamSameDiffLayer.Builder<T> {


    }
}
