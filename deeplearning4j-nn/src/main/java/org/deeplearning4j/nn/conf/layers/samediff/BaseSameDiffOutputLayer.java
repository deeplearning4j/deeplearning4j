package org.deeplearning4j.nn.conf.layers.samediff;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.samediff.SameDiffOutputLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;
import java.util.List;
import java.util.Map;

public abstract class BaseSameDiffOutputLayer extends AbstractSameDiffLayer {

    protected BaseSameDiffOutputLayer(Builder builder){
        super(builder);
    }

    protected BaseSameDiffOutputLayer(){
        //No arg for JSON/Jackson
    }

    public abstract String outputActivationsKey();

    /**
     * Two keys:
     * First - For the score *per example* (1 value per example)
     * Second - for the average score (1 values for all examples)
     * @return
     */
    public abstract Pair<String,String> lossKeys();

    public abstract int[] labelShape();


    public abstract List<String> defineLayer(SameDiff sameDiff, SDVariable layerInput, SDVariable layerLabel, Map<String,SDVariable> paramTable);

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        SameDiffOutputLayer ret = new SameDiffOutputLayer(conf);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    public static abstract class Builder<T extends Builder<T>> extends AbstractSameDiffLayer.Builder<T> {




    }

}
