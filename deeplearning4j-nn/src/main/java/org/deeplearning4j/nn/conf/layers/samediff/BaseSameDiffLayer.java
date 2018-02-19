package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.params.SameDiffParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

@Data
@EqualsAndHashCode(callSuper = true)
public abstract class BaseSameDiffLayer extends AbstractSameDiffLayer {

    protected WeightInit weightInit;

    protected BaseSameDiffLayer(Builder builder){
        super(builder);
        this.weightInit = builder.weightInit;
    }

    protected BaseSameDiffLayer(){
        //No op constructor for Jackson
    }

    public abstract List<String> defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String,SDVariable> paramTable);

    //==================================================================================================================

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        SameDiffLayer ret = new SameDiffLayer(conf);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @SuppressWarnings("unchecked")
    public static abstract class Builder<T extends Builder<T>> extends AbstractSameDiffLayer.Builder<T> {

        protected WeightInit weightInit = WeightInit.XAVIER;

        public T weightInit(WeightInit weightInit){
            this.weightInit = weightInit;
            return (T)this;
        }

    }
}
