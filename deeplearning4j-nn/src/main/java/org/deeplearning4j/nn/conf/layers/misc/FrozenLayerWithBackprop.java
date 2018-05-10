package org.deeplearning4j.nn.conf.layers.misc;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.FrozenLayerParamInitializer;
import org.deeplearning4j.nn.params.FrozenLayerWithBackpropParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collection;
import java.util.List;

/**
 * Frozen layer freezes parameters of the layer it wraps, but allows the backpropagation to continue.
 * 
 * Created by Ugljesa Jovanovic (jovanovic.ugljesa@gmail.com) on 06/05/2018.
 */
@Data
public class FrozenLayerWithBackprop extends BaseWrapperLayer {

    public FrozenLayerWithBackprop(@JsonProperty("layer") Layer layer) {
        super(layer);
    }

    public NeuralNetConfiguration getInnerConf(NeuralNetConfiguration conf) {
        NeuralNetConfiguration nnc = conf.clone();
        nnc.setLayer(underlying);
        return nnc;
    }

    @Override
    public Layer clone() {
        FrozenLayerWithBackprop l = (FrozenLayerWithBackprop) super.clone();
        l.underlying = underlying.clone();
        return l;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                    Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                    boolean initializeParams) {

        //Need to be able to instantiate a layer, from a config - for JSON -> net type situations
        org.deeplearning4j.nn.api.Layer underlying = getUnderlying().instantiate(getInnerConf(conf), trainingListeners,
                        layerIndex, layerParamsView, initializeParams);

        NeuralNetConfiguration nncUnderlying = underlying.conf();

        if (nncUnderlying.variables() != null) {
            List<String> vars = nncUnderlying.variables(true);
            nncUnderlying.clearVariables();
            conf.clearVariables();
            for (String s : vars) {
                conf.variables(false).add(s);
                conf.getL1ByParam().put(s, 0.0);
                conf.getL2ByParam().put(s, 0.0);

                nncUnderlying.variables(false).add(s);
                nncUnderlying.getL1ByParam().put(s, 0.0);
                nncUnderlying.getL2ByParam().put(s, 0.0);
            }
        }

        return new org.deeplearning4j.nn.layers.FrozenLayerWithBackprop(underlying);
    }

    @Override
    public ParamInitializer initializer() {
        return FrozenLayerWithBackpropParamInitializer.getInstance();
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
    public IUpdater getUpdaterByParam(String paramName) {
        return null;
    }

    @Override
    public void setLayerName(String layerName) {
        super.setLayerName(layerName);
        underlying.setLayerName(layerName);
    }

    @Override
    public void setConstraints(List<LayerConstraint> constraints){
        this.constraints = constraints;
        this.underlying.setConstraints(constraints);
    }
}
