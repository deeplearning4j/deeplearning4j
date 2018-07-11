package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * A base layer used for implementing Deeplearning4j layers using SameDiff. These layers are not scoring/output layers:
 * that is, they should be used as the intermediate layer in a network only.<br>
 * To implement an output layer, extend {@link SameDiffOutputLayer} instead.<br>
 * Note also that if multiple inputs are required, it is possible to implement a vertex instead: {@link SameDiffVertex}<br>
 * <br>
 * To implement a Deeplearning layer using SameDiff, extend this class.<br>
 * There are 4 required methods:<br>
 * - defineLayer: Defines the forward pass for the layer<br>
 * - defineParameters: Define the layer's parameters in a way suitable for DL4J<br>
 * - initializeParameters: if required, set the initial parameter values for the layer<br>
 * - getOutputType: determine the type of output/activations for the layer (without actually executing the layer's
 * forward pass)<br>
 * <br>
 * Furthermore, there are 3 optional methods:<br>
 * - setNIn(InputType inputType, boolean override): if implemented, set the number of inputs to the layer during network
 *   initialization<br>
 * - getPreProcessorForInputType: return the preprocessor that should be added (if any), for the given input type<br>
 * - applyGlobalConfigToLayer: apply any global configuration options (weight init, activation functions etc) to the
 *   layer's configuration.<br>
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public abstract class SameDiffLayer extends AbstractSameDiffLayer {

    protected WeightInit weightInit;

    protected SameDiffLayer(Builder builder){
        super(builder);
        this.weightInit = builder.weightInit;
    }

    protected SameDiffLayer(){
        //No op constructor for Jackson
    }

    /**
     * Define the layer
     * @param sameDiff   SameDiff instance
     * @param layerInput Input to the layer
     * @param paramTable Parameter table - keys as defined by {@link #defineParameters(SDLayerParams)}
     * @return The final layer variable corresponding to the activations/output from the forward pass
     */
    public abstract SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String,SDVariable> paramTable);

    //==================================================================================================================

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.samediff.SameDiffLayer ret = new org.deeplearning4j.nn.layers.samediff.SameDiffLayer(conf);
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

        /**
         * @param weightInit Weight initialization to use for the layer
         */
        public T weightInit(WeightInit weightInit){
            this.weightInit = weightInit;
            return (T)this;
        }
    }
}
