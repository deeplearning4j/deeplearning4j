package org.deeplearning4j.nn.conf.layers.samediff;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * A base layer used for implementing Deeplearning4j Output layers using SameDiff. These layers are scoring/output layers:
 * they should only be used as the final layer a network. For general/intermediate <br>
 * To implement a Deeplearinng layer using SameDiff, extend this class.<br>
 * There are 5 required methods:<br>
 * - defineLayer: Defines the forward pass for the layer<br>
 * - defineParameters: Define the layer's parameters in a way suitable for DL4J<br>
 * - initializeParameters: if required, set the initial parameter values for the layer<br>
 * - getOutputType: determine the type of output/activations for the layer (without actually executing the layer's
 * forward pass)<br>
 * - activationsVertexName(): see {@link #activationsVertexName()} for details<br>
 * <br>
 * Furthermore, there are 3 optional methods:<br>
 * - setNIn(InputType inputType, boolean override): if implemented, set the number of inputs to the layer during network
 *   initialization<br>
 * - getPreProcessorForInputType: return the preprocessor that should be added (if any), for the given input type<br>
 * - applyGlobalConfigToLayer: apply any global configuration options (weight init, activation functions etc) to the
 *   layer's configuration.<br>
 * - labelsRequired: see {@link #labelsRequired()} for details<br>
 *
 * @author Alex Black
 */
public abstract class SameDiffOutputLayer extends AbstractSameDiffLayer {


    protected SameDiffOutputLayer(){
        //No op constructor for Jackson
    }

    /**
     * Define the output layer
     * @param sameDiff   SameDiff instance
     * @param layerInput Input to the layer
     * @param labels     Labels variable (or null if {@link #labelsRequired()} returns false
     * @param paramTable Parameter table - keys as defined by {@link #defineParameters(SDLayerParams)}
     * @return The final layer variable corresponding to the score/loss during forward pass. This must be a single scalar value.
     */
    public abstract SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, SDVariable labels,
                                           Map<String,SDVariable> paramTable);

    /**
     * Output layers should terminate in a single scalar value (i.e., a score) - however, sometimes the output activations
     * (such as softmax probabilities) need to be returned. When this is the case, we need to know the name of the
     * SDVariable that corresponds to these.<br>
     * If the final network activations are just the input to the layer, simply return "input"
     *
     * @return The name of the activations to return when performing forward pass
     */
    public abstract String activationsVertexName();

    /**
     * Whether labels are required for calculating the score. Defaults to true - however, if the score
     * can be calculated without labels (for example, in some output layers used for unsupervised learning)
     * this can be set to false.
     * @return True if labels are required to calculate the score/output, false otherwise.
     */
    public boolean labelsRequired(){
        return true;
    }

    //==================================================================================================================

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.samediff.SameDiffOutputLayer ret = new org.deeplearning4j.nn.layers.samediff.SameDiffOutputLayer(conf);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

}
