package org.deeplearning4j.nn.conf.layers.samediff;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public abstract class SameDiffOutputLayer extends AbstractSameDiffLayer {


    protected SameDiffOutputLayer(){
        //No op constructor for Jackson
    }

    public abstract SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, SDVariable labels,
                                           Map<String,SDVariable> paramTable);

    /**
     * Output layers should terminate in a single scalar value (i.e., a score) - however,
     *
     * @return
     */
    public abstract String activationsVertexName();

    /**
     * Whether labels are required for calculating the score. Defaults to true - however, if the score
     * can be calculated without labels (for example, in some output layers used for unsupervised learning)
     * this can be set to false.
     * @return
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
