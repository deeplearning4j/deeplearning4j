package org.deeplearning4j.nn.conf.weightnoise;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * IWeightNoise instances operate on an weight array(s), modifying values at training time or test
 * time, before they are used. Note that the weights are copied before being modified - the original parameters
 * are not changed. However, if the pameters are not changed, the original array is returned.
 *
 * This interface can be used to implement functionality like DropConnect, weight quantization and weight
 * noise.
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface IWeightNoise extends Serializable, Cloneable{

    /**
     * Get the parameter, after applying weight noise
     *
     * @param layer     Layer to get the parameter for
     * @param paramKey  Parameter key
     * @param iteration Iteration number
     * @param epoch     Epoch number
     * @param train     If true: training. False: at test time
     * @return          Parameter, after applying weight noise
     */
    INDArray getParameter(Layer layer, String paramKey, int iteration, int epoch, boolean train, LayerWorkspaceMgr workspaceMgr);

    IWeightNoise clone();

}
