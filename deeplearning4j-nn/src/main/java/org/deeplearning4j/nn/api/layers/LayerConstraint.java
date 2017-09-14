package org.deeplearning4j.nn.api.layers;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.Set;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface LayerConstraint extends Cloneable, Serializable {

    /**
     * Apply a given constraint to a layer at each iteration
     * in the provided epoch, after parameters have been updated.
     *
     * @param layer org.deeplearning4j.nn.api.Layer
     * @param iteration given iteration as integer
     * @param epoch current epoch as integer
     */
    void applyConstraint(Layer layer, int iteration, int epoch);

    /**
     * Set the parameters that this layer constraint should be applied to
     *
     * @param params Parameters that the layer constraint should be applied to
     */
    void setParams(Set<String> params);

    /**
     * @return Set of parameters that this layer constraint will be applied to
     */
    Set<String> getParams();

    LayerConstraint clone();

}
