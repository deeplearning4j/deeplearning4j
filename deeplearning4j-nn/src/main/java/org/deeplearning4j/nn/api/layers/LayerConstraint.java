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
     * @param biasConstraint if constraint is applied to bias parameters
     * @param weightConstraint if contraint is applied to weight parameters
     * @param paramNames Parameter names to which to apply the constraints
     */
    void applyConstraint(Layer layer, int iteration, int epoch, Boolean biasConstraint,
                         Boolean weightConstraint, Set<String> paramNames);

    LayerConstraint clone();

}
