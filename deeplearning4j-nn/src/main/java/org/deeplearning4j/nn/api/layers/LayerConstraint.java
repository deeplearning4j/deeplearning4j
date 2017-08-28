package org.deeplearning4j.nn.api.layers;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface LayerConstraint {

    void applyConstraint(Layer layer, int iteration, int epoch);

}
