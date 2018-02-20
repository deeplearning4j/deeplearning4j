package org.deeplearning4j.nn.conf.layers.samediff;

import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.util.List;
import java.util.Map;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface SDLayerParams {

    void addWeightParam(String paramKey, int[] paramShape);

    void addBiasParam(String paramKey, int[] paramShape);

    List<String> getParameterKeys();

    List<String> getWeightParameterKeys();

    List<String> getBiasParameterKeys();

    Map<String,int[]> getParamShapes();

    void clear();
}
