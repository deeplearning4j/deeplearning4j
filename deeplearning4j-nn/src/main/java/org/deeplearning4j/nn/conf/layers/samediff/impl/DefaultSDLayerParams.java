package org.deeplearning4j.nn.conf.layers.samediff.impl;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.*;

@JsonIgnoreProperties({"paramsList", "weightParamsList", "biasParamsList"})
@EqualsAndHashCode(exclude = {"paramsList", "weightParamsList", "biasParamsList"})
public class DefaultSDLayerParams implements SDLayerParams {

    private Map<String,int[]> weightParams = new LinkedHashMap<>();
    private Map<String,int[]> biasParams = new LinkedHashMap<>();

    private List<String> paramsList;
    private List<String> weightParamsList;
    private List<String> biasParamsList;

    @Override
    public void addWeightParam(String paramKey, int[] paramShape) {
        weightParams.put(paramKey, paramShape);
        paramsList = null;
        weightParams = null;
        biasParams = null;
    }

    @Override
    public void addBiasParam(String paramKey, int[] paramShape) {
        biasParams.put(paramKey, paramShape);
        paramsList = null;
        weightParams = null;
        biasParams = null;
    }

    @Override
    public List<String> getParameterKeys() {
        if(paramsList == null) {
            List<String> out = new ArrayList<>();
            out.addAll(getWeightParameterKeys());
            out.addAll(getBiasParameterKeys());
            this.paramsList = Collections.unmodifiableList(out);
        }
        return paramsList;
    }

    @Override
    public List<String> getWeightParameterKeys() {
        if(weightParamsList == null){
            weightParamsList = Collections.unmodifiableList(new ArrayList<>(weightParams.keySet()));
        }
        return weightParamsList;
    }

    @Override
    public List<String> getBiasParameterKeys() {
        if(biasParamsList == null){
            biasParamsList = Collections.unmodifiableList(new ArrayList<>(biasParams.keySet()));
        }
        return biasParamsList;
    }

    @Override
    public Map<String, int[]> getParamShapes() {
        Map<String, int[]> map = new LinkedHashMap<>();
        map.putAll(weightParams);
        map.putAll(biasParams);
        return map;
    }

    @Override
    public void clear() {
        weightParams.clear();
        biasParams.clear();
        paramsList = null;
        weightParamsList = null;
        biasParamsList = null;
    }
}
