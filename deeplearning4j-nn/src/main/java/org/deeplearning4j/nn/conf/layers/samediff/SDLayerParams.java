package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.util.*;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonIgnoreProperties({"paramsList", "weightParamsList", "biasParamsList"})
@NoArgsConstructor
public class SDLayerParams {

    private Map<String,int[]> weightParams = new LinkedHashMap<>();
    private Map<String,int[]> biasParams = new LinkedHashMap<>();

    @JsonIgnore private List<String> paramsList;
    @JsonIgnore private List<String> weightParamsList;
    @JsonIgnore private List<String> biasParamsList;

    public SDLayerParams(@JsonProperty("weightParams") Map<String,int[]> weightParams,
                                @JsonProperty("biasParams") Map<String,int[]> biasParams){
        this.weightParams = weightParams;
        this.biasParams = biasParams;
    }

    public void addWeightParam(String paramKey, int[] paramShape) {
        weightParams.put(paramKey, paramShape);
        paramsList = null;
        weightParamsList = null;
        biasParamsList = null;
    }

    public void addBiasParam(String paramKey, int[] paramShape) {
        biasParams.put(paramKey, paramShape);
        paramsList = null;
        weightParamsList = null;
        biasParamsList = null;
    }

    public List<String> getParameterKeys() {
        if(paramsList == null) {
            List<String> out = new ArrayList<>();
            out.addAll(getWeightParameterKeys());
            out.addAll(getBiasParameterKeys());
            this.paramsList = Collections.unmodifiableList(out);
        }
        return paramsList;
    }

    public List<String> getWeightParameterKeys() {
        if(weightParamsList == null){
            weightParamsList = Collections.unmodifiableList(new ArrayList<>(weightParams.keySet()));
        }
        return weightParamsList;
    }

    public List<String> getBiasParameterKeys() {
        if(biasParamsList == null){
            biasParamsList = Collections.unmodifiableList(new ArrayList<>(biasParams.keySet()));
        }
        return biasParamsList;
    }

    public Map<String, int[]> getParamShapes() {
        Map<String, int[]> map = new LinkedHashMap<>();
        map.putAll(weightParams);
        map.putAll(biasParams);
        return map;
    }

    public void clear() {
        weightParams.clear();
        biasParams.clear();
        paramsList = null;
        weightParamsList = null;
        biasParamsList = null;
    }

    public boolean equals(Object o) {
        if(!(o instanceof SDLayerParams)){
            return false;
        }
        SDLayerParams s = (SDLayerParams)o;
        return equals(weightParams, s.weightParams) && equals(biasParams, s.biasParams);
    }

    private static boolean equals(Map<String,int[]> first, Map<String,int[]> second){
        //Helper method - Lombok equals method seems to have trouble with arrays...
        if(!first.keySet().equals(second.keySet())){
            return false;
        }
        for(Map.Entry<String,int[]> e : first.entrySet()){
            if(!Arrays.equals(e.getValue(), second.get(e.getKey()))){
                return false;
            }
        }
        return true;
    }

    public int hashCode() {
        return weightParams.hashCode() ^ biasParams.hashCode();
    }

    protected boolean canEqual(Object other) {
        return other instanceof SDLayerParams;
    }
}
