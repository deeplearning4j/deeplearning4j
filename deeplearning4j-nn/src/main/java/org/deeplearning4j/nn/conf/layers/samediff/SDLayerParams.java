package org.deeplearning4j.nn.conf.layers.samediff;

import com.google.common.base.Preconditions;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.*;

/**
 * SDLayerParams is used to define the parameters for a Deeplearning4j SameDiff layer
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"paramsList", "weightParamsList", "biasParamsList"})
@NoArgsConstructor
@Data
public class SDLayerParams implements Serializable {

    private Map<String,long[]> weightParams = new LinkedHashMap<>();
    private Map<String,long[]> biasParams = new LinkedHashMap<>();

    @JsonIgnore private List<String> paramsList;
    @JsonIgnore private List<String> weightParamsList;
    @JsonIgnore private List<String> biasParamsList;

    public SDLayerParams(@JsonProperty("weightParams") Map<String,long[]> weightParams,
                                @JsonProperty("biasParams") Map<String,long[]> biasParams){
        this.weightParams = weightParams;
        this.biasParams = biasParams;
    }

    /**
     * Add a weight parameter to the layer, with the specified shape. For example, a standard fully connected layer
     * could have weight parameters with shape [numInputs, layerSize]
     *
     * @param paramKey   The parameter key (name) for the weight parameter
     * @param paramShape Shape of the weight parameter array
     */
    public void addWeightParam(@NonNull String paramKey, @NonNull long... paramShape) {
        Preconditions.checkArgument(paramShape.length > 0, "Provided weight parameter shape is" +
                " invalid: length 0 provided for shape. Parameter: " + paramKey);
        weightParams.put(paramKey, paramShape);
        paramsList = null;
        weightParamsList = null;
        biasParamsList = null;
    }

    /**
     * Add a bias parameter to the layer, with the specified shape. For example, a standard fully connected layer
     * could have bias parameters with shape [1, layerSize]
     *
     * @param paramKey   The parameter key (name) for the bias parameter
     * @param paramShape Shape of the bias parameter array
     */
    public void addBiasParam(@NonNull String paramKey, @NonNull long[] paramShape) {
        Preconditions.checkArgument(paramShape.length > 0, "Provided mia- parameter shape is" +
                " invalid: length 0 provided for shape. Parameter: " + paramKey);
        biasParams.put(paramKey, paramShape);
        paramsList = null;
        weightParamsList = null;
        biasParamsList = null;
    }

    /**
     * @return Get a list of parameter names / keys (previously added via {@link #addWeightParam(String, int...)} and
     * {@link #addBiasParam(String, int[])}
     */
    @JsonIgnore
    public List<String> getParameterKeys() {
        if(paramsList == null) {
            List<String> out = new ArrayList<>();
            out.addAll(getWeightParameterKeys());
            out.addAll(getBiasParameterKeys());
            this.paramsList = Collections.unmodifiableList(out);
        }
        return paramsList;
    }

    /**
     * @return Get a list of parameter names / keys for weight parameters only, previously added via
     * {@link #addWeightParam(String, int...)}
     */
    @JsonIgnore
    public List<String> getWeightParameterKeys() {
        if(weightParamsList == null){
            weightParamsList = Collections.unmodifiableList(new ArrayList<>(weightParams.keySet()));
        }
        return weightParamsList;
    }

    /**
     * @return Get a list of parameter names / keys for weight parameters only, previously added via
     * {@link #addWeightParam(String, int...)}
     */
    @JsonIgnore
    public List<String> getBiasParameterKeys() {
        if(biasParamsList == null){
            biasParamsList = Collections.unmodifiableList(new ArrayList<>(biasParams.keySet()));
        }
        return biasParamsList;
    }

    /**
     * Get the parameter shapes for all parameters
     *
     * @return Map of parameter shapes, by parameter
     */
    @JsonIgnore
    public Map<String, long[]> getParamShapes() {
        Map<String, long[]> map = new LinkedHashMap<>();
        map.putAll(weightParams);
        map.putAll(biasParams);
        return map;
    }

    /**
     * Clear any previously set weight/bias parameters (including their shapes)
     */
    public void clear() {
        weightParams.clear();
        biasParams.clear();
        paramsList = null;
        weightParamsList = null;
        biasParamsList = null;
    }

    @Override
    public boolean equals(Object o) {
        if(!(o instanceof SDLayerParams)){
            return false;
        }
        SDLayerParams s = (SDLayerParams)o;
        return equals(weightParams, s.weightParams) && equals(biasParams, s.biasParams);
    }

    private static boolean equals(Map<String,long[]> first, Map<String,long[]> second){
        //Helper method - Lombok equals method seems to have trouble with arrays...
        if(!first.keySet().equals(second.keySet())){
            return false;
        }
        for(Map.Entry<String,long[]> e : first.entrySet()){
            if(!Arrays.equals(e.getValue(), second.get(e.getKey()))){
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return weightParams.hashCode() ^ biasParams.hashCode();
    }
}
