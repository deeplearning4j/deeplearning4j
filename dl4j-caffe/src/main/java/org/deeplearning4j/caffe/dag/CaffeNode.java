package org.deeplearning4j.caffe.dag;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.dag.Node;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@ToString(of = {"name", "layerSubType"})
public class CaffeNode implements Node {
    private String name;
    private LayerType layerType;
    private LayerSubType layerSubType;
    Map<String, Object> data = new HashMap<>();

    public enum LayerType {
        HIDDEN, PROCESSING,
        INNERPRODUCT, CONNECTOR
    }

    public enum LayerSubType {
        CONVOLUTION, POOLING, RELU, SIGMOID, TANH,
        SOFTMAX, SOFTMAXWITHLOSS,
        EUCLIDEANLOSS, SIGMOIDCROSSENTROPYLOSS,
        FLATTEN, RESHAPE, CONCAT, SLICE, SPLIT,
        CONNECTOR
    }

    public CaffeNode(String name, LayerType layerType, LayerSubType layerSubType) {
        this.name = name;
        this.layerType = layerType;
        this.layerSubType = layerSubType;
    }
}
