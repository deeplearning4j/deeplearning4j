package org.deeplearning4j.caffe.dag;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.dag.Node;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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
    Map<String, Object> metaData = new HashMap<>();
    List<INDArray> data = new ArrayList<>();

    public enum LayerType {
        HIDDEN, PROCESSING,
        INNERPRODUCT, CONNECTOR
    }

    public enum LayerSubType {
        //HIDDEN
        CONVOLUTION, POOLING, RELU, SIGMOID, TANH,
        SOFTMAX, SOFTMAXWITHLOSS,
        EUCLIDEANLOSS, SIGMOIDCROSSENTROPYLOSS,
        //PROCESSING
        FLATTEN, RESHAPE, CONCAT, SLICE, SPLIT,
        //INNERPRODUCT
        INNERPRODUCT,
        //CONNECTOR
        CONNECTOR
    }

    public CaffeNode(String name, LayerType layerType, LayerSubType layerSubType) {
        this.name = name;
        this.layerType = layerType;
        this.layerSubType = layerSubType;
    }

    public CaffeNode(String name, LayerType layerType, LayerSubType layerSubType, Map<String, Object> metaData) {
        this.name = name;
        this.layerType = layerType;
        this.layerSubType = layerSubType;
        this.metaData = metaData;
    }

}
