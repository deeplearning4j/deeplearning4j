package org.deeplearning4j.caffe.dag;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * @author jeffreytang
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@ToString(of = {"name", "layerSubType"})
@EqualsAndHashCode(exclude = {"bottomNodeList"})
public class CaffeNode implements Node {
    private String name;
    private LayerType layerType;
    private LayerSubType layerSubType;
    private Map<String, Object> metaData = new HashMap<>();
    private List<INDArray> data = new ArrayList<>();
    private List<CaffeNode> bottomNodeList;

    public enum LayerType {
        HIDDEN, PROCESSING, VISION,
        INNERPRODUCT, CONNECTOR, DROPOUT, LOSS
    }

    public enum LayerSubType {
        //HIDDEN
        CONVOLUTION, POOLING, RELU, SIGMOID, TANH,
        SOFTMAX, SOFTMAXWITHLOSS, SOFTMAX_LOSS,
        EUCLIDEANLOSS, SIGMOIDCROSSENTROPYLOSS,
        //PROCESSING
        FLATTEN, RESHAPE, CONCAT, SLICE, SPLIT,
        //INNERPRODUCT
        INNERPRODUCT,
        //CONNECTOR
        CONNECTOR,
        //DROPOUT
        DROPOUT

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

    public CaffeNode(String name, LayerType layerType, LayerSubType layerSubType,
                     Map<String, Object> metaData, List<INDArray> data) {
        this.name = name;
        this.layerType = layerType;
        this.layerSubType = layerSubType;
        this.metaData = metaData;
        this.data = data;
    }

    public CaffeNode(String name, LayerType layerType, LayerSubType layerSubType, List<CaffeNode> bottomNodeSet) {
        this.name = name;
        this.layerType = layerType;
        this.layerSubType = layerSubType;
        this.bottomNodeList = bottomNodeSet;
    }

}
