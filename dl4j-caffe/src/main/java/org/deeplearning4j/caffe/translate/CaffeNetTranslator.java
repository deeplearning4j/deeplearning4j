package org.deeplearning4j.caffe.translate;

import lombok.Data;
import org.deeplearning4j.caffe.common.NNCofigBuilderContainer;
import org.deeplearning4j.caffe.dag.CaffeNode.*;
import org.deeplearning4j.caffe.projo.Caffe.NetParameter;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
@Data
public class CaffeNetTranslator {
    private Map<LayerSubType, Object> layerMapping;
    private Map<LayerSubType, Map<String, String>> layerParamMapping;

    private void populateLayerMapping() {

        // All the dense layers
        layerMapping = new HashMap<LayerSubType, Object>() {{
            put(LayerSubType.CONVOLUTION, new ConvolutionLayer.Builder());
            put(LayerSubType.POOLING, new SubsamplingLayer.Builder());
            put(LayerSubType.RELU, new DenseLayer.Builder().activation("relu"));
            put(LayerSubType.SIGMOID, new DenseLayer.Builder().activation("sigmoid"));
            put(LayerSubType.TANH, new DenseLayer.Builder().activation("tanh"));
            put(LayerSubType.SOFTMAX, new DenseLayer.Builder().activation("softmax"));
            put(LayerSubType.SOFTMAXWITHLOSS, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .activation("softmax"));
            put(LayerSubType.EUCLIDEANLOSS, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE));
            put(LayerSubType.SIGMOIDCROSSENTROPYLOSS, new OutputLayer.Builder()
                    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)); //TODO: Fix loss functions
            put(LayerSubType.FLATTEN, null);
            put(LayerSubType.RESHAPE, null);
            put(LayerSubType.CONCAT, null);
            put(LayerSubType.SLICE, null);
            put(LayerSubType.SPLIT, null);
        }};
    }

    private void populateLayerParamMapping() {
        final Map<String, String> convolutionMapping = new HashMap<String, String>() {{
            put("numOutput_", "nOut");
            put("kernelSize_", "kernelSize");
            put("pad_", "");
            put("stride_", "");
            put("weightFiller_", "weightInit");
            put("biasFiller_", "");
            put("group_", "");
        }};

        final Map<String, String> poolMapping = new HashMap<String, String>() {{
            put("pool_", "poolType");
            put("kernelSize_", "kernelSize");
            put("stride_", "stride");
        }};

        final Map<String, String> innerProductMapping = new HashMap<String, String>() {{
            put("numOutput_", "");
            put("weightFiller_", "weightInit");
            put("biasFiller_", "");
            put("biasTerm_", "");
        }};

        layerParamMapping = new HashMap<LayerSubType, Map<String, String>>() {{
            put(LayerSubType.CONVOLUTION, convolutionMapping);
            put(LayerSubType.POOLING, poolMapping);
            put(LayerSubType.INNERPRODUCT, innerProductMapping);
        }};
    }

    public CaffeNetTranslator() {
        populateLayerMapping();
        populateLayerParamMapping();
    }

    public NNCofigBuilderContainer translate(NetParameter net, NNCofigBuilderContainer builderContainer) {

        return builderContainer; // dummy
    }
}
