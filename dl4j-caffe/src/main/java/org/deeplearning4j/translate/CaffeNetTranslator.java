package org.deeplearning4j.translate;

import org.deeplearning4j.caffeprojo.Caffe.NetParameter;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.common.NNCofigBuilderContainer;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class CaffeNetTranslator {

    public Map<String, Object> unknownLayerMapping;
    public Map<String, Object> hiddenLayerMapping;
    public Map<String, Object> lossLayerMapping;
    public Map<String, Object> processLayerMapping;
    public Map<String, String> layerParamMapping;

    private void populateLayerMapping() {
        // All the irrelevant layers
        unknownLayerMapping = new HashMap<String, Object>() {{
            put("data", '0');
            put("memorydata", '0');
            put("hdf5data", '0');
            put("hdf5output", '0');
            put("accuracy", '0');
            put("absval", '0');
            put("power", '0');
            put("bnll", '0');
            put("hingeloss", '0');
        }};

        // All the dense layers
        hiddenLayerMapping = new HashMap<String, Object>() {{
            put("convolution", new ConvolutionLayer.Builder());
            put("pooling", new SubsamplingLayer.Builder());
            put("relu", new DenseLayer.Builder().activation("relu"));
            put("sigmoid", new DenseLayer.Builder().activation("sigmoid"));
            put("tanh", new DenseLayer.Builder().activation("tanh"));
            put("softmax", new DenseLayer.Builder().activation("softmax"));
            put("softmaxwithloss", new OutputLayer.Builder().activation("softmax")
                    .loss(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY));
            put("euclideanloss", new OutputLayer.Builder().loss(LossFunctions.LossFunction.MSE));
            put("sigmoidcrossentropyloss", new OutputLayer.Builder()
                    .loss(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)); //TODO: Fix loss functions
        }};

        processLayerMapping = new HashMap<String, Object>() {{
            put("split", '1');
            put("flatten", '1');
            put("reshape", '1');
            put("concat", '1');
            put("slice", '1');
            put("innerproduct", '1');
        }};

    }

    private void populateLayerParamMapping() {
        Map convolutionMapping = new HashMap<String, String>() {{
            put("numOutput_", "nOut");
            put("kernelSize_", "kernelSize");
            put("pad_", "");
            put("stride_", "");
            put("weightFiller_", "weightInit");
            put("biasFiller_", "");
            put("group_", "");
        }};

        Map poolMapping = new HashMap<String, String>() {{
            put("pool_", "poolType");
            put("kernelSize_", "kernelSize");
            put("stride_", "stride");
        }};

        Map innerProductMapping = new HashMap<String, String>() {{
            put("numOutput_", "");
            put("weightFiller_", "weightInit");
            put("biasFiller_", "");
            put("biasTerm_", "");
        }};

        layerParamMapping = new HashMap<String, String>() {{
           put("", "");
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
