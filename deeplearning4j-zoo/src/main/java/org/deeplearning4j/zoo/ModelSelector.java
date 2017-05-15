package org.deeplearning4j.zoo;

import org.deeplearning4j.zoo.model.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Helper class for selecting multiple models from the zoo.
 *
 * @author Justin Long (crockpotveggies)
 */
public class ModelSelector {
    public static Map<ModelType,InstantiableModel> select(ModelType modelType, int numLabels, int seed, int iterations) {
        Map<ModelType,InstantiableModel> netmap = new HashMap<>();

        switch(modelType) {
            case ALL:
                netmap.putAll(ModelSelector.select(ModelType.CNN, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.RNN, numLabels, seed, iterations));
                break;
            // CNN models
            case CNN:
                netmap.putAll(ModelSelector.select(ModelType.ALEXNET, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.VGG16, numLabels, seed, iterations));
                break;
            case ALEXNET:
                netmap.put(ModelType.ALEXNET, new AlexNet(numLabels, seed, iterations));
                break;
            case LENET:
                netmap.put(ModelType.LENET, new LeNet(numLabels, seed, iterations));
                break;
            case INCEPTIONRESNETV1:
                netmap.put(ModelType.INCEPTIONRESNETV1, new InceptionResNetV1(numLabels, seed, iterations));
                break;
            case FACENETNN4SMALL2:
                netmap.put(ModelType.FACENETNN4SMALL2, new FaceNetNN4Small2(numLabels, seed, iterations));
                break;
            case VGG16:
                netmap.put(ModelType.VGG16, new VGG16(numLabels, seed, iterations));
                break;
            default:
//                // do nothing
        }

        if(netmap.size()==0) throw new IllegalArgumentException("Zero models have been selected for benchmarking.");

        return netmap;
    }
}
