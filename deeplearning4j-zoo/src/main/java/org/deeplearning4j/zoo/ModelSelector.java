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
    public static Map<ZooType,InstantiableModel> select(ZooType zooType, int numLabels, int seed, int iterations) {
        Map<ZooType,InstantiableModel> netmap = new HashMap<>();

        switch(zooType) {
            case ALL:
                netmap.putAll(ModelSelector.select(ZooType.CNN, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ZooType.RNN, numLabels, seed, iterations));
                break;
            // CNN models
            case CNN:
                netmap.putAll(ModelSelector.select(ZooType.ALEXNET, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ZooType.VGG16, numLabels, seed, iterations));
                break;
            case ALEXNET:
                netmap.put(ZooType.ALEXNET, new AlexNet(numLabels, seed, iterations));
                break;
            case LENET:
                netmap.put(ZooType.LENET, new LeNet(numLabels, seed, iterations));
                break;
            case INCEPTIONRESNETV1:
                netmap.put(ZooType.INCEPTIONRESNETV1, new InceptionResNetV1(numLabels, seed, iterations));
                break;
            case FACENETNN4SMALL2:
                netmap.put(ZooType.FACENETNN4SMALL2, new FaceNetNN4Small2(numLabels, seed, iterations));
                break;
            case VGG16:
                netmap.put(ZooType.VGG16, new VGG16(numLabels, seed, iterations));
                break;
            default:
//                // do nothing
        }

        if(netmap.size()==0) throw new IllegalArgumentException("Zero models have been selected for benchmarking.");

        return netmap;
    }
}
