package org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils;

import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Helper class with a static method that returns the label description
 * @author susaneraly
 */
public class ImageNetLabels {

    //FIXME
    private final static String jsonUrl = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json";
    private static ArrayList<String> predictionLabels = null;

    public static ArrayList<String> getLabels() {
        if (predictionLabels == null) {
            HashMap<String, ArrayList<String>> jsonMap;
            try {
                jsonMap = new ObjectMapper().readValue(new URL(jsonUrl), HashMap.class);
                predictionLabels = new ArrayList<>(jsonMap.size());
                for (int i = 0; i < jsonMap.size(); i++) {
                    predictionLabels.add(jsonMap.get(String.valueOf(i)).get(1));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return predictionLabels;
    }

    /*
        Returns the description of the nth class in the 1000 classes of ImageNet
     */
    public static String getLabel(int n) {
        if (predictionLabels == null) {
            getLabels();
        }
        return predictionLabels.get(n);
    }

}
