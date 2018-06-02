package org.deeplearning4j.zoo.util.imagenet;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.zoo.util.BaseLabels;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Helper class with a static method that returns the label description.
 *
 * @author susaneraly
 */
public class ImageNetLabels extends BaseLabels {

    private static final String jsonResource = "imagenet_class_index.json";
    private ArrayList<String> predictionLabels;

    public ImageNetLabels() throws IOException {
        this.predictionLabels = getLabels();
    }

    protected ArrayList<String> getLabels() throws IOException {

        File localFile = getResourceFile();
        if (predictionLabels == null) {
            HashMap<String, ArrayList<String>> jsonMap;
            jsonMap = new ObjectMapper().readValue(localFile, HashMap.class);
            predictionLabels = new ArrayList<>(jsonMap.size());
            for (int i = 0; i < jsonMap.size(); i++) {
                predictionLabels.add(jsonMap.get(String.valueOf(i)).get(1));
            }
        }
        return predictionLabels;
    }

    /**
     * Returns the description of tne nth class in the 1000 classes of ImageNet.
     * @param n
     * @return
     */
    public String getLabel(int n) {
        return predictionLabels.get(n);
    }

    @Override
    protected URL getURL() {
        try {
            return DL4JResources.getURL("resources/imagenet/" + jsonResource);
        } catch (MalformedURLException e){
            throw new RuntimeException(e);
        }
    }

    @Override
    protected String resourceName() {
        return jsonResource;
    }

    @Override
    protected String resourceMD5() {
        return "c2c37ea517e94d9795004a39431a14cb";
    }

    /**
     * Given predictions from the trained model this method will return a string
     * listing the top five matches and the respective probabilities
     * @param predictions
     * @return
     */
    public String decodePredictions(INDArray predictions) {
        Preconditions.checkState(predictions.size(1) == predictionLabels.size(), "Invalid input array:" +
                " expected array with size(1) equal to numLabels (%s), got array with shape %s", predictionLabels.size(), predictions.shape());

        String predictionDescription = "";
        int[] top5 = new int[5];
        float[] top5Prob = new float[5];

        //brute force collect top 5
        int i = 0;
        for (int batch = 0; batch < predictions.size(0); batch++) {
            predictionDescription += "Predictions for batch ";
            if (predictions.size(0) > 1) {
                predictionDescription += String.valueOf(batch);
            }
            predictionDescription += " :";
            INDArray currentBatch = predictions.getRow(batch).dup();
            while (i < 5) {
                top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
                top5Prob[i] = currentBatch.getFloat(batch, top5[i]);
                currentBatch.putScalar(0, top5[i], 0);
                predictionDescription += "\n\t" + String.format("%3f", top5Prob[i] * 100) + "%, "
                                + predictionLabels.get(top5[i]);
                i++;
            }
        }
        return predictionDescription;
    }

}
