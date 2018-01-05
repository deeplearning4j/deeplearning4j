package org.deeplearning4j.zoo.util.imagenet;

import org.deeplearning4j.zoo.util.BaseLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Helper class with a static method that returns the label description.
 *
 * @author susaneraly
 */
public class ImageNetLabels extends BaseLabels {

    private static final String jsonResource = "imagenet_class_index.json";
    private ArrayList<String> predictionLabels = null;

    public ImageNetLabels() throws IOException {
        this.predictionLabels = getLabels();
    }

    protected ArrayList<String> getLabels() throws IOException {
        if (predictionLabels == null) {
            HashMap<String, ArrayList<String>> jsonMap;
            jsonMap = new ObjectMapper().readValue(this.getClass().getResourceAsStream(jsonResource), HashMap.class);
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

    /**
     * Given predictions from the trained model this method will return a string
     * listing the top five matches and the respective probabilities
     * @param predictions
     * @return
     */
    public String decodePredictions(INDArray predictions) {
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
