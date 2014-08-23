package org.deeplearning4j.example.classification.labels;



import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.FeatureUtil;

import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 8/5/14.
 */
public class LabelsExample {




    public static void main(String[] args) {
        List<String> myLabels = Arrays.asList("1","2");
        INDArray labelMatrix = FeatureUtil.toOutcomeVector(0, myLabels.size());
    }


}
