package org.deeplearning4j.nn.simple.binary;

import lombok.Data;

/**
 * Created by agibsonccc on 4/28/17.
 */
@Data
public class BinaryClassificationResult {
    private double decisionThreshold = 0.5;
    private double[] classWeights;

}
