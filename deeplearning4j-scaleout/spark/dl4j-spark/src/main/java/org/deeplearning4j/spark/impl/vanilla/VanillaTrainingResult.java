package org.deeplearning4j.spark.impl.vanilla;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.spark.api.TrainingResult;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 14/06/2016.
 */
@AllArgsConstructor @Data
public class VanillaTrainingResult implements TrainingResult {

    private final INDArray parameters;
    private final Updater updater;
    private final double score;

}
