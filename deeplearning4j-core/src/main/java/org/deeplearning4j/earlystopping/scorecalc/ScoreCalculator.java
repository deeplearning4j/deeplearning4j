package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.Serializable;

/** ScoreCalculator interface is used to calculate a score for a MultiLayerNetwork.
 * For example, the loss function, test set accuracy, F1, or some other (possibly custom) metric.
 */
public interface ScoreCalculator<T extends Model> extends Serializable {

    /** Calculate the score for the given MultiLayerNetwork */
    double calculateScore(T network);
}
