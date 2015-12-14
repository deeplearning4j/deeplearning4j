package org.deeplearning4j.nn.earlystopping.scorecalc;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/** ScoreCalculator interface is used to calculate a score for a MultiLayerNetwork.
 * For example, the loss function, test set accuracy, F1, or some other (possibly custom) metric.
 */
public interface ScoreCalculator {

    double calculateScore(MultiLayerNetwork network);
}
