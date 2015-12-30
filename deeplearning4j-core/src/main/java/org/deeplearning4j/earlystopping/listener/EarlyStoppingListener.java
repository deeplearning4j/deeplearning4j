package org.deeplearning4j.earlystopping.listener;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public interface EarlyStoppingListener {

    void onStart(EarlyStoppingConfiguration esConfig, MultiLayerNetwork net);

    void onEpoch(int epochNum, double score, EarlyStoppingConfiguration esConfig, MultiLayerNetwork net);

    void onCompletion(EarlyStoppingResult esResult);

}
