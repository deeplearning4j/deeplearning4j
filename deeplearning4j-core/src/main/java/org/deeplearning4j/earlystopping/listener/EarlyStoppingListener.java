package org.deeplearning4j.earlystopping.listener;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.nn.api.Model;

public interface EarlyStoppingListener<T extends Model> {

    void onStart(EarlyStoppingConfiguration esConfig, T net);

    void onEpoch(int epochNum, double score, EarlyStoppingConfiguration esConfig, T net);

    void onCompletion(EarlyStoppingResult esResult);

}
