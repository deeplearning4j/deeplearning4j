package org.deeplearning4j.earlystopping.listener;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.nn.api.Model;

/**EarlyStoppingListener is a listener interface for conducting early stopping training.
 * It provides onStart, onEpoch, and onCompletion methods, which are called as appropriate
 * @param <T> Type of model. For example, {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork} or {@link org.deeplearning4j.nn.graph.ComputationGraph}
 */
public interface EarlyStoppingListener<T extends Model> {

    /**Method to be called when early stopping training is first started
     */
    void onStart(EarlyStoppingConfiguration<T> esConfig, T net);

    /**Method that is called at the end of each epoch completed during early stopping training
     * @param epochNum The number of the epoch just completed (starting at 0)
     * @param score The score calculated
     * @param esConfig Configuration
     * @param net Network (current)
     */
    void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<T> esConfig, T net);

    /**Method that is called at the end of early stopping training
     * @param esResult The early stopping result. Provides details of why early stopping training was terminated, etc
     */
    void onCompletion(EarlyStoppingResult esResult);

}
