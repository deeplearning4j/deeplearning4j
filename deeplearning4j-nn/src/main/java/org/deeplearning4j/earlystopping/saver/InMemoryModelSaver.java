package org.deeplearning4j.earlystopping.saver;

import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.nn.api.Model;

import java.io.IOException;

/** Save the best (and latest) models for early stopping training to memory for later retrieval
 * <b>Note</b>: Assumes that network is cloneable via .clone() method
 * @param <T> Type of model. For example, {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork} or {@link org.deeplearning4j.nn.graph.ComputationGraph}
 */
public class InMemoryModelSaver<T extends Model> implements EarlyStoppingModelSaver<T> {

    private transient T bestModel;
    private transient T latestModel;

    @Override
    @SuppressWarnings("unchecked")
    public void saveBestModel(T net, double score) throws IOException {
        try {
            //Necessary because close is protected :S
            bestModel = (T) (net.getClass().getDeclaredMethod("clone")).invoke(net);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void saveLatestModel(T net, double score) throws IOException {
        try {
            //Necessary because close is protected :S
            latestModel = (T) (net.getClass().getDeclaredMethod("clone")).invoke(net);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public T getBestModel() throws IOException {
        return bestModel;
    }

    @Override
    public T getLatestModel() throws IOException {
        return latestModel;
    }

    @Override
    public String toString() {
        return "InMemoryModelSaver()";
    }
}
