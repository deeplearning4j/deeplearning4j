package org.deeplearning4j.optimize.api;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.List;
import java.util.Map;

/**
 * TrainingListener: an extension of {@link IterationListener} that adds onEpochStart, onEpochEnd, onForwardPass and
 * onBackwardPass methods
 *
 * @author Alex Black
 */
public interface TrainingListener extends IterationListener {

    /**
     * Called once at the start of each epoch, when using methods such as {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork#fit(DataSetIterator)},
     * {@link org.deeplearning4j.nn.graph.ComputationGraph#fit(DataSetIterator)} or {@link org.deeplearning4j.nn.graph.ComputationGraph#fit(MultiDataSetIterator)}
     */
    void onEpochStart(Model model);

    /**
     * Called once at the end of each epoch, when using methods such as {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork#fit(DataSetIterator)},
     * {@link org.deeplearning4j.nn.graph.ComputationGraph#fit(DataSetIterator)} or {@link org.deeplearning4j.nn.graph.ComputationGraph#fit(MultiDataSetIterator)}
     */
    void onEpochEnd(Model model);

    /**
     * Called once per iteration (forward pass) for activations (usually for a {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork}),
     * only at training time
     *
     * @param model       Model
     * @param activations Layer activations (including input)
     */
    void onForwardPass(Model model, List<INDArray> activations);

    /**
     * Called once per iteration (forward pass) for activations (usually for a {@link org.deeplearning4j.nn.graph.ComputationGraph}),
     * only at training time
     *
     * @param model       Model
     * @param activations Layer activations (including input)
     */
    void onForwardPass(Model model, Map<String, INDArray> activations);

    /**
     * Called once per iteration (backward pass) after gradients have been calculated.
     * Gradients are available via {@link Model#gradient()}
     *
     * @param model Model
     */
    void onBackwardPass(Model model);

}
