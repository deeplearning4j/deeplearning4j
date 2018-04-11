package org.deeplearning4j.optimize.api;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

/**
 * Extend this and selectively override the methods you will actually use, rather than have a bunch
 * of no-op methods
 */
public abstract class BaseTrainingListener implements TrainingListener {

    @Override
    public void onEpochStart(Model model) {
        //No op
    }

    @Override
    public void onEpochEnd(Model model) {
        //No op
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        //No op
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        //No op
    }

    @Override
    public void onGradientCalculation(Model model) {
        //No op
    }

    @Override
    public void onBackwardPass(Model model) {
        //No op
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        //No op
    }
}
