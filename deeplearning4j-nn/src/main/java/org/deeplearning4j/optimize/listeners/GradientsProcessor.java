package org.deeplearning4j.optimize.listeners;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * This class is suited for gradients extraction out of the model, and their sharing between models
 *
 * PLEASE NOTE: It operates on gradients as a whole, not partial gradients for layers
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class GradientsProcessor implements TrainingListener {
    protected String id = java.util.UUID.randomUUID().toString();

    @Getter
    protected Queue<SharedGradient> ownGradients = new ConcurrentLinkedQueue<>();
    @Getter protected Queue<SharedGradient> foreignGradients = new ConcurrentLinkedQueue<>();


    public void enqueueGradient(SharedGradient gradient) {
        if (gradient.getId().equals(id))
            return;

        foreignGradients.add(gradient);
    }

    @Override
    public void onEpochStart(Model model) {
        // no-op
    }

    @Override
    public void onEpochEnd(Model model) {
        // no-op
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {
        // no-op
    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {
        // no-op
    }

    @Override
    public void onGradientCalculation(Model model) {
        // no-op
    }

    /**
     * In this method we extract gradients from the model
     *
     * @param model Model
     */
    @Override
    public void onBackwardPass(Model model) {
        // Beware: this code block operates out of workspaces
        Gradient gradient = model.gradient();
        INDArray array = gradient.gradient().dup();
        //array.assign(0.0);

        log.info("Gradients length: {}; Mean number: {}", array.lengthLong(), array.meanNumber().doubleValue());

        // TODO: we want to push make gradient copy, and push it to host memory here

        ownGradients.add(new SharedGradient(id, array.detach()));
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {
        // no-op
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        // no-op
    }
}