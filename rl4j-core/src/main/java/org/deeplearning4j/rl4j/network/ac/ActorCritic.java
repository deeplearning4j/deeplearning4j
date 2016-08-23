package org.deeplearning4j.rl4j.network.ac;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 *
 * Standard implementation of ActorCritic
 */
public class ActorCritic implements IActorCritic {

    final protected ComputationGraph cg;


    public ActorCritic(ComputationGraph cg) {
        this.cg = cg;
    }

    public void fit(INDArray input, INDArray[] labels) {
        cg.fit(new INDArray[]{input}, labels);
    }


    public INDArray[] outputAll(INDArray batch) {
        return cg.output(batch);
    }

    public ActorCritic clone() {
        return new ActorCritic(cg.clone());
    }

    public Gradient gradient(INDArray input, INDArray[] labels) {
        cg.setInput(0, input);
        cg.setLabels(labels);
        cg.computeGradientAndScore();
        return cg.gradient();
    }


    public void applyGradient(Gradient gradient) {
        cg.getUpdater().update(cg, gradient, 1, 32);
        cg.params().subi(gradient.gradient());
    }

    public double getLatestScore() {
        return cg.score();
    }

    public void save(OutputStream stream) {
        try {
            ModelSerializer.writeModel(cg, stream, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void save(String path) {
        try {
            ModelSerializer.writeModel(cg, path, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

