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
 * Standard implementation of ActorCriticCompGraph
 */
public class ActorCriticCompGraph implements IActorCritic {

    final protected ComputationGraph cg;


    public ActorCriticCompGraph(ComputationGraph cg) {
        this.cg = cg;
    }

    public void fit(INDArray input, INDArray[] labels) {
        cg.fit(new INDArray[] {input}, labels);
    }


    public INDArray[] outputAll(INDArray batch) {
        return cg.output(batch);
    }

    public ActorCriticCompGraph clone() {
        return new ActorCriticCompGraph(cg.clone());
    }

    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        cg.setInput(0, input);
        cg.setLabels(labels);
        cg.computeGradientAndScore();
        return new Gradient[] {cg.gradient()};
    }


    public void applyGradient(Gradient[] gradient, int batchSize) {
        cg.getUpdater().update(gradient[0], 1, batchSize);
        cg.params().subi(gradient[0].gradient());
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

