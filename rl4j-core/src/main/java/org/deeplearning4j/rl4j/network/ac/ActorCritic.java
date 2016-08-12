package org.deeplearning4j.rl4j.network.ac;

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 * Actor Critic class
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
        throw new NotImplementedException("calculate gradient");
    }


    public void applyGradient(Gradient gradient) {
        throw new NotImplementedException("apply gradient");
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

