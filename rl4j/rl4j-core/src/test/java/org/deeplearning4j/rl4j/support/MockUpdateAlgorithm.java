package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;

import java.util.ArrayList;
import java.util.List;

public class MockUpdateAlgorithm implements UpdateAlgorithm<MockNeuralNet> {

    public final List<List<StateActionPair<Integer>>> experiences = new ArrayList<List<StateActionPair<Integer>>>();

    @Override
    public Gradient[] computeGradients(MockNeuralNet current, List<StateActionPair<Integer>> experience) {
        experiences.add(experience);
        return new Gradient[0];
    }
}
