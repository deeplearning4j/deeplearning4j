package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.observation.Observation;

import java.util.ArrayList;
import java.util.List;

public class MockExperienceHandler implements ExperienceHandler<Integer, Transition<Integer>> {
    public List<StateActionPair<Integer>> addExperienceArgs = new ArrayList<StateActionPair<Integer>>();
    public Observation finalObservation;
    public boolean isGenerateTrainingBatchCalled;
    public boolean isResetCalled;

    @Override
    public void addExperience(Observation observation, Integer action, double reward, boolean isTerminal) {
        addExperienceArgs.add(new StateActionPair<>(observation, action, reward, isTerminal));
    }

    @Override
    public void setFinalObservation(Observation observation) {
        finalObservation = observation;
    }

    @Override
    public List<Transition<Integer>> generateTrainingBatch() {
        isGenerateTrainingBatchCalled = true;
        return new ArrayList<Transition<Integer>>() {
            {
                add(new Transition<Integer>(null, 0, 0.0, false));
            }
        };
    }

    @Override
    public void reset() {
        isResetCalled = true;
    }

    @Override
    public int getTrainingBatchSize() {
        return 1;
    }
}
