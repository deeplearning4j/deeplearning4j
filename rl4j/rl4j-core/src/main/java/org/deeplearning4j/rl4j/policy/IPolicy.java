package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface IPolicy<ACTION> {
    @Deprecated
    <O, AS extends ActionSpace<ACTION>> double play(MDP<O, ACTION, AS> mdp, IHistoryProcessor hp);

    @Deprecated
    ACTION nextAction(INDArray input);

    ACTION nextAction(Observation observation);

    void reset();
}
