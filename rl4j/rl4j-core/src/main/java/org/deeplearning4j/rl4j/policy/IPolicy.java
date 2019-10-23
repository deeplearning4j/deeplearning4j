package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface IPolicy<O extends Encodable, A> {
    <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp, IHistoryProcessor hp);
    A nextAction(INDArray input);
}
