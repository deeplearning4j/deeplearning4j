package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;

public interface ILearningInitializer<O  extends Encodable, A, AS extends ActionSpace<A>> {
        Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp);
}
