package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;

public class LearningInitializer<O  extends Encodable, A, AS extends ActionSpace<A>> implements ILearningInitializer<O, A, AS> {

    @Override
    public Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp) {
        O obs = mdp.reset();
        O nextO = obs;

        int step = 0;
        double reward = 0;

        return new Learning.InitMdp(step, nextO, reward);
    }
}
