package org.deeplearning4j.rl4j.mdp;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MDPRunner implements IMDPRunner {
    public <O extends Encodable, A, AS extends ActionSpace<A>> Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp) {
        O obs = mdp.reset();
        O nextO = obs;

        int step = 0;
        double reward = 0;

        return new Learning.InitMdp(step, nextO, reward);

    }}
