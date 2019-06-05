package org.deeplearning4j.rl4j.mdp;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

public class MDPRunner implements IMDPRunner {
    public <O extends Encodable, A, AS extends ActionSpace<A>> Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp) {
        O obs = mdp.reset();
        O nextO = obs;

        int step = 0;
        double reward = 0;

        return new Learning.InitMdp(step, nextO, reward);

    }

    // FIXME: Work in progress
    public INDArray getHStack(INDArray input, IMDPRunner.GetHStackContext context) {
        INDArray[] history = context.getHistory();

        if (history == null) {
            // FIXME: maybe have a History class with its own init/update behavior
            history = new INDArray[] {input};
            context.setHistory(history);
        }

        INDArray hstack = Transition.concat(Transition.dup(history)); // FIXME: check if equivalent to hstack = history[0] in this case

        // FIXME: remove
        // Reshape hstack to make a 1-element batch
        // Special case: the method Learning.getInput(MDP<O, A, AS> mdp, O obs) will output 2D array when observations are 1D
        if (hstack.shape().length > 2) {
            hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape())));
        }

        return hstack;
    }
}
