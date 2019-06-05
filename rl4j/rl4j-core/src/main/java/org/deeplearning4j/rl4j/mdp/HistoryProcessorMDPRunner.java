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

public class HistoryProcessorMDPRunner implements IMDPRunner {

    private final IHistoryProcessor historyProcessor;

    public HistoryProcessorMDPRunner(IHistoryProcessor historyProcessor) {

        this.historyProcessor = historyProcessor;
    }

    public <O extends Encodable, A, AS extends ActionSpace<A>> Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp) {

        O obs = mdp.reset();

        O nextO = obs;

        int step = 0;
        double reward = 0;

        int skipFrame = historyProcessor.getConf().getSkipFrame();
        int requiredFrame = skipFrame * (historyProcessor.getConf().getHistoryLength() - 1);

        while (step < requiredFrame) {
            INDArray input = Learning.getInput(mdp, obs);

            historyProcessor.record(input);

            A action = mdp.getActionSpace().noOp(); //by convention should be the NO_OP
            if (step % skipFrame == 0) {
                historyProcessor.add(input);
            }

            StepReply<O> stepReply = mdp.step(action);
            reward += stepReply.getReward();
            nextO = stepReply.getObservation();

            step++;

        }

        return new Learning.InitMdp(step, nextO, reward);
    }

    // FIXME: Work in progress
    public INDArray getHStack(INDArray input, IMDPRunner.GetHStackContext context) {
        INDArray[] history = context.getHistory();

        if (history == null) {
            historyProcessor.add(input);
            history = historyProcessor.getHistory();
            context.setHistory(history);
        }
        //concat the history into a single INDArray input
        INDArray hstack = Transition.concat(Transition.dup(history));
        hstack.muli(1.0 / historyProcessor.getScale()); // FIXME: change to more generic normalization

        // FIXME: remove
        // Reshape hstack to make a 1-element batch
        // Special case: the method Learning.getInput(MDP<O, A, AS> mdp, O obs) will output 2D array when observations are 1D
        if (hstack.shape().length > 2) {
            hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape())));
        }

        return hstack;
    }

}
