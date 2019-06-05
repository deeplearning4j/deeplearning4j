package org.deeplearning4j.rl4j.mdp;

import lombok.Setter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.deeplearning4j.rl4j.mdp.BaseMDPRunner;

public class HistoryProcessorMDPRunner<O extends Encodable, A> extends BaseMDPRunner<O, A> {

    private final IHistoryProcessor historyProcessor;

    @Setter
    private int step;

    public HistoryProcessorMDPRunner(IHistoryProcessor historyProcessor) {

        this.historyProcessor = historyProcessor;
    }

    public <AS extends ActionSpace<A>> Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp) {

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
    public INDArray getHStack(INDArray input) {
        INDArray[] history = getHistory();

        if (history == null) {
            historyProcessor.add(input);
            history = historyProcessor.getHistory();
            setHistory(history);
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

    public A getNextAction(IDQN currentDQN, Policy<O, A> policy, INDArray input) {
        int skipFrame = historyProcessor.getConf().getSkipFrame();

        setMaxQ(Double.NaN); //ignore if Nan for stats

        //if step of training, just repeat lastAction
        if (step % skipFrame != 0) {
            return null;
        } else {
            return super.getNextAction(currentDQN, policy, input);
        }
    }
}
