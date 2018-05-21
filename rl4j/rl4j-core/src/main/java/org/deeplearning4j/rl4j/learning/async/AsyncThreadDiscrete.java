package org.deeplearning4j.rl4j.learning.async;

import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Async Learning specialized for the Discrete Domain
 *
 */
public abstract class AsyncThreadDiscrete<O extends Encodable, NN extends NeuralNet>
                extends AsyncThread<O, Integer, DiscreteSpace, NN> {

    @Getter
    private NN current;

    public AsyncThreadDiscrete(AsyncGlobal<NN> asyncGlobal, int threadNumber) {
        super(asyncGlobal, threadNumber);
        synchronized (asyncGlobal) {
            current = (NN)asyncGlobal.getCurrent().clone();
        }
    }

    /**
     * "Subepoch"  correspond to the t_max-step iterations
     * that stack rewards with t_max MiniTrans
     *
     * @param sObs the obs to start from
     * @param nstep the number of max nstep (step until t_max or state is terminal)
     * @return subepoch training informations
     */
    public SubEpochReturn<O> trainSubEpoch(O sObs, int nstep) {

        synchronized (getAsyncGlobal()) {
            current.copy(getAsyncGlobal().getCurrent());
        }
        Stack<MiniTrans<Integer>> rewards = new Stack<>();

        O obs = sObs;
        Policy<O, Integer> policy = getPolicy(current);

        Integer action;
        Integer lastAction = null;
        IHistoryProcessor hp = getHistoryProcessor();
        int skipFrame = hp != null ? hp.getConf().getSkipFrame() : 1;

        double reward = 0;
        double accuReward = 0;
        int i = 0;
        while (!getMdp().isDone() && i < nstep * skipFrame) {

            INDArray input = Learning.getInput(getMdp(), obs);
            INDArray hstack = null;

            if (hp != null) {
                hp.record(input);
            }

            //if step of training, just repeat lastAction
            if (i % skipFrame != 0 && lastAction != null) {
                action = lastAction;
            } else {
                hstack = processHistory(input);
                action = policy.nextAction(hstack);
            }

            StepReply<O> stepReply = getMdp().step(action);
            accuReward += stepReply.getReward() * getConf().getRewardFactor();

            //if it's not a skipped frame, you can do a step of training
            if (i % skipFrame == 0 || lastAction == null || stepReply.isDone()) {
                obs = stepReply.getObservation();

                if (hstack == null) {
                    hstack = processHistory(input);
                }
                INDArray[] output = current.outputAll(hstack);
                rewards.add(new MiniTrans(hstack, action, output, accuReward));

                accuReward = 0;
            }

            reward += stepReply.getReward();

            i++;
            lastAction = action;
        }

        //a bit of a trick usable because of how the stack is treated to init R
        INDArray input = Learning.getInput(getMdp(), obs);
        INDArray hstack = processHistory(input);

        if (hp != null) {
            hp.record(input);
        }

        if (getMdp().isDone() && i < nstep * skipFrame)
            rewards.add(new MiniTrans(hstack, null, null, 0));
        else {
            INDArray[] output = null;
            if (getConf().getTargetDqnUpdateFreq() == -1)
                output = current.outputAll(hstack);
            else synchronized (getAsyncGlobal()) {
                output = getAsyncGlobal().getTarget().outputAll(hstack);
            }
            double maxQ = Nd4j.max(output[0]).getDouble(0);
            rewards.add(new MiniTrans(hstack, null, output, maxQ));
        }

        getAsyncGlobal().enqueue(calcGradient(current, rewards), i);

        return new SubEpochReturn<O>(i, obs, reward, current.getLatestScore());
    }

    protected INDArray processHistory(INDArray input) {
        IHistoryProcessor hp = getHistoryProcessor();
        INDArray[] history;
        if (hp != null) {
            hp.add(input);
            history = hp.getHistory();
        } else
            history = new INDArray[] {input};
        //concat the history into a single INDArray input
        INDArray hstack = Transition.concat(history);
        if (hp != null) {
            hstack.muli(1.0 / hp.getScale());
        }

        if (getCurrent().isRecurrent()) {
            //flatten everything for the RNN
            hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape()), 1));
        } else {
            //if input is not 2d, you have to append that the batch is 1 length high
            if (hstack.shape().length > 2)
                hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape())));
        }

        return hstack;
    }

    public abstract Gradient[] calcGradient(NN nn, Stack<MiniTrans<Integer>> rewards);
}
