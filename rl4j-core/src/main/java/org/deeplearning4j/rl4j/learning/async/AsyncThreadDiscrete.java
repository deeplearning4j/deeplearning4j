package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public abstract class AsyncThreadDiscrete<O extends Encodable, NN extends NeuralNet> extends AsyncThread<O, Integer, DiscreteSpace, NN> {


    public SubEpochReturn<O> trainSubEpoch(O sObs, int nstep) {

        nn = getAsyncGlobal().cloneCurrent();
        Stack<MiniTrans<Integer>> rewards = new Stack<>();

        O obs = sObs;
        Policy<O, Integer> policy = getPolicy(nn);

        Integer action;
        Integer lastAction = null;
        INDArray history[] = null;
        boolean isHistoryProcessor = getHistoryProcessor() != null;
        NN target = getAsyncGlobal().getTarget();

        int skipFrame = isHistoryProcessor ? getHistoryProcessor().getConf().getSkipFrame() : 1;

        double reward = 0;

        int i = 0;
        while (!getMdp().isDone() && i < nstep) {

            INDArray input = Learning.getInput(getMdp(), obs);

            if (getStepCounter() % skipFrame != 0) {
                action = lastAction;
            } else {
                if (history == null) {
                    if (isHistoryProcessor) {
                        getHistoryProcessor().add(input);
                        history = getHistoryProcessor().getHistory();
                    } else
                        history = new INDArray[]{input};
                }
                INDArray hstack = Transition.concat(history);
                if (hstack.shape().length > 2)
                    hstack = hstack.reshape(Learning.makeShape(1, hstack.shape()));
                action = policy.nextAction(hstack);
            }
            lastAction = action;

            StepReply<O> stepReply = getMdp().step(action);
            obs = stepReply.getObservation();

            INDArray[] output = target.outputAll(input.reshape(Learning.makeShape(1, input.shape())));
            rewards.add(new MiniTrans(Transition.concat(history), action, output, stepReply.getReward()));
            reward += stepReply.getReward();

            if (isHistoryProcessor)
                getHistoryProcessor().add(Learning.getInput(getMdp(), stepReply.getObservation()));

            history = isHistoryProcessor ? getHistoryProcessor().getHistory() : new INDArray[]{Learning.getInput(getMdp(), stepReply.getObservation())};

            i++;
        }

        //a bit of a trick usable because of how the stack is treated to init R
        INDArray input = Learning.getInput(getMdp(), obs);
        if (getMdp().isDone())
            rewards.add(new MiniTrans(input, null, null, 0));
        else {
            INDArray[] output = target.outputAll(input);
            double maxQ = Nd4j.max(output[0]).getDouble(0);
            rewards.add(new MiniTrans(input, null, output, maxQ));
        }
        if (rewards.size() > 1)
            getAsyncGlobal().enqueue(calcGradient(rewards), i);
        else
            log.info("not long enough");

        //log.info("Sent an update");
        return new SubEpochReturn<O>(i, obs, reward);
    }

    ;

    public abstract Gradient calcGradient(Stack<MiniTrans<Integer>> rewards);
}
