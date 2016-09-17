package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Async Learning specialized for the Discrete Domain
 *
 */
public abstract class AsyncThreadDiscrete<O extends Encodable, NN extends NeuralNet> extends AsyncThread<O, Integer, DiscreteSpace, NN> {

    public AsyncThreadDiscrete(AsyncGlobal<NN> asyncGlobal, int threadNumber){
        super(asyncGlobal, threadNumber);
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

        NN current = getAsyncGlobal().cloneCurrent();
        Stack<MiniTrans<Integer>> rewards = new Stack<>();

        O obs = sObs;
        Policy<O, Integer> policy = getPolicy(current);

        Integer action;
        Integer lastAction = null;
        INDArray history[] = null;
        boolean isHistoryProcessor = getHistoryProcessor() != null;

        int skipFrame = isHistoryProcessor ? getHistoryProcessor().getConf().getSkipFrame() : 1;

        double reward = 0;
        double accuReward = 0;
        int i = 0;
        while (!getMdp().isDone() && i < nstep*skipFrame) {

            INDArray input = Learning.getInput(getMdp(), obs);

            //if step of training, just repeat lastAction
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
                //concat the history into a single INDArray input
                INDArray hstack = Transition.concat(history);

                //if input is not 2d, you have to append that the batch is 1 length high
                if (hstack.shape().length > 2)
                    hstack = hstack.reshape(Learning.makeShape(1, hstack.shape()));

                action = policy.nextAction(hstack);
            }

            lastAction = action;

            StepReply<O> stepReply = getMdp().step(action);
            accuReward += stepReply.getReward() * getConf().getRewardFactor();

            //if it's not a skipped frame, you can do a step of training
            if (getStepCounter() % skipFrame == 0 || stepReply.isDone()) {
                obs = stepReply.getObservation();

                if (input.shape().length > 2)
                    input = input.reshape(Learning.makeShape(1, input.shape()));

                INDArray[] output = current.outputAll(input);
                rewards.add(new MiniTrans(Transition.concat(history), action, output, accuReward));

                reward += stepReply.getReward();

                if (isHistoryProcessor)
                    getHistoryProcessor().add(Learning.getInput(getMdp(), stepReply.getObservation()));

                history = isHistoryProcessor ? getHistoryProcessor().getHistory() : new INDArray[]{Learning.getInput(getMdp(), stepReply.getObservation())};
                accuReward = 0;
            }
            i++;
        }

        //a bit of a trick usable because of how the stack is treated to init R
        INDArray input = Learning.getInput(getMdp(), obs);
        if (getMdp().isDone())
            rewards.add(new MiniTrans(input, null, null, 0));
        else {
            INDArray[] output = null;
            if (getConf().getTargetDqnUpdateFreq() == -1)
                output = current.outputAll(input);
            else
                output = getAsyncGlobal().cloneTarget().outputAll(input);
            double maxQ = Nd4j.max(output[0]).getDouble(0);
            rewards.add(new MiniTrans(input, null, output, maxQ));
        }

        getAsyncGlobal().enqueue(calcGradient(current, rewards), i);

        return new SubEpochReturn<O>(i, obs, reward, current.getLatestScore());
    }

    ;

    public abstract Gradient[] calcGradient(NN nn, Stack<MiniTrans<Integer>> rewards);
}
