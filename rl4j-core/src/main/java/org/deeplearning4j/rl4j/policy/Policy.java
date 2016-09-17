package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.mdp.MDP;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 * Abstract class common to all policies
 *
 * A Policy responsability is to choose the next action given a state
 */
public abstract class Policy<O extends Encodable, A> {

    public abstract A nextAction(INDArray input);

    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp) {
        return play(mdp, null);
    }

    public <AS extends ActionSpace<A>> double play(MDP<O, A, AS> mdp, IHistoryProcessor hp) {

        Learning.InitMdp<O> initMdp = Learning.initMdp(mdp, hp);
        O obs = initMdp.getLastObs();

        double reward = initMdp.getReward();

        A lastAction = mdp.getActionSpace().noOp();
        A action;
        int step = initMdp.getSteps();
        INDArray[] history = null;

        while (!mdp.isDone()) {

            INDArray input = Learning.getInput(mdp, obs);
            boolean isHistoryProcessor = hp != null;

            if (isHistoryProcessor)
                hp.record(input);

            int skipFrame = isHistoryProcessor ? hp.getConf().getSkipFrame() : 1;


            if (step % skipFrame != 0) {
                action = lastAction;
            } else {

                if (history == null) {
                    if (isHistoryProcessor) {
                        hp.add(input);
                        history = hp.getHistory();
                    } else
                        history = new INDArray[]{input};
                }
                INDArray hstack = Transition.concat(history);
                if (hstack.shape().length > 2)
                    hstack = hstack.reshape(Learning.makeShape(1, hstack.shape()));
                action = nextAction(hstack);
            }
            lastAction = action;

            StepReply<O> stepReply = mdp.step(action);
            reward += stepReply.getReward();

            if (isHistoryProcessor)
                hp.add(Learning.getInput(mdp, stepReply.getObservation()));

            history = isHistoryProcessor ? hp.getHistory() : new INDArray[]{Learning.getInput(mdp, stepReply.getObservation())};
            step++;
        }


        return reward;
    }

}
