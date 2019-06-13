package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

public class HistoryProcessorLearningInitializer<O  extends Encodable, A, AS extends ActionSpace<A>> implements ILearningInitializer<O, A, AS> {

    private final IHistoryProcessor historyProcessor;

    public HistoryProcessorLearningInitializer(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;
    }

    @Override
    public Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp) {

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
}
