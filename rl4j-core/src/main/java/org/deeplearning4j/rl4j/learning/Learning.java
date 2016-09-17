package org.deeplearning4j.rl4j.learning;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Value;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.gym.space.ActionSpace;
import org.deeplearning4j.gym.space.Encodable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;

import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/27/16.
 *
 * Useful factorisations and helper methods for class inheriting
 * ILearning.
 *
 * Big majority of training method should inherit this
 *
 */
public abstract class Learning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet> implements ILearning<O, A, AS>, NeuralNetFetchable<NN>{


    @Getter
    final private Logger logger;
    @Getter
    final private Random random;

    @Getter
    private int stepCounter = 0;
    @Getter
    private int epochCounter = 0;

    @Getter
    private IHistoryProcessor historyProcessor = null;

    public Learning(LConfiguration conf) {

        random = new Random(conf.getSeed());
        logger = LoggerFactory.getLogger(this.getClass());

    }

    public static Integer getMaxAction(INDArray vector) {
        return Nd4j.argMax(vector, Integer.MAX_VALUE).getInt(0);
    }

    public static <O extends Encodable, A, AS extends ActionSpace<A>> INDArray getInput(MDP<O, A, AS> mdp, O obs) {
        INDArray arr = Nd4j.create(obs.toArray());
        int[] shape = mdp.getObservationSpace().getShape();
        if (shape.length == 1)
            return arr;
        else
            return arr.reshape(shape);
    }

    public static <O extends Encodable, A, AS extends ActionSpace<A>> InitMdp<O> initMdp(MDP<O, A, AS> mdp, IHistoryProcessor hp) {

        O obs = mdp.reset();

        O nextO = obs;

        int step = 0;
        double reward = 0;

        boolean isHistoryProcessor = hp != null;

        int skipFrame = isHistoryProcessor ? hp.getConf().getSkipFrame() : 1;
        int requiredFrame = isHistoryProcessor ? skipFrame * (hp.getConf().getHistoryLength() - 1) : 0;

        while (step < requiredFrame) {
            INDArray input = Learning.getInput(mdp, obs);

            if (isHistoryProcessor)
                hp.record(input);

            A action = mdp.getActionSpace().noOp(); //by convention should be the NO_OP
            if (step % skipFrame == 0 && isHistoryProcessor)
                hp.add(input);

            StepReply<O> stepReply = mdp.step(action);
            reward += stepReply.getReward();
            nextO = stepReply.getObservation();

            step++;

        }

        return new InitMdp(step, nextO, reward);

    }

    public static int[] makeShape(int size, int[] shape) {
        int[] nshape = new int[shape.length + 1];
        nshape[0] = size;
        for (int i = 0; i < shape.length; i++) {
            nshape[i + 1] = shape[i];
        }
        return nshape;
    }

    protected abstract DataManager getDataManager();

    public abstract NN getNeuralNet();

    public int incrementStep() {
        return stepCounter++;
    }

    public int incrementEpoch() {
        return epochCounter++;
    }

    protected void setHistoryProcessor(HistoryProcessor.Configuration conf) {
        historyProcessor = new HistoryProcessor(conf);
    }

    public INDArray getInput(O obs) {
        return getInput(getMdp(), obs);
    }

    public InitMdp<O> initMdp() {
        return initMdp(getMdp(), getHistoryProcessor());
    }

    @AllArgsConstructor
    @Value
    public static class InitMdp<O> {
        int steps;
        O lastObs;
        double reward;
    }

}
