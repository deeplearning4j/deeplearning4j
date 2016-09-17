package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import lombok.Getter;
import org.deeplearning4j.gym.space.DiscreteSpace;
import org.deeplearning4j.gym.space.Encodable;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncThreadDiscrete;
import org.deeplearning4j.rl4j.learning.async.MiniTrans;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;

import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 *
 * Local thread as described in the https://arxiv.org/abs/1602.01783 paper.
 */
public class A3CThreadDiscrete<O extends Encodable> extends AsyncThreadDiscrete<O, IActorCritic> {

    @Getter
    final protected A3CDiscrete.A3CConfiguration conf;
    @Getter
    final protected MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    final protected AsyncGlobal<IActorCritic> asyncGlobal;
    @Getter
    final protected int threadNumber;
    @Getter
    final protected DataManager dataManager;


    public A3CThreadDiscrete(MDP<O, Integer, DiscreteSpace> mdp, AsyncGlobal<IActorCritic> asyncGlobal, A3CDiscrete.A3CConfiguration a3cc, int threadNumber, DataManager dataManager) {
        super(asyncGlobal, threadNumber);
        this.conf = a3cc;
        this.asyncGlobal = asyncGlobal;
        this.threadNumber = threadNumber;
        this.mdp = mdp;
        this.dataManager = dataManager;
    }

    @Override
    protected Policy<O, Integer> getPolicy(IActorCritic net) {
        return new ACPolicy(net, new Random(conf.getSeed()));
    }

    /**
     *  calc the gradients based on the n-step rewards
     */
    @Override
    public Gradient[] calcGradient(IActorCritic iac, Stack<MiniTrans<Integer>> rewards) {
        MiniTrans<Integer> minTrans = rewards.pop();

        int size = rewards.size();

        int[] shape = getHistoryProcessor() == null ? mdp.getObservationSpace().getShape() : getHistoryProcessor().getConf().getShape();
        int[] nshape = Learning.makeShape(size, shape);

        INDArray input = Nd4j.create(nshape);
        INDArray targets = Nd4j.create(size, 1);
        INDArray logSoftmax = Nd4j.create(size, mdp.getActionSpace().getSize());

        double r = minTrans.getReward();
        for (int i = 0; i < size; i++) {
            minTrans = rewards.pop();


            r = minTrans.getReward() + conf.getGamma() * r;
            input.putRow(i, minTrans.getObs());

            //the critic
            targets.putScalar(i, r);

            //the actor
            INDArray row = minTrans.getOutput()[1];
            double prevV = row.getDouble(minTrans.getAction());
            double expectedV =  minTrans.getOutput()[0].getDouble(0);
            double advantage = r - expectedV;
            row = row.putScalar(minTrans.getAction(), prevV + advantage);
            logSoftmax.putRow(i, row);
        }

        return iac.gradient(input, new INDArray[]{targets, logSoftmax});
    }
}