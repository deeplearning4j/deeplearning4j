package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import lombok.Getter;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThreadDiscrete;
import org.deeplearning4j.rl4j.learning.async.MiniTrans;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.Stack;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 */
public class A3CThreadDiscrete<O extends Encodable> extends AsyncThreadDiscrete<O, IActorCritic> {

    @Getter
    final protected AsyncLearning.AsyncConfiguration conf;
    @Getter
    final protected MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    final protected AsyncGlobal<IActorCritic> asyncGlobal;
    @Getter
    final protected int threadNumber;
    @Getter
    final protected DataManager dataManager;


    public A3CThreadDiscrete(MDP<O, Integer, DiscreteSpace> mdp, AsyncGlobal<IActorCritic> asyncGlobal, AsyncLearning.AsyncConfiguration a3cc, int threadNumber, DataManager dataManager) {
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

    //FIXME double DQN ? (Not present in the original paper tho)
    @Override
    public Gradient calcGradient(Stack<MiniTrans<Integer>> rewards) {
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

            targets.putScalar(i, r);

            INDArray row = Nd4j.create(1, mdp.getActionSpace().getSize());
            row = row.putScalar(minTrans.getAction(), r - minTrans.getOutput()[0].getDouble(0));
            logSoftmax.putRow(i, row);
        }

        return nn.gradient(input, new INDArray[]{targets, logSoftmax});
    }
}