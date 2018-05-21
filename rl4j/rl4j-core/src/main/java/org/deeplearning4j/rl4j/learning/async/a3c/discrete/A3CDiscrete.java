package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import lombok.*;
import org.deeplearning4j.rl4j.learning.async.AsyncConfiguration;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 * Training for A3C in the Discrete Domain
 *
 * All methods are fully implemented as described in the
 * https://arxiv.org/abs/1602.01783 paper.
 *
 */
public abstract class A3CDiscrete<O extends Encodable> extends AsyncLearning<O, Integer, DiscreteSpace, IActorCritic> {

    @Getter
    final public A3CConfiguration configuration;
    @Getter
    final protected MDP<O, Integer, DiscreteSpace> mdp;
    final private IActorCritic iActorCritic;
    @Getter
    final private AsyncGlobal asyncGlobal;
    @Getter
    final private ACPolicy<O> policy;
    @Getter
    final private DataManager dataManager;

    public A3CDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic iActorCritic, A3CConfiguration conf,
                    DataManager dataManager) {
        super(conf);
        this.iActorCritic = iActorCritic;
        this.mdp = mdp;
        this.configuration = conf;
        this.dataManager = dataManager;
        policy = new ACPolicy<>(iActorCritic, getRandom());
        asyncGlobal = new AsyncGlobal<>(iActorCritic, conf);
        mdp.getActionSpace().setSeed(conf.getSeed());
    }


    protected AsyncThread newThread(int i) {
        return new A3CThreadDiscrete(mdp.newInstance(), asyncGlobal, getConfiguration(), i, dataManager);
    }

    public IActorCritic getNeuralNet() {
        return iActorCritic;
    }

    @Data
    @AllArgsConstructor
    @Builder
    @EqualsAndHashCode(callSuper = false)
    public static class A3CConfiguration implements AsyncConfiguration {

        int seed;
        int maxEpochStep;
        int maxStep;
        int numThread;
        int nstep;
        int updateStart;
        double rewardFactor;
        double gamma;
        double errorClamp;

        public int getTargetDqnUpdateFreq() {
            return -1;
        }

    }
}
