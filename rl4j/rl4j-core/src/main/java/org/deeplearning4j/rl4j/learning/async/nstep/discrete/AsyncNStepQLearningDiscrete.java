package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import lombok.*;
import org.deeplearning4j.rl4j.learning.async.AsyncConfiguration;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public abstract class AsyncNStepQLearningDiscrete<O extends Encodable>
                extends AsyncLearning<O, Integer, DiscreteSpace, IDQN> {

    @Getter
    final public AsyncNStepQLConfiguration configuration;
    @Getter
    final private MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    final private DataManager dataManager;
    @Getter
    final private AsyncGlobal<IDQN> asyncGlobal;


    public AsyncNStepQLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, AsyncNStepQLConfiguration conf,
                    DataManager dataManager) {
        super(conf);
        this.mdp = mdp;
        this.dataManager = dataManager;
        this.configuration = conf;
        this.asyncGlobal = new AsyncGlobal<>(dqn, conf);
        mdp.getActionSpace().setSeed(conf.getSeed());
    }


    public AsyncThread newThread(int i) {
        return new AsyncNStepQLearningThreadDiscrete(mdp.newInstance(), asyncGlobal, configuration, i, dataManager);
    }

    public IDQN getNeuralNet() {
        return asyncGlobal.getCurrent();
    }

    public Policy<O, Integer> getPolicy() {
        return new DQNPolicy<O>(getNeuralNet());
    }


    @Data
    @AllArgsConstructor
    @Builder
    @EqualsAndHashCode(callSuper = false)
    public static class AsyncNStepQLConfiguration implements AsyncConfiguration {

        int seed;
        int maxEpochStep;
        int maxStep;
        int numThread;
        int nstep;
        int targetDqnUpdateFreq;
        int updateStart;
        double rewardFactor;
        double gamma;
        double errorClamp;
        float minEpsilon;
        int epsilonNbStep;

    }
}
