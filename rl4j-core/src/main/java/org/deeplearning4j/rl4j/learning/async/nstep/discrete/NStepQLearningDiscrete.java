package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import lombok.Getter;
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
public abstract class NStepQLearningDiscrete<O extends Encodable> extends AsyncLearning<O, Integer, DiscreteSpace> {

    @Getter
    final public AsyncConfiguration configuration;
    @Getter
    final private MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    final private DataManager dataManager;
    @Getter
    final private AsyncGlobal<IDQN> asyncGlobal;


    public NStepQLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, AsyncConfiguration conf, DataManager dataManager) {
        super(conf);
        this.mdp = mdp;
        this.dataManager = dataManager;
        this.configuration = conf;
        this.asyncGlobal = new AsyncGlobal<>(dqn, conf);
    }


    public AsyncThread newThread(int i) {
        return new NStepQLearningThreadDiscrete(mdp, asyncGlobal, configuration, i, dataManager);
    }

    public IDQN getNeuralNet() {
        return asyncGlobal.getTarget();
    }

    public Policy<O, Integer> getPolicy() {
        return new DQNPolicy<O>(getNeuralNet());
    }
}
