package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCritic;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 */
public abstract class A3CDiscrete<O extends Encodable> extends AsyncLearning<O, Integer, DiscreteSpace, IActorCritic> {

    @Getter
    final public AsyncConfiguration configuration;
    @Getter
    final protected MDP<O, Integer, DiscreteSpace> mdp;
    final private IActorCritic iActorCritic;
    @Getter
    final private AsyncGlobal asyncGlobal;
    @Getter
    final private Policy<O, Integer> policy;
    @Getter
    final private DataManager dataManager;

    public A3CDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic iActorCritic, AsyncConfiguration conf, DataManager dataManager) {
        super(conf);
        this.iActorCritic = iActorCritic;
        this.mdp = mdp;
        this.configuration = conf;
        this.dataManager = dataManager;
        policy = new ACPolicy<>(iActorCritic, getRandom());
        asyncGlobal = new AsyncGlobal<>(iActorCritic, conf);
    }


    protected AsyncThread newThread(int i) {
        return new A3CThreadDiscrete(mdp.newInstance(), asyncGlobal, getConfiguration(), i, dataManager);
    }

    public IActorCritic getNeuralNet() {
        return iActorCritic;
    }

}
