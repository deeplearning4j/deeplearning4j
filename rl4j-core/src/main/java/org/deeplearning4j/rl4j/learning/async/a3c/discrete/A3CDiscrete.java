package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import lombok.Getter;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/23/16.
 */
public abstract class A3CDiscrete<O extends Encodable> extends AsyncLearning<O, Integer, DiscreteSpace> {

    @Getter
    final protected MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    final public AsyncConfiguration configuration;
    final private IActorCritic IActorCritic;
    @Getter
    final private AsyncGlobal asyncGlobal;
    @Getter
    final private Policy<O, Integer> policy;
    @Getter
    final private DataManager dataManager;

    public A3CDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic, AsyncConfiguration conf, DataManager dataManager) {
        super(conf);
        this.IActorCritic = IActorCritic;
        this.mdp = mdp;
        this.configuration = conf;
        this.dataManager = dataManager;
        policy = new ACPolicy<>(IActorCritic, getRandom());
        asyncGlobal = new AsyncGlobal<>(IActorCritic, conf);
    }


    protected AsyncThread newThread(int i) {
        return new A3CThreadDiscrete(mdp.newInstance(), asyncGlobal, getConfiguration(), i, dataManager);
    }

    public NeuralNet getNeuralNet() {
        return IActorCritic;
    }

}
