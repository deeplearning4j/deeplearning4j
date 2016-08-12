package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactory;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryStdConv;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/8/16.
 */
public class A3CDiscreteConv<O extends Encodable> extends A3CDiscrete<O> {

    final private HistoryProcessor.Configuration hpconf;

    public A3CDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic, HistoryProcessor.Configuration hpconf, AsyncLearning.AsyncConfiguration conf, DataManager dataManager) {
        super(mdp, IActorCritic, conf, dataManager);
        this.hpconf = hpconf;
        setHistoryProcessor(hpconf);
    }


    public A3CDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactory factory, HistoryProcessor.Configuration hpconf, AsyncLearning.AsyncConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildActorCritic(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf, dataManager);
    }

    public A3CDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactoryStdConv.Configuration netConf, HistoryProcessor.Configuration hpconf, AsyncLearning.AsyncConfiguration conf, DataManager dataManager) {
        this(mdp, new ActorCriticFactoryStdConv(netConf), hpconf, conf, dataManager);
    }

    @Override
    public AsyncThread newThread(int i) {
        AsyncThread at = super.newThread(i);
        at.setHistoryProcessor(hpconf);
        return super.newThread(i);
    }
}
