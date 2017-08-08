package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/8/16.
 *
 * Training for A3C in the Discrete Domain
 *
 * Specialized constructors for the Conv (pixels input) case
 * Specialized conf + provide additional type safety
 *
 * It uses CompGraph because there is benefit to combine the
 * first layers since they're essentially doing the same dimension
 * reduction task
 *
 **/
public class A3CDiscreteConv<O extends Encodable> extends A3CDiscrete<O> {

    final private HistoryProcessor.Configuration hpconf;

    public A3CDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic,
                    HistoryProcessor.Configuration hpconf, A3CConfiguration conf, DataManager dataManager) {
        super(mdp, IActorCritic, conf, dataManager);
        this.hpconf = hpconf;
        setHistoryProcessor(hpconf);
    }


    public A3CDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                    HistoryProcessor.Configuration hpconf, A3CConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildActorCritic(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf,
                        dataManager);
    }

    public A3CDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraphStdConv.Configuration netConf,
                    HistoryProcessor.Configuration hpconf, A3CConfiguration conf, DataManager dataManager) {
        this(mdp, new ActorCriticFactoryCompGraphStdConv(netConf), hpconf, conf, dataManager);
    }

    @Override
    public AsyncThread newThread(int i) {
        AsyncThread at = super.newThread(i);
        at.setHistoryProcessor(hpconf);
        return at;
    }
}
