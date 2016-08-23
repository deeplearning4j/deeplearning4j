package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactory;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryStdDense;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/8/16.
 */
public class A3CDiscreteDense<O extends Encodable> extends A3CDiscrete<O> {

    public A3CDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic, A3CConfiguration conf, DataManager dataManager) {
        super(mdp, IActorCritic, conf, dataManager);
    }

    public A3CDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactory factory, A3CConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildActorCritic(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf, dataManager);
    }

    public A3CDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactoryStdDense.Configuration netConf, A3CConfiguration conf, DataManager dataManager) {
        this(mdp, new ActorCriticFactoryStdDense(netConf), conf, dataManager);
    }

}
