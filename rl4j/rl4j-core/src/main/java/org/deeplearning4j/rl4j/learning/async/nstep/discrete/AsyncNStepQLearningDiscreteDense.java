package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/7/16.
 */
public class AsyncNStepQLearningDiscreteDense<O extends Encodable> extends AsyncNStepQLearningDiscrete<O> {

    public AsyncNStepQLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn,
                    AsyncNStepQLConfiguration conf, DataManager dataManager) {
        super(mdp, dqn, conf, dataManager);
    }

    public AsyncNStepQLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory,
                    AsyncNStepQLConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                        dataManager);
    }

    public AsyncNStepQLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp,
                    DQNFactoryStdDense.Configuration netConf, AsyncNStepQLConfiguration conf, DataManager dataManager) {
        this(mdp, new DQNFactoryStdDense(netConf), conf, dataManager);
    }
}
