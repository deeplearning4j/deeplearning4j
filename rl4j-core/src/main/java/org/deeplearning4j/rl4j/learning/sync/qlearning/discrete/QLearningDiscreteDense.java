package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 */
public class QLearningDiscreteDense<O extends Encodable> extends QLearningDiscrete<O> {



    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLearning.QLConfiguration conf,
                    DataManager dataManager) {
        super(mdp, dqn, conf, dataManager, conf.getEpsilonNbStep());
    }

    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory,
                    QLearning.QLConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                        dataManager);
    }

    public QLearningDiscreteDense(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdDense.Configuration netConf,
                    QLearning.QLConfiguration conf, DataManager dataManager) {
        this(mdp, new DQNFactoryStdDense(netConf), conf, dataManager);
    }

}
