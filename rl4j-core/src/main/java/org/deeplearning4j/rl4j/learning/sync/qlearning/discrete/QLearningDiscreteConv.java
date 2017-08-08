package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 * Specialized constructors for the Conv (pixels input) case
 * Specialized conf + provide additional type safety
 */
public class QLearningDiscreteConv<O extends Encodable> extends QLearningDiscrete<O> {



    public QLearningDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, HistoryProcessor.Configuration hpconf,
                    QLConfiguration conf, DataManager dataManager) {
        super(mdp, dqn, conf, dataManager, conf.getEpsilonNbStep() * hpconf.getSkipFrame());
        setHistoryProcessor(hpconf);
    }

    public QLearningDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory,
                    HistoryProcessor.Configuration hpconf, QLConfiguration conf, DataManager dataManager) {
        this(mdp, factory.buildDQN(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf, dataManager);
    }

    public QLearningDiscreteConv(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdConv.Configuration netConf,
                    HistoryProcessor.Configuration hpconf, QLConfiguration conf, DataManager dataManager) {
        this(mdp, new DQNFactoryStdConv(netConf), hpconf, conf, dataManager);
    }
}
