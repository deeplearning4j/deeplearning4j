package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.support.*;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class QLearningDiscreteTest {
    @Test
    public void refac_QLearningDiscrete_trainStep() {
        // Arrange
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockMDP mdp = new MockMDP(observationSpace);
        MockDQN dqn = new MockDQN();
        QLearning.QLConfiguration conf = new QLearning.QLConfiguration(0, 0, 0, 5, 1, 0,
                0, 1.0, 0, 0, 0, 0, true);
        MockDataManager dataManager = new MockDataManager(false);
        MockExpReplay expReplay = new MockExpReplay();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, dqn, conf, dataManager, expReplay, 10);
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 4, 4, 4, 4, 0, 0, 2);
        MockHistoryProcessor hp = new MockHistoryProcessor(hpConf);
        sut.setHistoryProcessor(hp);
        MockEncodable obs = new MockEncodable(1);
        List<QLearning.QLStepReturn<MockEncodable>> results = new ArrayList<>();

        // Act
        sut.initMdp();
        for(int step = 0; step < 16; ++step) {
            results.add(sut.trainStep(obs));
            sut.incrementStep();
        }

        // Assert
        // HistoryProcessor calls
        assertEquals(24, hp.recordCallCount);
        assertEquals(13, hp.addCallCount);
        assertEquals(0, hp.startMonitorCallCount);
        assertEquals(0, hp.stopMonitorCallCount);

        // DQN calls
        assertEquals(1, dqn.fitParams.size());
        assertEquals(123.0, dqn.fitParams.get(0).getFirst().getDouble(0), 0.001);
        assertEquals(234.0, dqn.fitParams.get(0).getSecond().getDouble(0), 0.001);
        assertEquals(14, dqn.outputParams.size());
        double[][] expectedDQNOutput = new double[][] {
                new double[] { 0.0, 0.0, 0.0, 0.0, 1.0 },
                new double[] { 0.0, 0.0, 0.0, 1.0, 9.0 },
                new double[] { 0.0, 0.0, 0.0, 1.0, 9.0 },
                new double[] { 0.0, 0.0, 1.0, 9.0, 11.0 },
                new double[] { 0.0, 1.0, 9.0, 11.0, 13.0 },
                new double[] { 0.0, 1.0, 9.0, 11.0, 13.0 },
                new double[] { 1.0, 9.0, 11.0, 13.0, 15.0 },
                new double[] { 1.0, 9.0, 11.0, 13.0, 15.0 },
                new double[] { 9.0, 11.0, 13.0, 15.0, 17.0 },
                new double[] { 9.0, 11.0, 13.0, 15.0, 17.0 },
                new double[] { 11.0, 13.0, 15.0, 17.0, 19.0 },
                new double[] { 11.0, 13.0, 15.0, 17.0, 19.0 },
                new double[] { 13.0, 15.0, 17.0, 19.0, 21.0 },
                new double[] { 13.0, 15.0, 17.0, 19.0, 21.0 },

        };
        for(int i = 0; i < expectedDQNOutput.length; ++i) {
            INDArray outputParam = dqn.outputParams.get(i);

            assertEquals(5, outputParam.shape()[0]);
            assertEquals(1, outputParam.shape()[1]);

            double[] expectedRow = expectedDQNOutput[i];
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j] / 255.0, outputParam.getDouble(j), 0.00001);
            }
        }

        // MDP calls
        assertArrayEquals(new Integer[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 }, mdp.actions.toArray());

        // ExpReplay calls
        double[] expectedTrRewards = new double[] { 9.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0 };
        int[] expectedTrActions = new int[] { 0, 4, 3, 4, 4, 4, 4, 4 };
        double[] expectedTrNextObservation = new double[] { 0, 0, 0, 1.0, 9.0, 11.0, 13.0, 15.0 };
        double[][] expectedTrObservations = new double[][] {
                new double[] { 0.0, 0.0, 0.0, 0.0, 1.0 },
                new double[] { 0.0, 0.0, 0.0, 1.0, 9.0 },
                new double[] { 0.0, 0.0, 1.0, 9.0, 11.0 },
                new double[] { 0.0, 1.0, 9.0, 11.0, 13.0 },
                new double[] { 1.0, 9.0, 11.0, 13.0, 15.0 },
                new double[] { 9.0, 11.0, 13.0, 15.0, 17.0 },
                new double[] { 11.0, 13.0, 15.0, 17.0, 19.0 },
                new double[] { 13.0, 15.0, 17.0, 19.0, 21.0 },
        };
        for(int i = 0; i < expectedTrRewards.length; ++i) {
            Transition tr = expReplay.transitions.get(i);
            assertEquals(expectedTrRewards[i], tr.getReward(), 0.0001);
            assertEquals(expectedTrActions[i], tr.getAction());
            assertEquals(expectedTrNextObservation[i], tr.getNextObservation().getDouble(0), 0.0001);
            for(int j = 0; j < expectedTrObservations[i].length; ++j) {
                assertEquals(expectedTrObservations[i][j], tr.getObservation()[j].getDouble(0), 0.0001);
            }
        }

        // trainStep results
        assertEquals(16, results.size());
        double[] expectedMaxQ = new double[] { 1.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0 };
        double[] expectedRewards = new double[] { 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0 };
        for(int i=0; i < 16; ++i) {
            QLearning.QLStepReturn<MockEncodable> result = results.get(i);
            if(i % 2 == 0) {
                assertEquals(expectedMaxQ[i/2] / 255.0, result.getMaxQ(), 0.001);
                assertEquals(expectedRewards[i/2], result.getStepReply().getReward(), 0.001);
            }
            else {
                assertTrue(result.getMaxQ().isNaN());
            }
        }
    }

    public static class TestQLearningDiscrete extends QLearningDiscrete<MockEncodable> {
        public TestQLearningDiscrete(MDP<MockEncodable, Integer, DiscreteSpace> mdp,IDQN dqn,
                                     QLConfiguration conf, IDataManager dataManager, MockExpReplay expReplay,
                                     int epsilonNbStep) {
            super(mdp, dqn, conf, epsilonNbStep);
            addListener(new DataManagerTrainingListener(dataManager));
            setExpReplay(expReplay);
        }

        @Override
        protected Pair<INDArray, INDArray> setTarget(ArrayList<Transition<Integer>> transitions) {
            return new Pair<>(Nd4j.create(new double[] { 123.0 }), Nd4j.create(new double[] { 234.0 }));
        }

        public void setExpReplay(IExpReplay<Integer> exp){
            this.expReplay = exp;
        }

    }
}
