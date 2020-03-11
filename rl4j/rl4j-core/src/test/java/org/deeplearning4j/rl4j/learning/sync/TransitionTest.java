package org.deeplearning4j.rl4j.learning.sync;

import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TransitionTest {
    @Test
    public void when_callingCtorWithoutHistory_expect_2DObservationAndNextObservation() {
        // Arrange
        double[] obs = new double[] { 1.0, 2.0, 3.0 };
        Observation observation = buildObservation(obs);

        double[] nextObs = new double[] { 10.0, 20.0, 30.0 };
        Observation nextObservation = buildObservation(nextObs);

        // Act
        Transition transition = buildTransition(observation, 123, 234.0, nextObservation);

        // Assert
        double[][] expectedObservation = new double[][] { obs };
        assertExpected(expectedObservation, transition.getObservation().getData());

        double[][] expectedNextObservation = new double[][] { nextObs };
        assertExpected(expectedNextObservation, transition.getNextObservation());

        assertEquals(123, transition.getAction());
        assertEquals(234.0, transition.getReward(), 0.0001);
    }

    @Test
    public void when_callingCtorWithHistory_expect_ObservationWithHistoryAndNextObservationWithout() {
        // Arrange
        double[][] obs = new double[][] {
                { 0.0, 1.0, 2.0 },
                { 3.0, 4.0, 5.0 },
                { 6.0, 7.0, 8.0 },
        };
        Observation observation = buildObservation(obs);

        double[][] nextObs = new double[][] {
                { 10.0, 11.0, 12.0 },
                { 0.0, 1.0, 2.0 },
                { 3.0, 4.0, 5.0 },
        };
        Observation nextObservation = buildObservation(nextObs);

        // Act
        Transition transition = buildTransition(observation, 123, 234.0, nextObservation);

        // Assert
        assertExpected(obs, transition.getObservation().getData());

        assertExpected(nextObs[0], transition.getNextObservation());

        assertEquals(123, transition.getAction());
        assertEquals(234.0, transition.getReward(), 0.0001);
    }

    @Test
    public void when_CallingBuildStackedObservationsAndShapeRankIs2_expect_2DResultWithObservationsStackedOnDimension0() {
        // Arrange
        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>();

        double[] obs1 = new double[] { 0.0, 1.0, 2.0 };
        Observation observation1 = buildObservation(obs1);
        Observation nextObservation1 = buildObservation(new double[] { 100.0, 101.0, 102.0 });
        transitions.add(buildTransition(observation1,0, 0.0, nextObservation1));

        double[] obs2 = new double[] { 10.0, 11.0, 12.0 };
        Observation observation2 = buildObservation(obs2);
        Observation nextObservation2 = buildObservation(new double[] { 110.0, 111.0, 112.0 });
        transitions.add(buildTransition(observation2, 0, 0.0, nextObservation2));

        // Act
        INDArray result = Transition.buildStackedObservations(transitions);

        // Assert
        double[][] expected = new double[][] { obs1, obs2 };
        assertExpected(expected, result);
    }

    @Test
    public void when_CallingBuildStackedObservationsAndShapeRankIsGreaterThan2_expect_ResultWithOneMoreDimensionAndObservationsStackedOnDimension0() {
        // Arrange
        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>();

        double[][] obs1 = new double[][] {
                { 0.0, 1.0, 2.0 },
                { 3.0, 4.0, 5.0 },
                { 6.0, 7.0, 8.0 },
        };
        Observation observation1 = buildObservation(obs1);

        double[] nextObs1 = new double[] { 100.0, 101.0, 102.0 };
        Observation nextObservation1 = buildNextObservation(obs1, nextObs1);

        transitions.add(buildTransition(observation1, 0, 0.0, nextObservation1));

        double[][] obs2 = new double[][] {
                { 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0 },
                { 16.0, 17.0, 18.0 },
        };
        Observation observation2 = buildObservation(obs2);

        double[] nextObs2 = new double[] { 110.0, 111.0, 112.0 };
        Observation nextObservation2 = buildNextObservation(obs2, nextObs2);
        transitions.add(buildTransition(observation2, 0, 0.0, nextObservation2));

        // Act
        INDArray result = Transition.buildStackedObservations(transitions);

        // Assert
        double[][][] expected = new double[][][] { obs1, obs2 };
        assertExpected(expected, result);
    }

    @Test
    public void when_CallingBuildStackedNextObservationsAndShapeRankIs2_expect_2DResultWithObservationsStackedOnDimension0() {
        // Arrange
        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>();

        double[] obs1 = new double[] { 0.0, 1.0, 2.0 };
        double[] nextObs1 = new double[] { 100.0, 101.0, 102.0 };
        Observation observation1 = buildObservation(obs1);
        Observation nextObservation1 = buildObservation(nextObs1);
        transitions.add(buildTransition(observation1, 0, 0.0, nextObservation1));

        double[] obs2 = new double[] { 10.0, 11.0, 12.0 };
        double[] nextObs2 = new double[] { 110.0, 111.0, 112.0 };
        Observation observation2 = buildObservation(obs2);
        Observation nextObservation2 = buildObservation(nextObs2);
        transitions.add(buildTransition(observation2, 0, 0.0, nextObservation2));

        // Act
        INDArray result = Transition.buildStackedNextObservations(transitions);

        // Assert
        double[][] expected = new double[][] { nextObs1, nextObs2 };
        assertExpected(expected, result);
    }

    @Test
    public void when_CallingBuildStackedNextObservationsAndShapeRankIsGreaterThan2_expect_ResultWithOneMoreDimensionAndObservationsStackedOnDimension0() {
        // Arrange
        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>();

        double[][] obs1 = new double[][] {
                { 0.0, 1.0, 2.0 },
                { 3.0, 4.0, 5.0 },
                { 6.0, 7.0, 8.0 },
        };
        Observation observation1 = buildObservation(obs1);

        double[] nextObs1 = new double[] { 100.0, 101.0, 102.0 };
        Observation nextObservation1 = buildNextObservation(obs1, nextObs1);

        transitions.add(buildTransition(observation1, 0, 0.0, nextObservation1));

        double[][] obs2 = new double[][] {
                { 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0 },
                { 16.0, 17.0, 18.0 },
        };
        Observation observation2 = buildObservation(obs2);

        double[] nextObs2 = new double[] { 110.0, 111.0, 112.0 };
        Observation nextObservation2 = buildNextObservation(obs2, nextObs2);

        transitions.add(buildTransition(observation2, 0, 0.0, nextObservation2));

        // Act
        INDArray result = Transition.buildStackedNextObservations(transitions);

        // Assert
        double[][][] expected = new double[][][] {
                new double[][] { nextObs1, obs1[0], obs1[1] },
                new double[][] { nextObs2, obs2[0], obs2[1] }
        };
        assertExpected(expected, result);
    }

    private Observation buildObservation(double[][] obs) {
        INDArray[] history = new INDArray[] {
                Nd4j.create(obs[0]).reshape(1, 3),
                Nd4j.create(obs[1]).reshape(1, 3),
                Nd4j.create(obs[2]).reshape(1, 3),
        };
        return new Observation(Nd4j.concat(0, history));
    }

    private Observation buildObservation(double[] obs) {
        return new Observation(Nd4j.create(obs).reshape(1, 3));
    }

    private Observation buildNextObservation(double[][] obs, double[] nextObs) {
        INDArray[] nextHistory = new INDArray[] {
                Nd4j.create(nextObs).reshape(1, 3),
                Nd4j.create(obs[0]).reshape(1, 3),
                Nd4j.create(obs[1]).reshape(1, 3),
        };
        return new Observation(Nd4j.concat(0, nextHistory));
    }

    private Transition buildTransition(Observation observation, int action, double reward, Observation nextObservation) {
        Transition result = new Transition(observation, action, reward, false);
        result.setNextObservation(nextObservation);

        return result;
    }

    private void assertExpected(double[] expected, INDArray actual) {
        long[] shape = actual.shape();
        assertEquals(2, shape.length);
        assertEquals(1, shape[0]);
        assertEquals(expected.length, shape[1]);
        for(int i = 0; i < expected.length; ++i) {
            assertEquals(expected[i], actual.getDouble(0, i), 0.0001);
        }
    }

    private void assertExpected(double[][] expected, INDArray actual) {
        long[] shape = actual.shape();
        assertEquals(2, shape.length);
        assertEquals(expected.length, shape[0]);
        assertEquals(expected[0].length, shape[1]);

        for(int i = 0; i < expected.length; ++i) {
            double[] expectedLine = expected[i];
            for(int j = 0; j < expectedLine.length; ++j) {
                assertEquals(expectedLine[j], actual.getDouble(i, j), 0.0001);
            }
        }
    }

    private void assertExpected(double[][][] expected, INDArray actual) {
        long[] shape = actual.shape();
        assertEquals(3, shape.length);
        assertEquals(expected.length, shape[0]);
        assertEquals(expected[0].length, shape[1]);
        assertEquals(expected[0][0].length, shape[2]);

        for(int i = 0; i < expected.length; ++i) {
            double[][] expected2D = expected[i];
            for(int j = 0; j < expected2D.length; ++j) {
                double[] expectedLine = expected2D[j];
                for (int k = 0; k < expectedLine.length; ++k) {
                    assertEquals(expectedLine[k], actual.getDouble(i, j, k), 0.0001);
                }
            }
        }

    }
}
