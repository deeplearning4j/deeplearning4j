package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.TDTargetAlgorithm;

import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.support.MockDQN;
import org.deeplearning4j.rl4j.learning.sync.support.MockTargetQNetworkSource;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class StandardDQNTest {
    @Test
    public void when_isTerminal_expect_rewardValueAtIdx0() {

        // Assemble
        MockDQN qNetwork = new MockDQN();
        MockDQN targetQNetwork = new MockDQN();
        MockTargetQNetworkSource targetQNetworkSource = new MockTargetQNetworkSource(qNetwork, targetQNetwork);

        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>() {
            {
                add(new Transition<Integer>(new INDArray[]{Nd4j.create(new double[]{1.1, 2.2})}, 0, 1.0, true, Nd4j.create(new double[]{11.0, 22.0})));
            }
        };

        StandardDQN sut = new StandardDQN(targetQNetworkSource, 0.5);
        sut.setNShape(new int[] { 1, 2 });

        // Act
        DataSet result = sut.computeTDTargets(transitions);

        // Assert
        INDArray evaluatedQValues = result.getLabels();
        assertEquals(1.0, evaluatedQValues.getDouble(0, 0), 0.0001);
        assertEquals(2.2, evaluatedQValues.getDouble(0, 1), 0.0001);
    }

    @Test
    public void when_isNotTerminal_expect_rewardPlusEstimatedQValue() {

        // Assemble
        MockDQN qNetwork = new MockDQN();
        MockDQN targetQNetwork = new MockDQN();
        MockTargetQNetworkSource targetQNetworkSource = new MockTargetQNetworkSource(qNetwork, targetQNetwork);

        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>() {
            {
                add(new Transition<Integer>(new INDArray[]{Nd4j.create(new double[]{1.1, 2.2})}, 0, 1.0, false, Nd4j.create(new double[]{11.0, 22.0})));
            }
        };

        StandardDQN sut = new StandardDQN(targetQNetworkSource, 0.5);
        sut.setNShape(new int[] { 1, 2 });

        // Act
        DataSet result = sut.computeTDTargets(transitions);

        // Assert
        INDArray evaluatedQValues = result.getLabels();
        assertEquals(1.0 + 0.5 * 22.0, evaluatedQValues.getDouble(0, 0), 0.0001);
        assertEquals(2.2, evaluatedQValues.getDouble(0, 1), 0.0001);
    }

    @Test
    public void when_batchHasMoreThanOne_expect_everySampleEvaluated() {

        // Assemble
        MockDQN qNetwork = new MockDQN();
        MockDQN targetQNetwork = new MockDQN();
        MockTargetQNetworkSource targetQNetworkSource = new MockTargetQNetworkSource(qNetwork, targetQNetwork);

        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>() {
            {
                add(new Transition<Integer>(new INDArray[]{Nd4j.create(new double[]{1.1, 2.2})}, 0, 1.0, false, Nd4j.create(new double[]{11.0, 22.0})));
                add(new Transition<Integer>(new INDArray[]{Nd4j.create(new double[]{3.3, 4.4})}, 1, 2.0, false, Nd4j.create(new double[]{33.0, 44.0})));
                add(new Transition<Integer>(new INDArray[]{Nd4j.create(new double[]{5.5, 6.6})}, 0, 3.0, true, Nd4j.create(new double[]{55.0, 66.0})));
            }
        };

        StandardDQN sut = new StandardDQN(targetQNetworkSource, 0.5);
        sut.setNShape(new int[] { 3, 2 });

        // Act
        DataSet result = sut.computeTDTargets(transitions);

        // Assert
        INDArray evaluatedQValues = result.getLabels();
        assertEquals((1.0 + 0.5 * 22.0), evaluatedQValues.getDouble(0, 0), 0.0001);
        assertEquals(2.2, evaluatedQValues.getDouble(0, 1), 0.0001);

        assertEquals(3.3, evaluatedQValues.getDouble(1, 0), 0.0001);
        assertEquals((2.0 + 0.5 * 44.0), evaluatedQValues.getDouble(1, 1), 0.0001);

        assertEquals(3.0, evaluatedQValues.getDouble(2, 0), 0.0001); // terminal: reward only
        assertEquals(6.6, evaluatedQValues.getDouble(2, 1), 0.0001);

    }

}
