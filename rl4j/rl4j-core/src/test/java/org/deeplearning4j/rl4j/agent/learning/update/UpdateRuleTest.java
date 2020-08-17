package org.deeplearning4j.rl4j.agent.learning.update;

import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class UpdateRuleTest {

    @Mock
    private IUpdateAlgorithm<FeaturesLabels, Integer> updateAlgorithm;

    @Mock
    private INeuralNetUpdater<FeaturesLabels> updater;

    private UpdateRule<FeaturesLabels, Integer> sut;

    @Before
    public void init() {
        sut = new UpdateRule<FeaturesLabels, Integer>(updateAlgorithm, updater);
    }

    @Test
    public void when_callingUpdate_expect_computeAndUpdateNetwork() {
        // Arrange
        List<Integer> trainingBatch = new ArrayList<Integer>() {
            {
                Integer.valueOf(1);
                Integer.valueOf(2);
            }
        };
        final FeaturesLabels computeResult = new FeaturesLabels(null);
        when(updateAlgorithm.compute(any())).thenReturn(computeResult);

        // Act
        sut.update(trainingBatch);

        // Assert
        verify(updateAlgorithm, times(1)).compute(trainingBatch);
        verify(updater, times(1)).update(computeResult);
    }

    @Test
    public void when_callingUpdate_expect_updateCountIncremented() {
        // Arrange

        // Act
        sut.update(null);
        int updateCountBefore = sut.getUpdateCount();
        sut.update(null);
        int updateCountAfter = sut.getUpdateCount();

        // Assert
        assertEquals(updateCountBefore + 1, updateCountAfter);
    }

}
