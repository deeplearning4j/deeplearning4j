/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class CompoundNetworkHandlerTest {

    @Mock
    INetworkHandler handler1;

    @Mock
    INetworkHandler handler2;

    private CompoundNetworkHandler sut;

    public void setup(boolean setupRecurrent) {
        when(handler1.isRecurrent()).thenReturn(setupRecurrent);
        when(handler2.isRecurrent()).thenReturn(false);

        sut = new CompoundNetworkHandler(handler1, handler2);
    }

    @Test
    public void when_callingNotifyGradientCalculation_expect_listenersNotified() {
        // Arrange
        setup(false);

        // Act
        sut.notifyGradientCalculation();

        // Assert
        verify(handler1, times(1)).notifyGradientCalculation();
        verify(handler2, times(1)).notifyGradientCalculation();
    }

    @Test
    public void when_callingNotifyIterationDone_expect_listenersNotified() {
        // Arrange
        setup(false);

        // Act
        sut.notifyIterationDone();

        // Assert
        verify(handler1, times(1)).notifyIterationDone();
        verify(handler2, times(1)).notifyIterationDone();
    }

    @Test
    public void when_callingPerformFit_expect_performFitIsCalledOnHandlders() {
        // Arrange
        setup(false);
        FeaturesLabels featuresLabels = new FeaturesLabels(null);

        // Act
        sut.performFit(featuresLabels);

        // Assert
        verify(handler1, times(1)).performFit(featuresLabels);
        verify(handler2, times(1)).performFit(featuresLabels);
    }

    @Test
    public void when_callingPerformGradientsComputation_expect_performGradientsComputationIsCalledOnHandlers() {
        // Arrange
        setup(false);
        FeaturesLabels featuresLabels = new FeaturesLabels(null);

        // Act
        sut.performGradientsComputation(featuresLabels);

        // Assert
        verify(handler1, times(1)).performGradientsComputation(featuresLabels);
        verify(handler2, times(1)).performGradientsComputation(featuresLabels);
    }

    @Test
    public void when_callingFillGradientsResponse_expect_fillGradientsResponseIsCalledOnHandlers() {
        // Arrange
        setup(false);
        Gradients gradientsMock = mock(Gradients.class);

        // Act
        sut.fillGradientsResponse(gradientsMock);

        // Assert
        verify(handler1, times(1)).fillGradientsResponse(gradientsMock);
        verify(handler2, times(1)).fillGradientsResponse(gradientsMock);
    }

    @Test
    public void when_callingApplyGradient_expect_correctGradientAppliedAndIterationUpdated() {
        // Arrange
        setup(false);
        Gradients gradientsMock = mock(Gradients.class);

        // Act
        sut.applyGradient(gradientsMock, 345);

        // Assert
        verify(handler1, times(1)).applyGradient(gradientsMock, 345);
        verify(handler2, times(1)).applyGradient(gradientsMock, 345);
    }

    @Test
    public void when_callingRecurrentStepOutput_expect_recurrentStepCalledWithObservationData() {
        // Arrange
        setup(false);
        Observation observationMock = mock(Observation.class);
        double[] recurrentStepOutput1 = new double[] { 1.0, 2.0, 3.0};
        double[] recurrentStepOutput2 = new double[] { 10.0, 20.0, 30.0};
        when(handler1.recurrentStepOutput(observationMock)).thenReturn(new INDArray[] { Nd4j.create(recurrentStepOutput1) });
        when(handler2.recurrentStepOutput(observationMock)).thenReturn(new INDArray[] { Nd4j.create(recurrentStepOutput2) });

        // Act
        INDArray[] results = sut.recurrentStepOutput(observationMock);

        // Assert
        verify(handler1, times(1)).recurrentStepOutput(observationMock);
        verify(handler2, times(1)).recurrentStepOutput(observationMock);
        assertEquals(2, results.length);
        assertArrayEquals(results[0].toDoubleVector(), recurrentStepOutput1, 0.00001);
        assertArrayEquals(results[1].toDoubleVector(), recurrentStepOutput2, 0.00001);
    }

    @Test
    public void when_callingFeaturesBatchOutput_expect_outputCalledWithBatch() {
        // Arrange
        setup(false);
        INDArray batch = Nd4j.rand(1, 2);
        Features features = new Features(new INDArray[] { batch });
        when(handler1.batchOutput(features)).thenReturn(new INDArray[] { batch.mul(2.0) });
        when(handler2.batchOutput(features)).thenReturn(new INDArray[] { batch.div(2.0) });

        // Act
        INDArray[] results = sut.batchOutput(features);

        // Assert
        verify(handler1, times(1)).batchOutput(features);
        verify(handler2, times(1)).batchOutput(features);
        assertEquals(2, results.length);
        assertArrayEquals(results[0].toDoubleVector(), batch.mul(2.0).toDoubleVector(), 0.00001);
        assertArrayEquals(results[1].toDoubleVector(), batch.div(2.0).toDoubleVector(), 0.00001);
    }

    @Test
    public void when_callingResetState_expect_recurrentHandlersAreReset() {
        // Arrange
        setup(true);

        // Act
        sut.resetState();

        // Assert
        verify(handler1, times(1)).resetState();
        verify(handler2, never()).resetState();
    }

    @Test
    public void when_callingClone_expect_handlersAreCloned() throws Exception {
        // Arrange
        setup(false);
        when(handler1.clone()).thenReturn(handler1);
        when(handler2.clone()).thenReturn(handler2);


        // Act
        CompoundNetworkHandler result = (CompoundNetworkHandler)sut.clone();

        // Assert
        assertNotSame(sut, result);

        verify(handler1, times(1)).clone();
        verify(handler2, times(1)).clone();
    }

    @Test
    public void when_callingCopyFrom_expect_handlersParamsAreCopied() {
        // Arrange
        setup(false);
        CompoundNetworkHandler from = new CompoundNetworkHandler(handler1, handler2);

        // Act
        sut.copyFrom(from);

        // Assert
        verify(handler1, times(1)).copyFrom(handler1);
        verify(handler2, times(1)).copyFrom(handler2);
    }

    @Test
    public void when_noHandlerIsRecurrent_expect_isRecurrentFalse() {
        // Arrange
        setup(false);

        // Act
        boolean isRecurrent = sut.isRecurrent();

        // Assert
        assertFalse(isRecurrent);
    }

    @Test
    public void when_aHandlerIsRecurrent_expect_isRecurrentTrue() {
        // Arrange
        setup(true);

        // Act
        boolean isRecurrent = sut.isRecurrent();

        // Assert
        assertTrue(isRecurrent);
    }
}