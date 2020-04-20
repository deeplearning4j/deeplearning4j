/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.learning.listener;

import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;
import org.mockito.Mock;

import static org.junit.Assert.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class TrainingListenerListTest {

    @Mock
    IEpochTrainer mockTrainer;

    @Mock
    ILearning mockLearning;

    @Mock
    IDataManager.StatEntry mockStatEntry;

    @Test
    public void when_listIsEmpty_expect_notifyReturnTrue() {
        // Arrange
        TrainingListenerList trainingListenerList = new TrainingListenerList();

        // Act
        boolean resultTrainingStarted = trainingListenerList.notifyTrainingStarted();
        boolean resultNewEpoch = trainingListenerList.notifyNewEpoch(null);
        boolean resultEpochFinished = trainingListenerList.notifyEpochTrainingResult(null, null);

        // Assert
        assertTrue(resultTrainingStarted);
        assertTrue(resultNewEpoch);
        assertTrue(resultEpochFinished);
    }

    @Test
    public void when_firstListerStops_expect_othersListnersNotCalled() {
        // Arrange
        TrainingListener listener1 = mock(TrainingListener.class);
        TrainingListener listener2 = mock(TrainingListener.class);
        TrainingListenerList trainingListenerList = new TrainingListenerList();
        trainingListenerList.add(listener1);
        trainingListenerList.add(listener2);

        when(listener1.onTrainingStart()).thenReturn(TrainingListener.ListenerResponse.STOP);
        when(listener1.onNewEpoch(eq(mockTrainer))).thenReturn(TrainingListener.ListenerResponse.STOP);
        when(listener1.onEpochTrainingResult(eq(mockTrainer), eq(mockStatEntry))).thenReturn(TrainingListener.ListenerResponse.STOP);
        when(listener1.onTrainingProgress(eq(mockLearning))).thenReturn(TrainingListener.ListenerResponse.STOP);

        // Act
        trainingListenerList.notifyTrainingStarted();
        trainingListenerList.notifyNewEpoch(mockTrainer);
        trainingListenerList.notifyEpochTrainingResult(mockTrainer, null);
        trainingListenerList.notifyTrainingProgress(mockLearning);
        trainingListenerList.notifyTrainingFinished();

        // Assert

        verify(listener1, times(1)).onTrainingStart();
        verify(listener2, never()).onTrainingStart();

        verify(listener1, times(1)).onNewEpoch(eq(mockTrainer));
        verify(listener2, never()).onNewEpoch(eq(mockTrainer));

        verify(listener1, times(1)).onEpochTrainingResult(eq(mockTrainer), eq(mockStatEntry));
        verify(listener2, never()).onEpochTrainingResult(eq(mockTrainer), eq(mockStatEntry));

        verify(listener1, times(1)).onTrainingProgress(eq(mockLearning));
        verify(listener2, never()).onTrainingProgress(eq(mockLearning));

        verify(listener1, times(1)).onTrainingEnd();
        verify(listener2, times(1)).onTrainingEnd();
    }

    @Test
    public void when_allListenersContinue_expect_listReturnsTrue() {
        // Arrange
        TrainingListener listener1 = mock(TrainingListener.class);
        TrainingListener listener2 = mock(TrainingListener.class);
        TrainingListenerList trainingListenerList = new TrainingListenerList();
        trainingListenerList.add(listener1);
        trainingListenerList.add(listener2);

        // Act
        boolean resultTrainingStarted = trainingListenerList.notifyTrainingStarted();
        boolean resultNewEpoch = trainingListenerList.notifyNewEpoch(null);
        boolean resultEpochTrainingResult = trainingListenerList.notifyEpochTrainingResult(null, null);
        boolean resultProgress = trainingListenerList.notifyTrainingProgress(null);

        // Assert
        assertTrue(resultTrainingStarted);
        assertTrue(resultNewEpoch);
        assertTrue(resultEpochTrainingResult);
        assertTrue(resultProgress);
    }
}
