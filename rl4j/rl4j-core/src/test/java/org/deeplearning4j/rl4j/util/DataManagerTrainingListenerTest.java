/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.util;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.configuration.ILearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.sync.support.MockStatEntry;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.support.MockDataManager;
import org.deeplearning4j.rl4j.support.MockHistoryProcessor;
import org.deeplearning4j.rl4j.support.MockMDP;
import org.deeplearning4j.rl4j.support.MockObservationSpace;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

public class DataManagerTrainingListenerTest {

    @Test
    public void when_callingOnNewEpochWithoutHistoryProcessor_expect_noException() {
        // Arrange
        TestTrainer trainer = new TestTrainer();
        DataManagerTrainingListener sut = new DataManagerTrainingListener(new MockDataManager(false));

        // Act
        TrainingListener.ListenerResponse response = sut.onNewEpoch(trainer);

        // Assert
        assertEquals(TrainingListener.ListenerResponse.CONTINUE, response);
    }

    @Test
    public void when_callingOnNewEpochWithHistoryProcessor_expect_startMonitorNotCalled() {
        // Arrange
        TestTrainer trainer = new TestTrainer();
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 4, 4, 4, 4, 0, 0, 2);
        MockHistoryProcessor hp = new MockHistoryProcessor(hpConf);
        trainer.setHistoryProcessor(hp);
        DataManagerTrainingListener sut = new DataManagerTrainingListener(new MockDataManager(false));

        // Act
        TrainingListener.ListenerResponse response = sut.onNewEpoch(trainer);

        // Assert
        assertEquals(1, hp.startMonitorCallCount);
        assertEquals(TrainingListener.ListenerResponse.CONTINUE, response);
    }

    @Test
    public void when_callingOnEpochTrainingResultWithoutHistoryProcessor_expect_noException() {
        // Arrange
        TestTrainer trainer = new TestTrainer();
        DataManagerTrainingListener sut = new DataManagerTrainingListener(new MockDataManager(false));

        // Act
        TrainingListener.ListenerResponse response = sut.onEpochTrainingResult(trainer, null);

        // Assert
        assertEquals(TrainingListener.ListenerResponse.CONTINUE, response);
    }

    @Test
    public void when_callingOnNewEpochWithHistoryProcessor_expect_stopMonitorNotCalled() {
        // Arrange
        TestTrainer trainer = new TestTrainer();
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 4, 4, 4, 4, 0, 0, 2);
        MockHistoryProcessor hp = new MockHistoryProcessor(hpConf);
        trainer.setHistoryProcessor(hp);
        DataManagerTrainingListener sut = new DataManagerTrainingListener(new MockDataManager(false));

        // Act
        TrainingListener.ListenerResponse response = sut.onEpochTrainingResult(trainer, null);

        // Assert
        assertEquals(1, hp.stopMonitorCallCount);
        assertEquals(TrainingListener.ListenerResponse.CONTINUE, response);
    }

    @Test
    public void when_callingOnEpochTrainingResult_expect_callToDataManagerAppendStat() {
        // Arrange
        TestTrainer trainer = new TestTrainer();
        MockDataManager dm = new MockDataManager(false);
        DataManagerTrainingListener sut = new DataManagerTrainingListener(dm);
        MockStatEntry statEntry = new MockStatEntry(0, 0, 0.0);

        // Act
        TrainingListener.ListenerResponse response = sut.onEpochTrainingResult(trainer, statEntry);

        // Assert
        assertEquals(TrainingListener.ListenerResponse.CONTINUE, response);
        assertEquals(1, dm.statEntries.size());
        assertSame(statEntry, dm.statEntries.get(0));
    }

    @Test
    public void when_callingOnTrainingProgress_expect_callToDataManagerSaveAndWriteInfo() {
        // Arrange
        TestTrainer learning = new TestTrainer();
        MockDataManager dm = new MockDataManager(false);
        DataManagerTrainingListener sut = new DataManagerTrainingListener(dm);

        // Act
        TrainingListener.ListenerResponse response = sut.onTrainingProgress(learning);

        // Assert
        assertEquals(TrainingListener.ListenerResponse.CONTINUE, response);
        assertEquals(1, dm.writeInfoCallCount);
        assertEquals(1, dm.saveCallCount);
    }

    @Test
    public void when_stepCounterCloseToLastSave_expect_dataManagerSaveNotCalled() {
        // Arrange
        TestTrainer learning = new TestTrainer();
        MockDataManager dm = new MockDataManager(false);
        DataManagerTrainingListener sut = new DataManagerTrainingListener(dm);

        // Act
        TrainingListener.ListenerResponse response = sut.onTrainingProgress(learning);
        TrainingListener.ListenerResponse response2 = sut.onTrainingProgress(learning);

        // Assert
        assertEquals(TrainingListener.ListenerResponse.CONTINUE, response);
        assertEquals(TrainingListener.ListenerResponse.CONTINUE, response2);
        assertEquals(1, dm.saveCallCount);
    }

    private static class TestTrainer implements IEpochTrainer, ILearning
    {
        @Override
        public int getStepCount() {
            return 0;
        }

        @Override
        public int getEpochCount() {
            return 0;
        }

        @Override
        public int getEpisodeCount() {
            return 0;
        }

        @Override
        public int getCurrentEpisodeStepCount() {
            return 0;
        }


        @Getter
        @Setter
        private IHistoryProcessor historyProcessor;

        @Override
        public IPolicy getPolicy() {
            return null;
        }

        @Override
        public void train() {

        }

        @Override
        public ILearningConfiguration getConfiguration() {
            return null;
        }

        @Override
        public MDP getMdp() {
            return new MockMDP(new MockObservationSpace());
        }
    }
}
