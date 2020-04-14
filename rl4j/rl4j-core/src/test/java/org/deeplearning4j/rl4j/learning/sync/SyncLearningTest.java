/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.learning.sync;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.configuration.ILearningConfiguration;
import org.deeplearning4j.rl4j.learning.configuration.LearningConfiguration;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.support.MockStatEntry;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.support.MockTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SyncLearningTest {

    @Test
    public void when_training_expect_listenersToBeCalled() {
        // Arrange
        QLearningConfiguration lconfig = QLearningConfiguration.builder().maxStep(10).build();
        MockTrainingListener listener = new MockTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(10, listener.onNewEpochCallCount);
        assertEquals(10, listener.onEpochTrainingResultCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_trainingStartCanContinueFalse_expect_trainingStopped() {
        // Arrange
        QLearningConfiguration lconfig = QLearningConfiguration.builder().maxStep(10).build();
        MockTrainingListener listener = new MockTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.setRemainingTrainingStartCallCount(0);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(0, listener.onNewEpochCallCount);
        assertEquals(0, listener.onEpochTrainingResultCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_newEpochCanContinueFalse_expect_trainingStopped() {
        // Arrange
        QLearningConfiguration lconfig = QLearningConfiguration.builder().maxStep(10).build();
        MockTrainingListener listener = new MockTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.setRemainingOnNewEpochCallCount(2);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(3, listener.onNewEpochCallCount);
        assertEquals(2, listener.onEpochTrainingResultCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_epochTrainingResultCanContinueFalse_expect_trainingStopped() {
        // Arrange
        LearningConfiguration lconfig = QLearningConfiguration.builder().maxStep(10).build();
        MockTrainingListener listener = new MockTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.setRemainingOnEpochTrainingResult(2);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(3, listener.onNewEpochCallCount);
        assertEquals(3, listener.onEpochTrainingResultCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    public static class MockSyncLearning extends SyncLearning {

        private final ILearningConfiguration conf;

        @Getter
        private int currentEpochStep = 0;

        public MockSyncLearning(ILearningConfiguration conf) {
            this.conf = conf;
        }

        @Override
        protected void preEpoch() { currentEpochStep = 0;  }

        @Override
        protected void postEpoch() { }

        @Override
        protected IDataManager.StatEntry trainEpoch() {
            setStepCounter(getStepCounter() + 1);
            return new MockStatEntry(getCurrentEpochStep(), getStepCounter(), 1.0);
        }

        @Override
        public NeuralNet getNeuralNet() {
            return null;
        }

        @Override
        public IPolicy getPolicy() {
            return null;
        }

        @Override
        public ILearningConfiguration getConfiguration() {
            return conf;
        }

        @Override
        public MDP getMdp() {
            return null;
        }
    }
}
