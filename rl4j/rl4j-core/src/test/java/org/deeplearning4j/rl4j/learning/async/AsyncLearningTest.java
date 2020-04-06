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

package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.support.MockAsyncConfiguration;
import org.deeplearning4j.rl4j.support.MockAsyncGlobal;
import org.deeplearning4j.rl4j.support.MockEncodable;
import org.deeplearning4j.rl4j.support.MockNeuralNet;
import org.deeplearning4j.rl4j.support.MockPolicy;
import org.deeplearning4j.rl4j.support.MockTrainingListener;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AsyncLearningTest {

    @Test
    public void when_training_expect_AsyncGlobalStarted() {
        // Arrange
        TestContext context = new TestContext();
        context.asyncGlobal.setMaxLoops(1);

        // Act
        context.sut.train();

        // Assert
        assertTrue(context.asyncGlobal.hasBeenStarted);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_trainStartReturnsStop_expect_noTraining() {
        // Arrange
        TestContext context = new TestContext();
        context.listener.setRemainingTrainingStartCallCount(0);
        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingStartCallCount);
        assertEquals(1, context.listener.onTrainingEndCallCount);
        assertEquals(0, context.policy.playCallCount);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_trainingIsComplete_expect_trainingStop() {
        // Arrange
        TestContext context = new TestContext();

        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingStartCallCount);
        assertEquals(1, context.listener.onTrainingEndCallCount);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_training_expect_onTrainingProgressCalled() {
        // Arrange
        TestContext context = new TestContext();

        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingProgressCallCount);
    }


    public static class TestContext {
        MockAsyncConfiguration config = new MockAsyncConfiguration(1L, 11, 0, 0, 0, 0,0, 0, 0, 0);
        public final MockAsyncGlobal asyncGlobal = new MockAsyncGlobal();
        public final MockPolicy policy = new MockPolicy();
        public final TestAsyncLearning sut = new TestAsyncLearning(config, asyncGlobal, policy);
        public final MockTrainingListener listener = new MockTrainingListener(asyncGlobal);

        public TestContext() {
            sut.addListener(listener);
            asyncGlobal.setMaxLoops(1);
            sut.setProgressMonitorFrequency(1);
        }
    }

    public static class TestAsyncLearning extends AsyncLearning<MockEncodable, Integer, DiscreteSpace, MockNeuralNet> {
        private final IAsyncLearningConfiguration conf;
        private final IAsyncGlobal asyncGlobal;
        private final IPolicy<MockEncodable, Integer> policy;

        public TestAsyncLearning(IAsyncLearningConfiguration conf, IAsyncGlobal asyncGlobal, IPolicy<MockEncodable, Integer> policy) {
            this.conf = conf;
            this.asyncGlobal = asyncGlobal;
            this.policy = policy;
        }

        @Override
        public IPolicy getPolicy() {
            return policy;
        }

        @Override
        public IAsyncLearningConfiguration getConfiguration() {
            return conf;
        }

        @Override
        protected AsyncThread newThread(int i, int deviceAffinity) {
            return null;
        }

        @Override
        public MDP getMdp() {
            return null;
        }

        @Override
        protected IAsyncGlobal getAsyncGlobal() {
            return asyncGlobal;
        }

        @Override
        public MockNeuralNet getNeuralNet() {
            return null;
        }
    }

}
