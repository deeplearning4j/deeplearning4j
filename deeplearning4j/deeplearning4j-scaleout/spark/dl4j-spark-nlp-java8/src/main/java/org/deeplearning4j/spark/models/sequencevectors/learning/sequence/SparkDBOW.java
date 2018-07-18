/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.spark.models.sequencevectors.learning.sequence;

import org.deeplearning4j.models.embeddings.learning.impl.elements.RandomUtils;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.spark.models.sequencevectors.learning.elements.BaseSparkLearningAlgorithm;
import org.deeplearning4j.spark.models.sequencevectors.learning.elements.SparkSkipGram;
import org.nd4j.parameterserver.distributed.logic.sequence.BasicSequenceProvider;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Spark implementation for PV-DBOW training algorithm
 * @author raver119@gmail.com
 */
public class SparkDBOW extends SparkSkipGram {
    @Override
    public String getCodeName() {
        return "Spark-DBOW";
    }


    @Override
    public Frame<? extends TrainingMessage> frameSequence(Sequence<ShallowSequenceElement> sequence,
                    AtomicLong nextRandom, double learningRate) {
        if (vectorsConfiguration.getSampling() > 0)
            sequence = BaseSparkLearningAlgorithm.applySubsampling(sequence, nextRandom, 10L,
                            vectorsConfiguration.getSampling());

        int currentWindow = vectorsConfiguration.getWindow();

        if (vectorsConfiguration.getVariableWindows() != null
                        && vectorsConfiguration.getVariableWindows().length != 0) {
            currentWindow = vectorsConfiguration.getVariableWindows()[RandomUtils
                            .nextInt(vectorsConfiguration.getVariableWindows().length)];
        }
        if (frame == null)
            synchronized (this) {
                if (frame == null)
                    frame = new ThreadLocal<>();
            }

        if (frame.get() == null)
            frame.set(new Frame<SkipGramRequestMessage>(BasicSequenceProvider.getInstance().getNextValue()));

        for (ShallowSequenceElement lastWord : sequence.getSequenceLabels()) {
            for (ShallowSequenceElement word : sequence.getElements()) {
                iterateSample(word, lastWord, nextRandom, learningRate);
                nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            }
        }

        // at this moment we should have something in ThreadLocal Frame, so we'll send it to VoidParameterServer for processing

        Frame<SkipGramRequestMessage> currentFrame = frame.get();
        frame.set(new Frame<SkipGramRequestMessage>(BasicSequenceProvider.getInstance().getNextValue()));

        return currentFrame;
    }

    @Override
    public TrainingDriver<? extends TrainingMessage> getTrainingDriver() {
        return driver;
    }
}
