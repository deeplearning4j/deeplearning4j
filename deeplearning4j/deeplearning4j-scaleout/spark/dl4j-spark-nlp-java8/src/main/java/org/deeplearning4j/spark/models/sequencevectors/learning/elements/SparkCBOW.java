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

package org.deeplearning4j.spark.models.sequencevectors.learning.elements;

import org.deeplearning4j.models.embeddings.learning.impl.elements.RandomUtils;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.logic.sequence.BasicSequenceProvider;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.requests.CbowRequestMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.training.impl.CbowTrainer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class SparkCBOW extends BaseSparkLearningAlgorithm {

    TrainingDriver<CbowRequestMessage> driver = new CbowTrainer();

    protected transient ThreadLocal<Frame<CbowRequestMessage>> frame;

    @Override
    public String getCodeName() {
        return "Spark-CBOW";
    }

    @Override
    public Frame<? extends TrainingMessage> frameSequence(Sequence<ShallowSequenceElement> sequence,
                    AtomicLong nextRandom, double learningRate) {
        // FIXME: totalElementsCount should have real value
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
            frame.set(new Frame<CbowRequestMessage>(BasicSequenceProvider.getInstance().getNextValue()));


        for (int i = 0; i < sequence.getElements().size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
            int b = (int) nextRandom.get() % currentWindow;

            int end = currentWindow * 2 + 1 - b;

            ShallowSequenceElement currentWord = sequence.getElementByIndex(i);

            List<Integer> intsList = new ArrayList<>();
            for (int a = b; a < end; a++) {
                if (a != currentWindow) {
                    int c = i - currentWindow + a;
                    if (c >= 0 && c < sequence.size()) {
                        ShallowSequenceElement lastWord = sequence.getElementByIndex(c);

                        intsList.add(lastWord.getIndex());
                    }
                }
            }

            // just converting values to int
            int[] windowWords = new int[intsList.size()];
            for (int x = 0; x < windowWords.length; x++) {
                windowWords[x] = intsList.get(x);
            }

            if (windowWords.length < 1)
                continue;

            iterateSample(currentWord, windowWords, nextRandom, learningRate, false, 0, true, null);
        }

        Frame<CbowRequestMessage> currentFrame = frame.get();
        frame.set(new Frame<CbowRequestMessage>(BasicSequenceProvider.getInstance().getNextValue()));

        return currentFrame;
    }

    protected void iterateSample(ShallowSequenceElement currentWord, int[] windowWords, AtomicLong nextRandom,
                    double alpha, boolean isInference, int numLabels, boolean trainWords, INDArray inferenceVector) {
        int[] idxSyn1 = null;
        byte[] codes = null;

        if (vectorsConfiguration.isUseHierarchicSoftmax()) {
            idxSyn1 = new int[currentWord.getCodeLength()];
            codes = new byte[currentWord.getCodeLength()];
            for (int p = 0; p < currentWord.getCodeLength(); p++) {
                if (currentWord.getPoints().get(p) < 0)
                    continue;

                codes[p] = currentWord.getCodes().get(p);
                idxSyn1[p] = currentWord.getPoints().get(p);
            }
        } else {
            idxSyn1 = new int[0];
            codes = new byte[0];
        }


        CbowRequestMessage cbrm = new CbowRequestMessage(windowWords, idxSyn1, currentWord.getIndex(), codes,
                        (int) vectorsConfiguration.getNegative(), alpha, nextRandom.get());
        frame.get().stackMessage(cbrm);
    }

    @Override
    public TrainingDriver<? extends TrainingMessage> getTrainingDriver() {
        return driver;
    }
}
