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

package org.deeplearning4j.spark.models.sequencevectors.learning;

import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Identification layer for Spark-ready implementations of LearningAlgorithms
 *
 * @author raver119@gmail.com
 */
public interface SparkElementsLearningAlgorithm extends ElementsLearningAlgorithm<ShallowSequenceElement> {
    TrainingDriver<? extends TrainingMessage> getTrainingDriver();

    Frame<? extends TrainingMessage> frameSequence(Sequence<ShallowSequenceElement> sequence, AtomicLong nextRandom,
                    double learningRate);
}
