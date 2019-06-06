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

package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkElementsLearningAlgorithm;
import org.nd4j.linalg.primitives.Counter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;

/**
 * This accumulator function does count individual elements, using provided Accumulator
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CountFunction<T extends SequenceElement> implements Function<Sequence<T>, Pair<Sequence<T>, Long>> {
    protected Accumulator<Counter<Long>> accumulator;
    protected boolean fetchLabels;
    protected Broadcast<VoidConfiguration> voidConfigurationBroadcast;
    protected Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast;

    protected transient SparkElementsLearningAlgorithm ela;
    protected transient TrainingDriver<? extends TrainingMessage> driver;

    public CountFunction(@NonNull Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast,
                    @NonNull Broadcast<VoidConfiguration> voidConfigurationBroadcast,
                    @NonNull Accumulator<Counter<Long>> accumulator, boolean fetchLabels) {
        this.accumulator = accumulator;
        this.fetchLabels = fetchLabels;
        this.voidConfigurationBroadcast = voidConfigurationBroadcast;
        this.vectorsConfigurationBroadcast = vectorsConfigurationBroadcast;
    }

    @Override
    public Pair<Sequence<T>, Long> call(Sequence<T> sequence) throws Exception {
        // since we can't be 100% sure that sequence size is ok itself, or it's not overflow through int limits, we'll recalculate it.
        // anyway we're going to loop through it for elements frequencies
        Counter<Long> localCounter = new Counter<>();
        long seqLen = 0;

        if (ela == null) {
            try {
                ela = (SparkElementsLearningAlgorithm) Class
                                .forName(vectorsConfigurationBroadcast.getValue().getElementsLearningAlgorithm())
                                .newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        driver = ela.getTrainingDriver();

        //System.out.println("Initializing VoidParameterServer in CountFunction");
        VoidParameterServer.getInstance().init(voidConfigurationBroadcast.getValue(), new RoutedTransport(), driver);

        for (T element : sequence.getElements()) {
            if (element == null)
                continue;

            // FIXME: hashcode is bad idea here. we need Long id
            localCounter.incrementCount(element.getStorageId(), 1.0f);
            seqLen++;
        }

        // FIXME: we're missing label information here due to shallow vocab mechanics
        if (sequence.getSequenceLabels() != null)
            for (T label : sequence.getSequenceLabels()) {
                localCounter.incrementCount(label.getStorageId(), 1.0f);
            }

        accumulator.add(localCounter);

        return Pair.makePair(sequence, seqLen);
    }
}
