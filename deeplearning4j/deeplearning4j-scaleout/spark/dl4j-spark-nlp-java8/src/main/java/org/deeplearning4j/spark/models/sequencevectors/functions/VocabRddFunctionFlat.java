/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.common.config.DL4JClassLoading;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkElementsLearningAlgorithm;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class VocabRddFunctionFlat<T extends SequenceElement> implements FlatMapFunction<Sequence<T>, T> {
    protected Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast;
    protected Broadcast<VoidConfiguration> paramServerConfigurationBroadcast;

    protected transient VectorsConfiguration configuration;
    protected transient SparkElementsLearningAlgorithm ela;
    protected transient TrainingDriver<? extends TrainingMessage> driver;

    public VocabRddFunctionFlat(@NonNull Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast,
                            @NonNull Broadcast<VoidConfiguration> paramServerConfigurationBroadcast) {
        this.vectorsConfigurationBroadcast = vectorsConfigurationBroadcast;
        this.paramServerConfigurationBroadcast = paramServerConfigurationBroadcast;
    }

    @Override
    public Iterator<T> call(Sequence<T> sequence) throws Exception {
        if (configuration == null)
            configuration = vectorsConfigurationBroadcast.getValue();

        if (ela == null) {
            String className = configuration.getElementsLearningAlgorithm();
            ela = DL4JClassLoading.createNewInstance(className);
        }
        driver = ela.getTrainingDriver();

        // we just silently initialize server
        VoidParameterServer.getInstance().init(paramServerConfigurationBroadcast.getValue(), new RoutedTransport(),
                driver);

        // TODO: call for initializeSeqVec here

        List<T> elements = new ArrayList<>();

        elements.addAll(sequence.getElements());

        // FIXME: this is PROBABLY bad, we might want to ensure, there's no duplicates.
        if (configuration.isTrainSequenceVectors())
            if (!sequence.getSequenceLabels().isEmpty())
                elements.addAll(sequence.getSequenceLabels());

        return elements.iterator();
    }
}
