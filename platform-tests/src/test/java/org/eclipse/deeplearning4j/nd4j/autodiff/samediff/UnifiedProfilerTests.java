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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.profiler.OpContextTracker;
import org.nd4j.linalg.profiler.UnifiedProfiler;
import org.nd4j.linalg.profiler.data.eventlogger.EventLogger;
import org.nd4j.linalg.profiler.data.eventlogger.EventType;
import org.nd4j.linalg.profiler.data.eventlogger.ObjectAllocationType;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;


import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.INTEGER_0_10;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

@Slf4j
@NativeTag
@Tag(TagNames.TRAINING)
@Tag(TagNames.SAMEDIFF)
public class UnifiedProfilerTests extends BaseNd4jTestWithBackends {
    @SneakyThrows
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testProfiler(Nd4jBackend backend) {
        Nd4j.getProfiler().start();
        try {
            EventLogger.getInstance().setLogStream(new PrintStream(new FileOutputStream("your-logfile.log")));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        EventLogger.getInstance().setEventTypesToLog(Arrays.asList(EventType.DEALLOCATION,EventType.ALLOCATION));
        EventLogger.getInstance().setAllocationTypesToLog(Arrays.asList(ObjectAllocationType.DATA_BUFFER));


        int batchSize = 4;
        int modelDim = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new XavierInitScheme('c', modelDim, modelDim), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.var("bias", new ZeroInitScheme('c'), FLOAT, modelDim);
        SDVariable predictions = sd.nn.linear("predictions", features, weights, bias);
        SDVariable loss = sd.loss.meanSquaredError("loss", labels, predictions, null);
        loss.markAsLoss();
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        DataSetIterator iterator = new RandomDataSetIterator(1, new long[]{batchSize, modelDim}, new long[]{batchSize, modelDim}, INTEGER_0_10, INTEGER_0_10);

        sd.fit(iterator, 10);
        sd.output(iterator,"predictions");
        System.out.println(Nd4j.getProfiler().printCurrentStats());
        Thread.sleep(10000);
        Nd4j.getProfiler().stop();

    }

}
