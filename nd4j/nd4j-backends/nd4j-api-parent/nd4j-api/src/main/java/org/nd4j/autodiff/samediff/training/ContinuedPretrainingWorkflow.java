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

package org.nd4j.autodiff.samediff.training;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.config.ContinuedPretrainingConfig;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.Collections;

/**
 * Orchestrates continued pretraining (domain adaptation) workflows.
 *
 * This workflow trains on all tokens (no masking), which distinguishes it
 * from instruction tuning. Used to adapt pretrained LLMs to specific domains.
 *
 * Usage:
 * <pre>
 * ContinuedPretrainingConfig config = ContinuedPretrainingConfig.builder()
 *     .learningRate(5e-5)
 *     .chunkSize(2048)
 *     .numEpochs(1)
 *     .build();
 *
 * ContinuedPretrainingWorkflow workflow = new ContinuedPretrainingWorkflow(model, config);
 * workflow.train(dataIterator, totalSteps);
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class ContinuedPretrainingWorkflow {

    @Getter
    private final SameDiff model;

    @Getter
    private final ContinuedPretrainingConfig config;

    public ContinuedPretrainingWorkflow(SameDiff model, ContinuedPretrainingConfig config) {
        this.model = model;
        this.config = config;
        config.validate();
    }

    /**
     * Build a TrainingConfig from the continued pretraining configuration.
     *
     * @param totalSteps Total number of training steps (for schedule computation)
     * @return A configured TrainingConfig ready for model.fit()
     */
    public TrainingConfig buildTrainingConfig(int totalSteps) {
        // Build learning rate schedule
        ISchedule lrSchedule = config.getLrSchedule();
        if (lrSchedule == null) {
            lrSchedule = CosineWarmupSchedule.fromRatio(
                    config.getLearningRate(),
                    config.getMinLearningRate(),
                    config.getWarmupRatio(),
                    totalSteps
            );
        }

        // Build optimizer
        Adam adam = Adam.builder()
                .learningRateSchedule(lrSchedule)
                .beta1(0.9)
                .beta2(0.999)
                .epsilon(1e-8)
                .build();

        TrainingConfig.Builder builder = TrainingConfig.builder()
                .updater(adam)
                .dataSetFeatureMapping("input_ids")
                .dataSetLabelMapping("labels")
                .computeDataType(config.getComputeDataType())
                .masterWeightDataType(org.nd4j.linalg.api.buffer.DataType.FLOAT)
                .gradientAccumulationSteps(config.getGradientAccumulationSteps());

        // Add weight decay
        if (config.getWeightDecay() > 0) {
            builder.weightDecay(config.getWeightDecay(), true);
        }

        return builder.build();
    }

    /**
     * Execute the continued pretraining workflow.
     *
     * @param dataIterator Data iterator providing chunked text as MultiDataSets
     * @param totalSteps   Total number of training steps
     */
    public void train(MultiDataSetIterator dataIterator, int totalSteps) {
        TrainingConfig trainingConfig = buildTrainingConfig(totalSteps);
        model.setTrainingConfig(trainingConfig);

        log.info("Starting continued pretraining: lr={}, chunks={}, overlap={}, epochs={}",
                config.getLearningRate(), config.getChunkSize(),
                config.getChunkOverlap(), config.getNumEpochs());

        for (int epoch = 0; epoch < config.getNumEpochs(); epoch++) {
            log.info("Epoch {}/{}", epoch + 1, config.getNumEpochs());
            model.fit(dataIterator, 1);
            dataIterator.reset();
        }

        log.info("Continued pretraining complete");
    }
}
