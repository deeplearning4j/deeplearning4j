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

package org.deeplearning4j.spark.impl.paramavg;

import lombok.Data;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * The results (parameters, optional updaters) returned by a {@link ParameterAveragingTrainingWorker} to the
 * {@link ParameterAveragingTrainingMaster}
 *
 * @author Alex Black
 */
@Data
public class ParameterAveragingTrainingResult implements TrainingResult {

    private final INDArray parameters;
    private final INDArray updaterState;
    private final double score;
    private SparkTrainingStats sparkTrainingStats;

    private final Collection<StorageMetaData> listenerMetaData;
    private final Collection<Persistable> listenerStaticInfo;
    private final Collection<Persistable> listenerUpdates;


    public ParameterAveragingTrainingResult(INDArray parameters, INDArray updaterState, double score,
                    Collection<StorageMetaData> listenerMetaData, Collection<Persistable> listenerStaticInfo,
                    Collection<Persistable> listenerUpdates) {
        this(parameters, updaterState, score, null, listenerMetaData, listenerStaticInfo, listenerUpdates);
    }

    public ParameterAveragingTrainingResult(INDArray parameters, INDArray updaterState, double score,
                    SparkTrainingStats sparkTrainingStats, Collection<StorageMetaData> listenerMetaData,
                    Collection<Persistable> listenerStaticInfo, Collection<Persistable> listenerUpdates) {
        this.parameters = parameters;
        this.updaterState = updaterState;
        this.score = score;
        this.sparkTrainingStats = sparkTrainingStats;

        this.listenerMetaData = listenerMetaData;
        this.listenerStaticInfo = listenerStaticInfo;
        this.listenerUpdates = listenerUpdates;
    }

    @Override
    public void setStats(SparkTrainingStats sparkTrainingStats) {
        this.sparkTrainingStats = sparkTrainingStats;
    }
}
