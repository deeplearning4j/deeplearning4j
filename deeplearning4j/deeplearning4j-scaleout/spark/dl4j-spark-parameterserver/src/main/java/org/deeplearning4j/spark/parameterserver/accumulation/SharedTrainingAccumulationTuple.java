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

package org.deeplearning4j.spark.parameterserver.accumulation;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
@AllArgsConstructor
@Data
@NoArgsConstructor
@Builder
public class SharedTrainingAccumulationTuple implements Serializable {
    private INDArray updaterStateArray;
    private double scoreSum;
    private int aggregationsCount;
    private SparkTrainingStats sparkTrainingStats;
    private Collection<StorageMetaData> listenerMetaData;
    private Collection<Persistable> listenerStaticInfo;
    private Collection<Persistable> listenerUpdates;
    private Map<String,Integer> minibatchesPerExecutor;
}
