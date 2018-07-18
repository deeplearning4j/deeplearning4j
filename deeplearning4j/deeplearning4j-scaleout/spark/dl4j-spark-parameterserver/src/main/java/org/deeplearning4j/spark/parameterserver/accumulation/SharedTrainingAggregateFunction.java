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

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingAggregateFunction implements
                Function2<SharedTrainingAccumulationTuple, SharedTrainingResult, SharedTrainingAccumulationTuple> {

    @Override
    public SharedTrainingAccumulationTuple call(SharedTrainingAccumulationTuple tuple, SharedTrainingResult result)
                    throws Exception {
        if (tuple == null) {
            return SharedTrainingAccumulationTuple.builder().updaterStateArray(result.getUpdaterStateArray())
                            .scoreSum(result.getScoreSum()).listenerStaticInfo(result.getListenerStaticInfo())
                            .listenerUpdates(result.getListenerUpdates()).listenerMetaData(result.getListenerMetaData())
                            .sparkTrainingStats(result.getSparkTrainingStats())
                            .aggregationsCount(result.getAggregationsCount())
                            .minibatchesPerExecutor(result.getMinibatchesPerExecutor())
                    .build();
        }


        INDArray updaterStateSum = null;
        int aggregationsCount = 0;
        double score = 0.0;
        if (tuple.getUpdaterStateArray() != null) {
            if (result.getUpdaterStateArray() != null) {
                updaterStateSum = tuple.getUpdaterStateArray().addi(result.getUpdaterStateArray());
                aggregationsCount = tuple.getAggregationsCount() + 1;
                score = tuple.getScoreSum() + result.getScoreSum();
            }
        } else {
            if (result.getUpdaterStateArray() != null) {
                updaterStateSum = result.getUpdaterStateArray();
                aggregationsCount = 1;
                score = result.getScoreSum();
            }
        }

        SparkTrainingStats stats = tuple.getSparkTrainingStats();
        if (result.getSparkTrainingStats() != null) {
            if (stats == null)
                stats = result.getSparkTrainingStats();
            else
                stats.addOtherTrainingStats(result.getSparkTrainingStats());
        }

        Nd4j.getExecutioner().commit();

        Collection<StorageMetaData> listenerMetaData = tuple.getListenerMetaData();
        if (listenerMetaData == null)
            listenerMetaData = result.getListenerMetaData();
        else {
            Collection<StorageMetaData> newMeta = result.getListenerMetaData();
            if (newMeta != null)
                listenerMetaData.addAll(newMeta);
        }

        Collection<Persistable> listenerStaticInfo = tuple.getListenerStaticInfo();
        if (listenerStaticInfo == null)
            listenerStaticInfo = result.getListenerStaticInfo();
        else {
            Collection<Persistable> newStatic = result.getListenerStaticInfo();
            if (newStatic != null)
                listenerStaticInfo.addAll(newStatic);
        }

        Collection<Persistable> listenerUpdates = tuple.getListenerUpdates();
        if (listenerUpdates == null)
            listenerUpdates = result.getListenerUpdates();
        else {
            Collection<Persistable> listenerUpdates2 = result.getListenerUpdates();
            if (listenerUpdates2 != null)
                listenerUpdates.addAll(listenerUpdates2);
        }

        Map<String,Integer> minibatchesPerExecutor = new HashMap<>();
        if(tuple.getMinibatchesPerExecutor() != null) {
            for (Map.Entry<String, Integer> e : tuple.getMinibatchesPerExecutor().entrySet()){
                minibatchesPerExecutor.put(e.getKey(), e.getValue());
            }
        }
        if(result.getMinibatchesPerExecutor() != null){
            for (Map.Entry<String, Integer> e : result.getMinibatchesPerExecutor().entrySet()){
                if(minibatchesPerExecutor.containsKey(e.getKey())){
                    minibatchesPerExecutor.put(e.getKey(), minibatchesPerExecutor.get(e.getKey()) + e.getValue());
                } else {
                    minibatchesPerExecutor.put(e.getKey(), e.getValue());
                }
            }
        }

        return SharedTrainingAccumulationTuple.builder().scoreSum(score).updaterStateArray(updaterStateSum)
                        .aggregationsCount(aggregationsCount).sparkTrainingStats(stats)
                        .listenerMetaData(listenerMetaData).listenerUpdates(listenerUpdates)
                        .listenerStaticInfo(listenerStaticInfo)
                        .minibatchesPerExecutor(minibatchesPerExecutor).build();
    }
}
