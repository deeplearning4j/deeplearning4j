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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingAccumulationFunction implements
                Function2<SharedTrainingAccumulationTuple, SharedTrainingAccumulationTuple, SharedTrainingAccumulationTuple> {

    @Override
    public SharedTrainingAccumulationTuple call(SharedTrainingAccumulationTuple tuple1,
                    SharedTrainingAccumulationTuple tuple2) throws Exception {
        // if one of tuples is null - return other one
        if (tuple1 == null)
            return tuple2;
        else if (tuple2 == null)
            return tuple1;

        double score = 0.0;
        INDArray stateView = null;
        int aggregationsCount = 0;
        if (tuple1.getUpdaterStateArray() != null && tuple2.getUpdaterStateArray() != null) {
            // we have multiple state views here. average them
            stateView = tuple1.getUpdaterStateArray().addi(tuple2.getUpdaterStateArray());
        } else if (tuple1.getUpdaterStateArray() != null || tuple2.getUpdaterStateArray() != null) {
            // only one of state views exists. just use it
            stateView = tuple1.getUpdaterStateArray() != null ? tuple1.getUpdaterStateArray()
                            : tuple2.getUpdaterStateArray();
        }

        // we assume that aggregationsCount field is set only for entries that hold updaters state
        aggregationsCount = tuple1.getAggregationsCount() + tuple2.getAggregationsCount();
        score = tuple1.getScoreSum() + tuple2.getScoreSum();

        // aggregating spark stats
        SparkTrainingStats stats = tuple1.getSparkTrainingStats();
        if (tuple2.getSparkTrainingStats() != null) {
            if (stats == null)
                stats = tuple2.getSparkTrainingStats();
            else
                stats.addOtherTrainingStats(tuple2.getSparkTrainingStats());
        }

        Nd4j.getExecutioner().commit();

        Collection<StorageMetaData> listenerMetaData = tuple1.getListenerMetaData();
        if (listenerMetaData == null)
            listenerMetaData = tuple2.getListenerMetaData();
        else {
            Collection<StorageMetaData> newMeta = tuple2.getListenerMetaData();
            if (newMeta != null)
                listenerMetaData.addAll(newMeta);
        }

        Collection<Persistable> listenerStaticInfo = tuple1.getListenerStaticInfo();
        if (listenerStaticInfo == null)
            listenerStaticInfo = tuple2.getListenerStaticInfo();
        else {
            Collection<Persistable> newStatic = tuple2.getListenerStaticInfo();
            if (newStatic != null)
                listenerStaticInfo.addAll(newStatic);
        }

        Collection<Persistable> listenerUpdates = tuple1.getListenerUpdates();
        if (listenerUpdates == null)
            listenerUpdates = tuple2.getListenerUpdates();
        else {
            Collection<Persistable> listenerUpdates2 = tuple2.getListenerUpdates();
            if (listenerUpdates2 != null)
                listenerUpdates.addAll(listenerUpdates2);
        }

        Map<String,Integer> minibatchesPerExecutor = new HashMap<>();
        if(tuple1.getMinibatchesPerExecutor() != null) {
            for (Map.Entry<String, Integer> e : tuple1.getMinibatchesPerExecutor().entrySet()){
                minibatchesPerExecutor.put(e.getKey(), e.getValue());
            }
        }
        if(tuple2.getMinibatchesPerExecutor() != null){
            for (Map.Entry<String, Integer> e : tuple2.getMinibatchesPerExecutor().entrySet()){
                if(minibatchesPerExecutor.containsKey(e.getKey())){
                    minibatchesPerExecutor.put(e.getKey(), minibatchesPerExecutor.get(e.getKey()) + e.getValue());
                } else {
                    minibatchesPerExecutor.put(e.getKey(), e.getValue());
                }
            }
        }

        return SharedTrainingAccumulationTuple.builder().scoreSum(score).updaterStateArray(stateView)
                        .aggregationsCount(aggregationsCount).sparkTrainingStats(stats)
                        .listenerMetaData(listenerMetaData).listenerUpdates(listenerUpdates)
                        .listenerStaticInfo(listenerStaticInfo)
                        .minibatchesPerExecutor(minibatchesPerExecutor).build();
    }
}
