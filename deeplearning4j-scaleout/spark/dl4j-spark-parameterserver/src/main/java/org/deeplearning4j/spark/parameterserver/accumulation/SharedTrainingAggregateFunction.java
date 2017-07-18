package org.deeplearning4j.spark.parameterserver.accumulation;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;

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
                            .aggregationsCount(result.getAggregationsCount()).build();
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


        return SharedTrainingAccumulationTuple.builder().scoreSum(score).updaterStateArray(updaterStateSum)
                        .aggregationsCount(aggregationsCount).sparkTrainingStats(stats)
                        .listenerMetaData(listenerMetaData).listenerUpdates(listenerUpdates)
                        .listenerStaticInfo(listenerStaticInfo).build();
    }
}
