package org.deeplearning4j.spark.impl.paramavg.aggregator;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;

/**
 * Add function for parameter averaging
 *
 * @author Alex Black
 */
public class ParameterAveragingElementAddFunction implements
                Function2<ParameterAveragingAggregationTuple, ParameterAveragingTrainingResult, ParameterAveragingAggregationTuple> {

    @Override
    public ParameterAveragingAggregationTuple call(ParameterAveragingAggregationTuple tuple,
                    ParameterAveragingTrainingResult result) throws Exception {
        if (tuple == null) {
            return ParameterAveragingAggregationTuple.builder().parametersSum(result.getParameters())
                            .updaterStateSum(result.getUpdaterState()).scoreSum(result.getScore()).aggregationsCount(1)
                            .sparkTrainingStats(result.getSparkTrainingStats())
                            .listenerMetaData(result.getListenerMetaData())
                            .listenerStaticInfo(result.getListenerStaticInfo())
                            .listenerUpdates(result.getListenerUpdates()).build();
        }

        INDArray params = tuple.getParametersSum().addi(result.getParameters());
        INDArray updaterStateSum;
        if (tuple.getUpdaterStateSum() == null) {
            updaterStateSum = result.getUpdaterState();
        } else {
            updaterStateSum = tuple.getUpdaterStateSum();
            if (result.getUpdaterState() != null)
                updaterStateSum.addi(result.getUpdaterState());
        }

        double scoreSum = tuple.getScoreSum() + result.getScore();
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
            Collection<Persistable> newStatic = tuple.getListenerStaticInfo();
            if (newStatic != null)
                listenerStaticInfo.addAll(newStatic);
        }

        Collection<Persistable> listenerUpdates = tuple.getListenerUpdates();
        if (listenerUpdates == null)
            listenerUpdates = result.getListenerUpdates();
        else {
            Collection<Persistable> newUpdates = result.getListenerUpdates();
            if (newUpdates != null)
                listenerUpdates.addAll(newUpdates);
        }



        return new ParameterAveragingAggregationTuple(params, updaterStateSum, scoreSum,
                        tuple.getAggregationsCount() + 1, stats, listenerMetaData, listenerStaticInfo, listenerUpdates);
    }
}
