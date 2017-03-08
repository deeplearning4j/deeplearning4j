package org.deeplearning4j.spark.api.stats;

import org.deeplearning4j.spark.api.worker.ExecuteWorkerFlatMap;
import org.deeplearning4j.spark.api.worker.ExecuteWorkerMultiDataSetFlatMap;
import org.deeplearning4j.spark.stats.BaseEventStats;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.ExampleCountEventStats;
import org.deeplearning4j.spark.time.TimeSource;
import org.deeplearning4j.spark.time.TimeSourceProvider;

import java.util.ArrayList;
import java.util.List;

/**
 * A helper class for collecting stats in {@link ExecuteWorkerFlatMap} and {@link ExecuteWorkerMultiDataSetFlatMap}
 *
 * @author Alex Black
 */
public class StatsCalculationHelper {
    private long methodStartTime;
    private long returnTime;
    private long initalModelBefore;
    private long initialModelAfter;
    private long lastDataSetBefore;
    private long lastProcessBefore;
    private int totalExampleCount;
    private List<EventStats> dataSetGetTimes = new ArrayList<>();
    private List<EventStats> processMiniBatchTimes = new ArrayList<>();

    private TimeSource timeSource = TimeSourceProvider.getInstance();

    public void logMethodStartTime() {
        methodStartTime = timeSource.currentTimeMillis();
    }

    public void logReturnTime() {
        returnTime = timeSource.currentTimeMillis();
    }

    public void logInitialModelBefore() {
        initalModelBefore = timeSource.currentTimeMillis();
    }

    public void logInitialModelAfter() {
        initialModelAfter = timeSource.currentTimeMillis();
    }

    public void logNextDataSetBefore() {
        lastDataSetBefore = timeSource.currentTimeMillis();
    }

    public void logNextDataSetAfter(int numExamples) {
        long now = timeSource.currentTimeMillis();
        long duration = now - lastDataSetBefore;
        dataSetGetTimes.add(new BaseEventStats(lastDataSetBefore, duration));
        totalExampleCount += numExamples;
    }

    public void logProcessMinibatchBefore() {
        lastProcessBefore = timeSource.currentTimeMillis();
    }

    public void logProcessMinibatchAfter() {
        long now = timeSource.currentTimeMillis();
        long duration = now - lastProcessBefore;
        processMiniBatchTimes.add(new BaseEventStats(lastProcessBefore, duration));
    }

    public CommonSparkTrainingStats build(SparkTrainingStats masterSpecificStats) {

        List<EventStats> totalTime = new ArrayList<>();
        totalTime.add(new ExampleCountEventStats(methodStartTime, returnTime - methodStartTime, totalExampleCount));
        List<EventStats> initTime = new ArrayList<>();
        initTime.add(new BaseEventStats(initalModelBefore, initialModelAfter - initalModelBefore));

        return new CommonSparkTrainingStats.Builder().trainingMasterSpecificStats(masterSpecificStats)
                        .workerFlatMapTotalTimeMs(totalTime).workerFlatMapGetInitialModelTimeMs(initTime)
                        .workerFlatMapDataSetGetTimesMs(dataSetGetTimes)
                        .workerFlatMapProcessMiniBatchTimesMs(processMiniBatchTimes).build();
    }
}
