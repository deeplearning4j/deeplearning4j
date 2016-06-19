package org.deeplearning4j.spark.api.worker;

import org.deeplearning4j.spark.api.stats.CommonSparkTrainingStats;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;

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
    //TODO: This adds more overhead than we want. Replace with a fast int collection (no boxing + conversion!)
    private List<Integer> dataSetGetTimes = new ArrayList<>();
    private List<Integer> processMiniBatchTimes = new ArrayList<>();

    public void logMethodStartTime(){
        methodStartTime = System.currentTimeMillis();
    }

    public void logReturnTime(){
        returnTime = System.currentTimeMillis();
    }

    public void logInitialModelBefore(){
        initalModelBefore = System.currentTimeMillis();
    }

    public void logInitialModelAfter(){
        initialModelAfter = System.currentTimeMillis();
    }

    public void logNextDataSetBefore(){
        lastDataSetBefore = System.currentTimeMillis();
    }

    public void logNextDataSetAfter(int numExamples){
        long now = System.currentTimeMillis();
        dataSetGetTimes.add((int)(now-lastDataSetBefore));
        totalExampleCount += numExamples;
    }

    public void logProcessMinibatchBefore(){
        lastProcessBefore = System.currentTimeMillis();
    }

    public void logProcessMinibatchAfter(){
        long now = System.currentTimeMillis();
        processMiniBatchTimes.add((int)(now-lastProcessBefore));
    }

    public CommonSparkTrainingStats build(SparkTrainingStats masterSpecificStats){
        //TODO again, do this without the lists...
        int[] dataSetGetTimesArr = new int[dataSetGetTimes.size()];
        for( int i=0; i<dataSetGetTimesArr.length; i++ ) dataSetGetTimesArr[i] = dataSetGetTimes.get(i);
        int[] processMiniBatchTimesArr = new int[processMiniBatchTimes.size()];
        for( int i=0; i<processMiniBatchTimesArr.length; i++ ) processMiniBatchTimesArr[i] = processMiniBatchTimes.get(i);

        return new CommonSparkTrainingStats.Builder()
                .trainingMasterSpecificStats(masterSpecificStats)
                .workerFlatMapTotalTimeMs((int)(returnTime-methodStartTime))
                .workerFlatMapTotalExampleCount(totalExampleCount)
                .workerFlatMapGetInitialModelTimeMs((int)(initialModelAfter-initalModelBefore))
                .workerFlatMapDataSetGetTimesMs(dataSetGetTimesArr)
                .workerFlatMapProcessMiniBatchTimesMs(processMiniBatchTimesArr)
                .workerFlatMapCountNoDataInstances(dataSetGetTimes.size() == 0 ? 1 : 0)
                .build();
    }
}
