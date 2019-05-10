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
