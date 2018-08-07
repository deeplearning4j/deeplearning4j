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

import lombok.Data;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkContext;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.StatsUtils;

import java.io.IOException;
import java.util.*;

/**
 * A {@link SparkTrainingStats} implementation for common stats functionality used by most workers
 *
 * @author Alex Black
 */
@Data
public class CommonSparkTrainingStats implements SparkTrainingStats {

    public static final String DEFAULT_DELIMITER = ",";
    public static final String FILENAME_TOTAL_TIME_STATS = "workerFlatMapTotalTimeMs.txt";
    public static final String FILENAME_GET_INITIAL_MODEL_STATS = "workerFlatMapGetInitialModelTimeMs.txt";
    public static final String FILENAME_DATASET_GET_TIME_STATS = "workerFlatMapDataSetGetTimesMs.txt";
    public static final String FILENAME_PROCESS_MINIBATCH_TIME_STATS = "workerFlatMapProcessMiniBatchTimesMs.txt";

    public static final String WORKER_FLAT_MAP_TOTAL_TIME_MS = "WorkerFlatMapTotalTimeMs";
    public static final String WORKER_FLAT_MAP_GET_INITIAL_MODEL_TIME_MS = "WorkerFlatMapGetInitialModelTimeMs";
    public static final String WORKER_FLAT_MAP_DATA_SET_GET_TIMES_MS = "WorkerFlatMapDataSetGetTimesMs";
    public static final String WORKER_FLAT_MAP_PROCESS_MINI_BATCH_TIMES_MS = "WorkerFlatMapProcessMiniBatchTimesMs";
    private static Set<String> columnNames =
                    Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(WORKER_FLAT_MAP_TOTAL_TIME_MS,
                                    WORKER_FLAT_MAP_GET_INITIAL_MODEL_TIME_MS, WORKER_FLAT_MAP_DATA_SET_GET_TIMES_MS,
                                    WORKER_FLAT_MAP_PROCESS_MINI_BATCH_TIMES_MS)));

    private SparkTrainingStats trainingWorkerSpecificStats;
    private List<EventStats> workerFlatMapTotalTimeMs;
    private List<EventStats> workerFlatMapGetInitialModelTimeMs;
    private List<EventStats> workerFlatMapDataSetGetTimesMs;
    private List<EventStats> workerFlatMapProcessMiniBatchTimesMs;



    public CommonSparkTrainingStats() {

    }

    private CommonSparkTrainingStats(Builder builder) {
        this.trainingWorkerSpecificStats = builder.trainingMasterSpecificStats;
        this.workerFlatMapTotalTimeMs = builder.workerFlatMapTotalTimeMs;
        this.workerFlatMapGetInitialModelTimeMs = builder.workerFlatMapGetInitialModelTimeMs;
        this.workerFlatMapDataSetGetTimesMs = builder.workerFlatMapDataSetGetTimesMs;
        this.workerFlatMapProcessMiniBatchTimesMs = builder.workerFlatMapProcessMiniBatchTimesMs;
    }


    @Override
    public Set<String> getKeySet() {
        Set<String> set = new LinkedHashSet<>(columnNames);
        if (trainingWorkerSpecificStats != null)
            set.addAll(trainingWorkerSpecificStats.getKeySet());

        return set;
    }

    @Override
    public List<EventStats> getValue(String key) {
        switch (key) {
            case WORKER_FLAT_MAP_TOTAL_TIME_MS:
                return workerFlatMapTotalTimeMs;
            case WORKER_FLAT_MAP_GET_INITIAL_MODEL_TIME_MS:
                return workerFlatMapGetInitialModelTimeMs;
            case WORKER_FLAT_MAP_DATA_SET_GET_TIMES_MS:
                return workerFlatMapDataSetGetTimesMs;
            case WORKER_FLAT_MAP_PROCESS_MINI_BATCH_TIMES_MS:
                return workerFlatMapProcessMiniBatchTimesMs;
            default:
                if (trainingWorkerSpecificStats != null)
                    return trainingWorkerSpecificStats.getValue(key);
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public String getShortNameForKey(String key) {
        switch (key) {
            case WORKER_FLAT_MAP_TOTAL_TIME_MS:
                return "Total";
            case WORKER_FLAT_MAP_GET_INITIAL_MODEL_TIME_MS:
                return "GetInitModel";
            case WORKER_FLAT_MAP_DATA_SET_GET_TIMES_MS:
                return "GetDataSet";
            case WORKER_FLAT_MAP_PROCESS_MINI_BATCH_TIMES_MS:
                return "ProcessBatch";
            default:
                if (trainingWorkerSpecificStats != null)
                    return trainingWorkerSpecificStats.getShortNameForKey(key);
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public boolean defaultIncludeInPlots(String key) {
        switch (key) {
            case WORKER_FLAT_MAP_TOTAL_TIME_MS:
            case WORKER_FLAT_MAP_GET_INITIAL_MODEL_TIME_MS:
            case WORKER_FLAT_MAP_PROCESS_MINI_BATCH_TIMES_MS:
                return false; //Covered by worker stats generally
            case WORKER_FLAT_MAP_DATA_SET_GET_TIMES_MS:
                return true;
            default:
                if (trainingWorkerSpecificStats != null)
                    return trainingWorkerSpecificStats.defaultIncludeInPlots(key);
                return false;
        }
    }

    @Override
    public void addOtherTrainingStats(SparkTrainingStats other) {
        if (!(other instanceof CommonSparkTrainingStats))
            throw new IllegalArgumentException(
                            "Cannot add other training stats: not an instance of CommonSparkTrainingStats");

        CommonSparkTrainingStats o = (CommonSparkTrainingStats) other;

        workerFlatMapTotalTimeMs.addAll(o.workerFlatMapTotalTimeMs);
        workerFlatMapGetInitialModelTimeMs.addAll(o.workerFlatMapGetInitialModelTimeMs);
        workerFlatMapDataSetGetTimesMs.addAll(o.workerFlatMapDataSetGetTimesMs);
        workerFlatMapProcessMiniBatchTimesMs.addAll(o.workerFlatMapProcessMiniBatchTimesMs);

        if (trainingWorkerSpecificStats != null)
            trainingWorkerSpecificStats.addOtherTrainingStats(o.trainingWorkerSpecificStats);
        else if (o.trainingWorkerSpecificStats != null)
            throw new IllegalStateException(
                            "Cannot merge: training master specific stats is null in one, but not the other");
    }

    @Override
    public SparkTrainingStats getNestedTrainingStats() {
        return trainingWorkerSpecificStats;
    }

    @Override
    public String statsAsString() {
        StringBuilder sb = new StringBuilder();
        String f = SparkTrainingStats.DEFAULT_PRINT_FORMAT;

        sb.append(String.format(f, WORKER_FLAT_MAP_TOTAL_TIME_MS));
        if (workerFlatMapTotalTimeMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(workerFlatMapTotalTimeMs, ",")).append("\n");

        sb.append(String.format(f, WORKER_FLAT_MAP_GET_INITIAL_MODEL_TIME_MS));
        if (workerFlatMapGetInitialModelTimeMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(workerFlatMapGetInitialModelTimeMs, ",")).append("\n");

        sb.append(String.format(f, WORKER_FLAT_MAP_DATA_SET_GET_TIMES_MS));
        if (workerFlatMapDataSetGetTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(workerFlatMapDataSetGetTimesMs, ",")).append("\n");

        sb.append(String.format(f, WORKER_FLAT_MAP_PROCESS_MINI_BATCH_TIMES_MS));
        if (workerFlatMapProcessMiniBatchTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(workerFlatMapProcessMiniBatchTimesMs, ",")).append("\n");

        if (trainingWorkerSpecificStats != null)
            sb.append(trainingWorkerSpecificStats.statsAsString()).append("\n");

        return sb.toString();
    }

    @Override
    public void exportStatFiles(String outputPath, SparkContext sc) throws IOException {
        String d = DEFAULT_DELIMITER;


        //Total time stats (includes total example counts)
        String totalTimeStatsPath = FilenameUtils.concat(outputPath, FILENAME_TOTAL_TIME_STATS);
        StatsUtils.exportStats(workerFlatMapTotalTimeMs, totalTimeStatsPath, d, sc);

        //"Get initial model" stats:
        String getInitialModelStatsPath = FilenameUtils.concat(outputPath, FILENAME_GET_INITIAL_MODEL_STATS);
        StatsUtils.exportStats(workerFlatMapGetInitialModelTimeMs, getInitialModelStatsPath, d, sc);

        //"DataSet get time" stats:
        String getDataSetStatsPath = FilenameUtils.concat(outputPath, FILENAME_DATASET_GET_TIME_STATS);
        StatsUtils.exportStats(workerFlatMapDataSetGetTimesMs, getDataSetStatsPath, d, sc);

        //Process minibatch time stats:
        String processMiniBatchStatsPath = FilenameUtils.concat(outputPath, FILENAME_PROCESS_MINIBATCH_TIME_STATS);
        StatsUtils.exportStats(workerFlatMapProcessMiniBatchTimesMs, processMiniBatchStatsPath, d, sc);

        if (trainingWorkerSpecificStats != null)
            trainingWorkerSpecificStats.exportStatFiles(outputPath, sc);
    }

    public static class Builder {
        private SparkTrainingStats trainingMasterSpecificStats;
        private List<EventStats> workerFlatMapTotalTimeMs;
        private List<EventStats> workerFlatMapGetInitialModelTimeMs;
        private List<EventStats> workerFlatMapDataSetGetTimesMs;
        private List<EventStats> workerFlatMapProcessMiniBatchTimesMs;

        public Builder trainingMasterSpecificStats(SparkTrainingStats trainingMasterSpecificStats) {
            this.trainingMasterSpecificStats = trainingMasterSpecificStats;
            return this;
        }

        public Builder workerFlatMapTotalTimeMs(List<EventStats> workerFlatMapTotalTimeMs) {
            this.workerFlatMapTotalTimeMs = workerFlatMapTotalTimeMs;
            return this;
        }

        public Builder workerFlatMapGetInitialModelTimeMs(List<EventStats> workerFlatMapGetInitialModelTimeMs) {
            this.workerFlatMapGetInitialModelTimeMs = workerFlatMapGetInitialModelTimeMs;
            return this;
        }

        public Builder workerFlatMapDataSetGetTimesMs(List<EventStats> workerFlatMapDataSetGetTimesMs) {
            this.workerFlatMapDataSetGetTimesMs = workerFlatMapDataSetGetTimesMs;
            return this;
        }

        public Builder workerFlatMapProcessMiniBatchTimesMs(List<EventStats> workerFlatMapProcessMiniBatchTimesMs) {
            this.workerFlatMapProcessMiniBatchTimesMs = workerFlatMapProcessMiniBatchTimesMs;
            return this;
        }

        public CommonSparkTrainingStats build() {
            return new CommonSparkTrainingStats(this);
        }
    }
}
