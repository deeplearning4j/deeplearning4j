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

package org.deeplearning4j.spark.impl.paramavg.stats;

import lombok.Data;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkContext;
import org.deeplearning4j.spark.api.stats.CommonSparkTrainingStats;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.stats.*;
import org.deeplearning4j.spark.time.TimeSource;
import org.deeplearning4j.spark.time.TimeSourceProvider;

import java.io.IOException;
import java.util.*;

/**
 * Statistics collected by a {@link org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster}
 *
 * @author Alex Black
 */
@Data
public class ParameterAveragingTrainingMasterStats implements SparkTrainingStats {

    public static final String DEFAULT_DELIMITER = CommonSparkTrainingStats.DEFAULT_DELIMITER;
    public static final String FILENAME_EXPORT_RDD_TIME = "parameterAveragingMasterExportTimesMs.txt";
    public static final String FILENAME_COUNT_RDD_SIZE = "parameterAveragingMasterCountRddSizeTimesMs.txt";
    public static final String FILENAME_BROADCAST_CREATE = "parameterAveragingMasterBroadcastCreateTimesMs.txt";
    public static final String FILENAME_FIT_TIME = "parameterAveragingMasterFitTimesMs.txt";
    public static final String FILENAME_SPLIT_TIME = "parameterAveragingMasterSplitTimesMs.txt";
    public static final String FILENAME_MAP_PARTITIONS_TIME = "parameterAveragingMasterMapPartitionsTimesMs.txt";
    public static final String FILENAME_AGGREGATE_TIME = "parameterAveragingMasterAggregateTimesMs.txt";
    public static final String FILENAME_PROCESS_PARAMS_TIME = "parameterAveragingMasterProcessParamsUpdaterTimesMs.txt";
    public static final String FILENAME_REPARTITION_STATS = "parameterAveragingMasterRepartitionTimesMs.txt";

    public static final String PARAMETER_AVERAGING_MASTER_EXPORT_RDD_TIMES_MS = "parameterAveragingMasterExportTimesMs";
    public static final String PARAMETER_AVERAGING_MASTER_COUNT_RDD_TIMES_MS =
                    "ParameterAveragingMasterCountRddSizeTimesMs";
    public static final String PARAMETER_AVERAGING_MASTER_BROADCAST_CREATE_TIMES_MS =
                    "ParameterAveragingMasterBroadcastCreateTimesMs";
    public static final String PARAMETER_AVERAGING_MASTER_FIT_TIMES_MS = "ParameterAveragingMasterFitTimesMs";
    public static final String PARAMETER_AVERAGING_MASTER_SPLIT_TIMES_MS = "ParameterAveragingMasterSplitTimesMs";
    public static final String PARAMETER_AVERAGING_MASTER_MAP_PARTITIONS_TIMES_MS =
                    "ParameterAveragingMasterMapPartitionsTimesMs";
    public static final String PARAMETER_AVERAGING_MASTER_AGGREGATE_TIMES_MS =
                    "ParameterAveragingMasterAggregateTimesMs";
    public static final String PARAMETER_AVERAGING_MASTER_PROCESS_PARAMS_UPDATER_TIMES_MS =
                    "ParameterAveragingMasterProcessParamsUpdaterTimesMs";
    public static final String PARAMETER_AVERAGING_MASTER_REPARTITION_TIMES_MS =
                    "ParameterAveragingMasterRepartitionTimesMs";

    private static Set<String> columnNames = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
                    PARAMETER_AVERAGING_MASTER_EXPORT_RDD_TIMES_MS, PARAMETER_AVERAGING_MASTER_COUNT_RDD_TIMES_MS,
                    PARAMETER_AVERAGING_MASTER_BROADCAST_CREATE_TIMES_MS, PARAMETER_AVERAGING_MASTER_FIT_TIMES_MS,
                    PARAMETER_AVERAGING_MASTER_SPLIT_TIMES_MS, PARAMETER_AVERAGING_MASTER_MAP_PARTITIONS_TIMES_MS,
                    PARAMETER_AVERAGING_MASTER_AGGREGATE_TIMES_MS,
                    PARAMETER_AVERAGING_MASTER_PROCESS_PARAMS_UPDATER_TIMES_MS,
                    PARAMETER_AVERAGING_MASTER_REPARTITION_TIMES_MS)));

    private SparkTrainingStats workerStats;
    private List<EventStats> parameterAveragingMasterExportTimesMs;
    private List<EventStats> parameterAveragingMasterCountRddSizeTimesMs;
    private List<EventStats> parameterAveragingMasterBroadcastCreateTimesMs;
    private List<EventStats> parameterAveragingMasterFitTimesMs;
    private List<EventStats> parameterAveragingMasterSplitTimesMs;
    private List<EventStats> parameterAveragingMasterMapPartitionsTimesMs;
    private List<EventStats> paramaterAveragingMasterAggregateTimesMs;
    private List<EventStats> parameterAveragingMasterProcessParamsUpdaterTimesMs;
    private List<EventStats> parameterAveragingMasterRepartitionTimesMs;


    public ParameterAveragingTrainingMasterStats(SparkTrainingStats workerStats,
                    List<EventStats> parameterAveragingMasterExportTimesMs,
                    List<EventStats> parameterAveragingMasterCountRddSizeTimesMs,
                    List<EventStats> parameterAveragingMasterBroadcastCreateTimeMs,
                    List<EventStats> parameterAveragingMasterFitTimeMs,
                    List<EventStats> parameterAveragingMasterSplitTimeMs,
                    List<EventStats> parameterAveragingMasterMapPartitionsTimesMs,
                    List<EventStats> parameterAveragingMasterAggregateTimesMs,
                    List<EventStats> parameterAveragingMasterProcessParamsUpdaterTimesMs,
                    List<EventStats> parameterAveragingMasterRepartitionTimesMs) {
        this.workerStats = workerStats;
        this.parameterAveragingMasterExportTimesMs = parameterAveragingMasterExportTimesMs;
        this.parameterAveragingMasterCountRddSizeTimesMs = parameterAveragingMasterCountRddSizeTimesMs;
        this.parameterAveragingMasterBroadcastCreateTimesMs = parameterAveragingMasterBroadcastCreateTimeMs;
        this.parameterAveragingMasterFitTimesMs = parameterAveragingMasterFitTimeMs;
        this.parameterAveragingMasterSplitTimesMs = parameterAveragingMasterSplitTimeMs;
        this.parameterAveragingMasterMapPartitionsTimesMs = parameterAveragingMasterMapPartitionsTimesMs;
        this.paramaterAveragingMasterAggregateTimesMs = parameterAveragingMasterAggregateTimesMs;
        this.parameterAveragingMasterProcessParamsUpdaterTimesMs = parameterAveragingMasterProcessParamsUpdaterTimesMs;
        this.parameterAveragingMasterRepartitionTimesMs = parameterAveragingMasterRepartitionTimesMs;
    }


    @Override
    public Set<String> getKeySet() {
        Set<String> out = new LinkedHashSet<>(columnNames);
        if (workerStats != null)
            out.addAll(workerStats.getKeySet());
        return out;
    }

    @Override
    public List<EventStats> getValue(String key) {
        switch (key) {
            case PARAMETER_AVERAGING_MASTER_EXPORT_RDD_TIMES_MS:
                return parameterAveragingMasterExportTimesMs;
            case PARAMETER_AVERAGING_MASTER_COUNT_RDD_TIMES_MS:
                return parameterAveragingMasterCountRddSizeTimesMs;
            case PARAMETER_AVERAGING_MASTER_BROADCAST_CREATE_TIMES_MS:
                return parameterAveragingMasterBroadcastCreateTimesMs;
            case PARAMETER_AVERAGING_MASTER_FIT_TIMES_MS:
                return parameterAveragingMasterFitTimesMs;
            case PARAMETER_AVERAGING_MASTER_SPLIT_TIMES_MS:
                return parameterAveragingMasterSplitTimesMs;
            case PARAMETER_AVERAGING_MASTER_MAP_PARTITIONS_TIMES_MS:
                return parameterAveragingMasterMapPartitionsTimesMs;
            case PARAMETER_AVERAGING_MASTER_AGGREGATE_TIMES_MS:
                return paramaterAveragingMasterAggregateTimesMs;
            case PARAMETER_AVERAGING_MASTER_PROCESS_PARAMS_UPDATER_TIMES_MS:
                return parameterAveragingMasterProcessParamsUpdaterTimesMs;
            case PARAMETER_AVERAGING_MASTER_REPARTITION_TIMES_MS:
                return parameterAveragingMasterRepartitionTimesMs;
            default:
                if (workerStats != null)
                    return workerStats.getValue(key);
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public String getShortNameForKey(String key) {
        switch (key) {
            case PARAMETER_AVERAGING_MASTER_EXPORT_RDD_TIMES_MS:
                return "Export";
            case PARAMETER_AVERAGING_MASTER_COUNT_RDD_TIMES_MS:
                return "CountRDD";
            case PARAMETER_AVERAGING_MASTER_BROADCAST_CREATE_TIMES_MS:
                return "CreateBroadcast";
            case PARAMETER_AVERAGING_MASTER_FIT_TIMES_MS:
                return "Fit";
            case PARAMETER_AVERAGING_MASTER_SPLIT_TIMES_MS:
                return "Split";
            case PARAMETER_AVERAGING_MASTER_MAP_PARTITIONS_TIMES_MS:
                return "MapPart";
            case PARAMETER_AVERAGING_MASTER_AGGREGATE_TIMES_MS:
                return "Aggregate";
            case PARAMETER_AVERAGING_MASTER_PROCESS_PARAMS_UPDATER_TIMES_MS:
                return "ProcessParams";
            case PARAMETER_AVERAGING_MASTER_REPARTITION_TIMES_MS:
                return "Repartition";
            default:
                if (workerStats != null)
                    return workerStats.getShortNameForKey(key);
                throw new IllegalArgumentException("Unknown key: \"" + key + "\"");
        }
    }

    @Override
    public boolean defaultIncludeInPlots(String key) {
        switch (key) {
            case PARAMETER_AVERAGING_MASTER_FIT_TIMES_MS:
            case PARAMETER_AVERAGING_MASTER_MAP_PARTITIONS_TIMES_MS:
                return false;
            case PARAMETER_AVERAGING_MASTER_EXPORT_RDD_TIMES_MS:
            case PARAMETER_AVERAGING_MASTER_COUNT_RDD_TIMES_MS:
            case PARAMETER_AVERAGING_MASTER_SPLIT_TIMES_MS:
            case PARAMETER_AVERAGING_MASTER_BROADCAST_CREATE_TIMES_MS:
            case PARAMETER_AVERAGING_MASTER_AGGREGATE_TIMES_MS:
            case PARAMETER_AVERAGING_MASTER_PROCESS_PARAMS_UPDATER_TIMES_MS:
            case PARAMETER_AVERAGING_MASTER_REPARTITION_TIMES_MS:
                return true;
            default:
                if (workerStats != null)
                    return workerStats.defaultIncludeInPlots(key);
                return false;
        }
    }

    @Override
    public void addOtherTrainingStats(SparkTrainingStats other) {
        if (!(other instanceof ParameterAveragingTrainingMasterStats))
            throw new IllegalArgumentException("Expected ParameterAveragingTrainingMasterStats, got "
                            + (other != null ? other.getClass() : null));

        ParameterAveragingTrainingMasterStats o = (ParameterAveragingTrainingMasterStats) other;

        if (workerStats != null) {
            if (o.workerStats != null)
                workerStats.addOtherTrainingStats(o.workerStats);
        } else {
            if (o.workerStats != null)
                workerStats = o.workerStats;
        }

        this.parameterAveragingMasterExportTimesMs.addAll(o.parameterAveragingMasterExportTimesMs);
        this.parameterAveragingMasterCountRddSizeTimesMs.addAll(o.parameterAveragingMasterCountRddSizeTimesMs);
        this.parameterAveragingMasterBroadcastCreateTimesMs.addAll(o.parameterAveragingMasterBroadcastCreateTimesMs);
        this.parameterAveragingMasterRepartitionTimesMs.addAll(o.parameterAveragingMasterRepartitionTimesMs);
        this.parameterAveragingMasterFitTimesMs.addAll(o.parameterAveragingMasterFitTimesMs);
        if (parameterAveragingMasterRepartitionTimesMs == null) {
            if (o.parameterAveragingMasterRepartitionTimesMs != null)
                parameterAveragingMasterRepartitionTimesMs = o.parameterAveragingMasterRepartitionTimesMs;
        } else {
            if (o.parameterAveragingMasterRepartitionTimesMs != null)
                parameterAveragingMasterRepartitionTimesMs.addAll(o.parameterAveragingMasterRepartitionTimesMs);
        }
    }

    @Override
    public SparkTrainingStats getNestedTrainingStats() {
        return workerStats;
    }

    @Override
    public String statsAsString() {
        StringBuilder sb = new StringBuilder();
        String f = SparkTrainingStats.DEFAULT_PRINT_FORMAT;

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_EXPORT_RDD_TIMES_MS));
        if (parameterAveragingMasterExportTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterExportTimesMs, ",")).append("\n");

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_COUNT_RDD_TIMES_MS));
        if (parameterAveragingMasterCountRddSizeTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterCountRddSizeTimesMs, ",")).append("\n");

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_BROADCAST_CREATE_TIMES_MS));
        if (parameterAveragingMasterBroadcastCreateTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterBroadcastCreateTimesMs, ",")).append("\n");

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_REPARTITION_TIMES_MS));
        if (parameterAveragingMasterRepartitionTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterRepartitionTimesMs, ",")).append("\n");

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_FIT_TIMES_MS));
        if (parameterAveragingMasterFitTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterFitTimesMs, ",")).append("\n");

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_SPLIT_TIMES_MS));
        if (parameterAveragingMasterSplitTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterSplitTimesMs, ",")).append("\n");

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_MAP_PARTITIONS_TIMES_MS));
        if (parameterAveragingMasterMapPartitionsTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterMapPartitionsTimesMs, ",")).append("\n");

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_AGGREGATE_TIMES_MS));
        if (paramaterAveragingMasterAggregateTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(paramaterAveragingMasterAggregateTimesMs, ",")).append("\n");

        sb.append(String.format(f, PARAMETER_AVERAGING_MASTER_PROCESS_PARAMS_UPDATER_TIMES_MS));
        if (parameterAveragingMasterProcessParamsUpdaterTimesMs == null)
            sb.append("-\n");
        else
            sb.append(StatsUtils.getDurationAsString(parameterAveragingMasterProcessParamsUpdaterTimesMs, ","))
                            .append("\n");

        if (workerStats != null)
            sb.append(workerStats.statsAsString());

        return sb.toString();
    }

    @Override
    public void exportStatFiles(String outputPath, SparkContext sc) throws IOException {
        String d = DEFAULT_DELIMITER;

        //Export times
        String exportRddPath = FilenameUtils.concat(outputPath, FILENAME_EXPORT_RDD_TIME);
        StatsUtils.exportStats(parameterAveragingMasterExportTimesMs, exportRddPath, d, sc);

        //Count RDD times:
        String countRddPath = FilenameUtils.concat(outputPath, FILENAME_COUNT_RDD_SIZE);
        StatsUtils.exportStats(parameterAveragingMasterCountRddSizeTimesMs, countRddPath, d, sc);

        //broadcast create time:
        String broadcastTimePath = FilenameUtils.concat(outputPath, FILENAME_BROADCAST_CREATE);
        StatsUtils.exportStats(parameterAveragingMasterBroadcastCreateTimesMs, broadcastTimePath, d, sc);

        //repartition
        String repartitionTime = FilenameUtils.concat(outputPath, FILENAME_REPARTITION_STATS);
        StatsUtils.exportStats(parameterAveragingMasterRepartitionTimesMs, repartitionTime, d, sc);

        //Fit time:
        String fitTimePath = FilenameUtils.concat(outputPath, FILENAME_FIT_TIME);
        StatsUtils.exportStats(parameterAveragingMasterFitTimesMs, fitTimePath, d, sc);

        //Split time:
        String splitTimePath = FilenameUtils.concat(outputPath, FILENAME_SPLIT_TIME);
        StatsUtils.exportStats(parameterAveragingMasterSplitTimesMs, splitTimePath, d, sc);

        //Map partitions:
        String mapPartitionsPath = FilenameUtils.concat(outputPath, FILENAME_MAP_PARTITIONS_TIME);
        StatsUtils.exportStats(parameterAveragingMasterMapPartitionsTimesMs, mapPartitionsPath, d, sc);

        //Aggregate time:
        String aggregatePath = FilenameUtils.concat(outputPath, FILENAME_AGGREGATE_TIME);
        StatsUtils.exportStats(paramaterAveragingMasterAggregateTimesMs, aggregatePath, d, sc);

        //broadcast create time:
        String processParamsPath = FilenameUtils.concat(outputPath, FILENAME_PROCESS_PARAMS_TIME);
        StatsUtils.exportStats(parameterAveragingMasterProcessParamsUpdaterTimesMs, processParamsPath, d, sc);

        //Repartition
        if (parameterAveragingMasterRepartitionTimesMs != null) {
            String repartitionPath = FilenameUtils.concat(outputPath, FILENAME_REPARTITION_STATS);
            StatsUtils.exportStats(parameterAveragingMasterRepartitionTimesMs, repartitionPath, d, sc);
        }

        if (workerStats != null)
            workerStats.exportStatFiles(outputPath, sc);
    }

    public static class ParameterAveragingTrainingMasterStatsHelper {

        private long lastExportStartTime;
        private long lastCountStartTime;
        private long lastBroadcastStartTime;
        private long lastRepartitionStartTime;
        private long lastFitStartTime;
        private long lastSplitStartTime;
        private long lastMapPartitionsStartTime;
        private long lastAggregateStartTime;
        private long lastProcessParamsUpdaterStartTime;

        private SparkTrainingStats workerStats;

        private List<EventStats> exportTimes = new ArrayList<>(); //Starts for exporting data
        private List<EventStats> countTimes = new ArrayList<>();
        private List<EventStats> broadcastTimes = new ArrayList<>();
        private List<EventStats> repartitionTimes = new ArrayList<>();
        private List<EventStats> fitTimes = new ArrayList<>();
        private List<EventStats> splitTimes = new ArrayList<>();
        private List<EventStats> mapPartitions = new ArrayList<>();
        private List<EventStats> aggregateTimes = new ArrayList<>();
        private List<EventStats> processParamsUpdaterTimes = new ArrayList<>();

        private final TimeSource timeSource = TimeSourceProvider.getInstance();

        public void logExportStart() {
            this.lastExportStartTime = timeSource.currentTimeMillis();
        }

        public void logExportEnd() {
            long now = timeSource.currentTimeMillis();

            exportTimes.add(new BaseEventStats(lastExportStartTime, now - lastExportStartTime));
        }

        public void logCountStart() {
            this.lastCountStartTime = timeSource.currentTimeMillis();
        }

        public void logCountEnd() {
            long now = timeSource.currentTimeMillis();

            countTimes.add(new BaseEventStats(lastCountStartTime, now - lastCountStartTime));
        }

        public void logBroadcastStart() {
            this.lastBroadcastStartTime = timeSource.currentTimeMillis();
        }

        public void logBroadcastEnd() {
            long now = timeSource.currentTimeMillis();

            broadcastTimes.add(new BaseEventStats(lastBroadcastStartTime, now - lastBroadcastStartTime));
        }

        public void logRepartitionStart() {
            lastRepartitionStartTime = timeSource.currentTimeMillis();
        }

        public void logRepartitionEnd() {
            long now = timeSource.currentTimeMillis();
            repartitionTimes.add(new BaseEventStats(lastRepartitionStartTime, now - lastRepartitionStartTime));
        }

        public void logFitStart() {
            lastFitStartTime = timeSource.currentTimeMillis();
        }

        public void logFitEnd(int examplesCount) {
            long now = timeSource.currentTimeMillis();
            fitTimes.add(new ExampleCountEventStats(lastFitStartTime, now - lastFitStartTime, examplesCount));
        }

        public void logSplitStart() {
            lastSplitStartTime = timeSource.currentTimeMillis();
        }

        public void logSplitEnd() {
            long now = timeSource.currentTimeMillis();
            splitTimes.add(new BaseEventStats(lastSplitStartTime, now - lastSplitStartTime));
        }

        public void logMapPartitionsStart() {
            lastMapPartitionsStartTime = timeSource.currentTimeMillis();
        }

        public void logMapPartitionsEnd(int nPartitions) {
            long now = timeSource.currentTimeMillis();
            mapPartitions.add(new PartitionCountEventStats(lastMapPartitionsStartTime,
                            (now - lastMapPartitionsStartTime), nPartitions));
        }

        public void logAggregateStartTime() {
            lastAggregateStartTime = timeSource.currentTimeMillis();
        }

        public void logAggregationEndTime() {
            long now = timeSource.currentTimeMillis();
            aggregateTimes.add(new BaseEventStats(lastAggregateStartTime, now - lastAggregateStartTime));
        }

        public void logProcessParamsUpdaterStart() {
            lastProcessParamsUpdaterStartTime = timeSource.currentTimeMillis();
        }

        public void logProcessParamsUpdaterEnd() {
            long now = timeSource.currentTimeMillis();
            processParamsUpdaterTimes.add(new BaseEventStats(lastProcessParamsUpdaterStartTime,
                            now - lastProcessParamsUpdaterStartTime));
        }

        public void addWorkerStats(SparkTrainingStats workerStats) {
            if (this.workerStats == null)
                this.workerStats = workerStats;
            else if (workerStats != null)
                this.workerStats.addOtherTrainingStats(workerStats);
        }

        public ParameterAveragingTrainingMasterStats build() {
            return new ParameterAveragingTrainingMasterStats(workerStats, exportTimes, countTimes, broadcastTimes,
                            fitTimes, splitTimes, mapPartitions, aggregateTimes, processParamsUpdaterTimes,
                            repartitionTimes);
        }

    }

}
