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

package org.deeplearning4j.spark.impl.paramavg;

import lombok.extern.slf4j.Slf4j;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.spark.api.*;
import org.deeplearning4j.spark.data.BatchAndExportDataSetsFunction;
import org.deeplearning4j.spark.data.BatchAndExportMultiDataSetsFunction;
import org.deeplearning4j.spark.impl.paramavg.stats.ParameterAveragingTrainingMasterStats;
import org.deeplearning4j.spark.impl.paramavg.util.ExportSupport;
import org.deeplearning4j.spark.util.serde.StorageLevelDeserializer;
import org.deeplearning4j.spark.util.serde.StorageLevelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Collection;
import java.util.Random;

/**
 * @author raver119@gmail.com
 * @author Alex Black
 */
@Slf4j
public abstract class BaseTrainingMaster<R extends TrainingResult, W extends TrainingWorker<R>>
                implements TrainingMaster<R, W> {
    protected static ObjectMapper jsonMapper;
    protected static ObjectMapper yamlMapper;

    protected boolean collectTrainingStats;
    protected ParameterAveragingTrainingMasterStats.ParameterAveragingTrainingMasterStatsHelper stats;

    protected int lastExportedRDDId = Integer.MIN_VALUE;
    protected String lastRDDExportPath;
    protected int batchSizePerWorker;
    protected String exportDirectory = null;
    protected Random rng;

    protected String trainingMasterUID;

    protected StatsStorageRouter statsStorage;

    //Listeners etc
    protected Collection<TrainingListener> listeners;


    protected Repartition repartition;
    protected RepartitionStrategy repartitionStrategy;
    @JsonSerialize(using = StorageLevelSerializer.class)
    @JsonDeserialize(using = StorageLevelDeserializer.class)
    protected StorageLevel storageLevel;
    @JsonSerialize(using = StorageLevelSerializer.class)
    @JsonDeserialize(using = StorageLevelDeserializer.class)
    protected StorageLevel storageLevelStreams = StorageLevel.MEMORY_ONLY();
    protected RDDTrainingApproach rddTrainingApproach = RDDTrainingApproach.Export;

    protected BaseTrainingMaster() {

    }


    protected static synchronized ObjectMapper getJsonMapper() {
        if (jsonMapper == null) {
            jsonMapper = getNewMapper(new JsonFactory());
        }
        return jsonMapper;
    }

    protected static synchronized ObjectMapper getYamlMapper() {
        if (yamlMapper == null) {
            yamlMapper = getNewMapper(new YAMLFactory());
        }
        return yamlMapper;
    }

    protected static ObjectMapper getNewMapper(JsonFactory jsonFactory) {
        ObjectMapper om = new ObjectMapper(jsonFactory);
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        return om;
    }



    protected JavaRDD<String> exportIfRequired(JavaSparkContext sc, JavaRDD<DataSet> trainingData) {
        ExportSupport.assertExportSupported(sc);
        if (collectTrainingStats)
            stats.logExportStart();

        //Two possibilities here:
        // 1. We've seen this RDD before (i.e., multiple epochs training case)
        // 2. We have not seen this RDD before
        //    (a) And we haven't got any stored data -> simply export
        //    (b) And we previously exported some data from a different RDD -> delete the last data
        int currentRDDUid = trainingData.id(); //Id is a "A unique ID for this RDD (within its SparkContext)."

        String baseDir;
        if (lastExportedRDDId == Integer.MIN_VALUE) {
            //Haven't seen a RDD<DataSet> yet in this training master -> export data
            baseDir = export(trainingData);
        } else {
            if (lastExportedRDDId == currentRDDUid) {
                //Use the already-exported data again for another epoch
                baseDir = getBaseDirForRDD(trainingData);
            } else {
                //The new RDD is different to the last one
                // Clean up the data for the last one, and export
                deleteTempDir(sc, lastRDDExportPath);
                baseDir = export(trainingData);
            }
        }

        if (collectTrainingStats)
            stats.logExportEnd();

        return sc.textFile(baseDir + "paths/");
    }

    protected JavaRDD<String> exportIfRequiredMDS(JavaSparkContext sc, JavaRDD<MultiDataSet> trainingData) {
        ExportSupport.assertExportSupported(sc);
        if (collectTrainingStats)
            stats.logExportStart();

        //Two possibilities here:
        // 1. We've seen this RDD before (i.e., multiple epochs training case)
        // 2. We have not seen this RDD before
        //    (a) And we haven't got any stored data -> simply export
        //    (b) And we previously exported some data from a different RDD -> delete the last data
        int currentRDDUid = trainingData.id(); //Id is a "A unique ID for this RDD (within its SparkContext)."

        String baseDir;
        if (lastExportedRDDId == Integer.MIN_VALUE) {
            //Haven't seen a RDD<DataSet> yet in this training master -> export data
            baseDir = exportMDS(trainingData);
        } else {
            if (lastExportedRDDId == currentRDDUid) {
                //Use the already-exported data again for another epoch
                baseDir = getBaseDirForRDD(trainingData);
            } else {
                //The new RDD is different to the last one
                // Clean up the data for the last one, and export
                deleteTempDir(sc, lastRDDExportPath);
                baseDir = exportMDS(trainingData);
            }
        }

        if (collectTrainingStats)
            stats.logExportEnd();

        return sc.textFile(baseDir + "paths/");
    }

    protected String export(JavaRDD<DataSet> trainingData) {
        String baseDir = getBaseDirForRDD(trainingData);
        String dataDir = baseDir + "data/";
        String pathsDir = baseDir + "paths/";

        log.info("Initiating RDD<DataSet> export at {}", baseDir);
        JavaRDD<String> paths = trainingData
                        .mapPartitionsWithIndex(new BatchAndExportDataSetsFunction(batchSizePerWorker, dataDir), true);
        paths.saveAsTextFile(pathsDir);
        log.info("RDD<DataSet> export complete at {}", baseDir);

        lastExportedRDDId = trainingData.id();
        lastRDDExportPath = baseDir;
        return baseDir;
    }

    protected String exportMDS(JavaRDD<MultiDataSet> trainingData) {
        String baseDir = getBaseDirForRDD(trainingData);
        String dataDir = baseDir + "data/";
        String pathsDir = baseDir + "paths/";

        log.info("Initiating RDD<MultiDataSet> export at {}", baseDir);
        JavaRDD<String> paths = trainingData.mapPartitionsWithIndex(
                        new BatchAndExportMultiDataSetsFunction(batchSizePerWorker, dataDir), true);
        paths.saveAsTextFile(pathsDir);
        log.info("RDD<MultiDataSet> export complete at {}", baseDir);

        lastExportedRDDId = trainingData.id();
        lastRDDExportPath = baseDir;
        return baseDir;
    }

    protected String getBaseDirForRDD(JavaRDD<?> rdd) {
        if (exportDirectory == null) {
            exportDirectory = getDefaultExportDirectory(rdd.context());
        }

        return exportDirectory + (exportDirectory.endsWith("/") ? "" : "/") + trainingMasterUID + "/" + rdd.id() + "/";
    }

    protected boolean deleteTempDir(JavaSparkContext sc, String tempDirPath) {
        log.info("Attempting to delete temporary directory: {}", tempDirPath);

        Configuration hadoopConfiguration = sc.hadoopConfiguration();
        FileSystem fileSystem;
        try {
            fileSystem = FileSystem.get(new URI(tempDirPath), hadoopConfiguration);
        } catch (URISyntaxException | IOException e) {
            throw new RuntimeException(e);
        }

        try {
            fileSystem.delete(new Path(tempDirPath), true);
            log.info("Deleted temporary directory: {}", tempDirPath);
            return true;
        } catch (IOException e) {
            log.warn("Could not delete temporary directory: {}", tempDirPath, e);
            return false;
        }
    }

    protected String getDefaultExportDirectory(SparkContext sc) {
        String hadoopTmpDir = sc.hadoopConfiguration().get("hadoop.tmp.dir");
        if (!hadoopTmpDir.endsWith("/") && !hadoopTmpDir.endsWith("\\"))
            hadoopTmpDir = hadoopTmpDir + "/";
        return hadoopTmpDir + "dl4j/";
    }


    @Override
    public boolean deleteTempFiles(JavaSparkContext sc) {
        return lastRDDExportPath == null || deleteTempDir(sc, lastRDDExportPath);
    }

    @Override
    public boolean deleteTempFiles(SparkContext sc) {
        return deleteTempFiles(new JavaSparkContext(sc));
    }


}
