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

package org.datavec.spark.storage;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.MapFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.writable.Writable;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;
import org.datavec.hadoop.records.reader.mapfile.record.SequenceRecordWritable;
import org.datavec.spark.storage.functions.RecordLoadPairFunction;
import org.datavec.spark.storage.functions.RecordSavePrepPairFunction;
import org.datavec.spark.storage.functions.SequenceRecordLoadPairFunction;
import org.datavec.spark.storage.functions.SequenceRecordSavePrepPairFunction;

import java.util.List;

/**
 * Utility methods for saving and restoring Writable objects from Spark RDD is to Hadoop formats
 *
 * @author Alex Black
 */
public class SparkStorageUtils {

    /**
     * Configuration key for the map file interval.
     * This is defined in MapFile.Writer.INDEX_INTERVAL but unfortunately that field is private, hence cannot be
     * referenced here.
     */
    public static final String MAP_FILE_INDEX_INTERVAL_KEY = "io.map.index.interval";

    /**
     * By default, a map file's index stores only a fraction of the keys. This is good, in that it reduces memory
     * requirements (all keys are loaded into memory); however, it has a cost in terms of time taken for look up.
     * Instead of using the default interval of 128, Will use a default interval of 1: given that the keys are LongWritable
     * objects, the marginal increase in space is more than outweighed by the increased performance for use cases such as
     * {@link org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader} and {@link org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader}
     */
    public static final int DEFAULT_MAP_FILE_INTERVAL = 1;

    private SparkStorageUtils() {}

    /**
     * Save a {@code JavaRDD<List<Writable>>} to a Hadoop {@link org.apache.hadoop.io.SequenceFile}. Each record is given
     * a unique (but noncontiguous) {@link LongWritable} key, and values are stored as {@link RecordWritable} instances.
     * <p>
     * Use {@link #restoreSequenceFile(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path Path to save the sequence file
     * @param rdd  RDD to save
     * @see #saveSequenceFileSequences(String, JavaRDD)
     * @see #saveMapFile(String, JavaRDD)
     */
    public static void saveSequenceFile(String path, JavaRDD<List<Writable>> rdd) {
        saveSequenceFile(path, rdd, null);
    }

    /**
     * Save a {@code JavaRDD<List<Writable>>} to a Hadoop {@link org.apache.hadoop.io.SequenceFile}. Each record is given
     * a unique (but noncontiguous) {@link LongWritable} key, and values are stored as {@link RecordWritable} instances.
     * <p>
     * Use {@link #restoreSequenceFile(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path           Path to save the sequence file
     * @param rdd            RDD to save
     * @param maxOutputFiles Nullable. If non-null: first coalesce the RDD to the specified size (number of partitions)
     *                       to limit the maximum number of output sequence files
     * @see #saveSequenceFileSequences(String, JavaRDD)
     * @see #saveMapFile(String, JavaRDD)
     */
    public static void saveSequenceFile(String path, JavaRDD<List<Writable>> rdd,  Integer maxOutputFiles) {
        path = FilenameUtils.normalize(path, true);
        if (maxOutputFiles != null) {
            rdd = rdd.coalesce(maxOutputFiles);
        }
        JavaPairRDD<List<Writable>, Long> dataIndexPairs = rdd.zipWithUniqueId(); //Note: Long values are unique + NOT contiguous; more efficient than zipWithIndex
        JavaPairRDD<LongWritable, RecordWritable> keyedByIndex =
                        dataIndexPairs.mapToPair(new RecordSavePrepPairFunction());

        keyedByIndex.saveAsNewAPIHadoopFile(path, LongWritable.class, RecordWritable.class,
                        SequenceFileOutputFormat.class);
    }

    /**
     * Restore a {@code JavaRDD<List<Writable>>} previously saved with {@link #saveSequenceFile(String, JavaRDD)}
     *
     * @param path Path of the sequence file
     * @param sc   Spark context
     * @return The restored RDD
     */
    public static JavaRDD<List<Writable>> restoreSequenceFile(String path, JavaSparkContext sc) {
        return restoreMapFile(path, sc).values();
    }

    /**
     * Save a {@code JavaRDD<List<List<Writable>>>} to a Hadoop {@link org.apache.hadoop.io.SequenceFile}. Each record
     * is given a unique (but noncontiguous) {@link LongWritable} key, and values are stored as {@link SequenceRecordWritable} instances.
     * <p>
     * Use {@link #restoreSequenceFileSequences(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path Path to save the sequence file
     * @param rdd  RDD to save
     * @see #saveSequenceFile(String, JavaRDD)
     * @see #saveMapFileSequences(String, JavaRDD)
     */
    public static void saveSequenceFileSequences(String path, JavaRDD<List<List<Writable>>> rdd) {
        saveSequenceFileSequences(path, rdd, null);
    }

    /**
     * Save a {@code JavaRDD<List<List<Writable>>>} to a Hadoop {@link org.apache.hadoop.io.SequenceFile}. Each record
     * is given a unique (but noncontiguous) {@link LongWritable} key, and values are stored as {@link SequenceRecordWritable} instances.
     * <p>
     * Use {@link #restoreSequenceFileSequences(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path           Path to save the sequence file
     * @param rdd            RDD to save
     * @param maxOutputFiles Nullable. If non-null: first coalesce the RDD to the specified size (number of partitions)
     *                       to limit the maximum number of output sequence files
     * @see #saveSequenceFile(String, JavaRDD)
     * @see #saveMapFileSequences(String, JavaRDD)
     */
    public static void saveSequenceFileSequences(String path, JavaRDD<List<List<Writable>>> rdd,
                     Integer maxOutputFiles) {
        path = FilenameUtils.normalize(path, true);
        if (maxOutputFiles != null) {
            rdd = rdd.coalesce(maxOutputFiles);
        }
        JavaPairRDD<List<List<Writable>>, Long> dataIndexPairs = rdd.zipWithUniqueId(); //Note: Long values are unique + NOT contiguous; more efficient than zipWithIndex
        JavaPairRDD<LongWritable, SequenceRecordWritable> keyedByIndex =
                        dataIndexPairs.mapToPair(new SequenceRecordSavePrepPairFunction());

        keyedByIndex.saveAsNewAPIHadoopFile(path, LongWritable.class, SequenceRecordWritable.class,
                        SequenceFileOutputFormat.class);
    }

    /**
     * Restore a {@code JavaRDD<List<List<Writable>>} previously saved with {@link #saveSequenceFileSequences(String, JavaRDD)}
     *
     * @param path Path of the sequence file
     * @param sc   Spark context
     * @return The restored RDD
     */
    public static JavaRDD<List<List<Writable>>> restoreSequenceFileSequences(String path, JavaSparkContext sc) {
        return restoreMapFileSequences(path, sc).values();
    }


    /**
     * Save a {@code JavaRDD<List<Writable>>} to a Hadoop {@link org.apache.hadoop.io.MapFile}. Each record is
     * given a <i>unique and contiguous</i> {@link LongWritable} key, and values are stored as
     * {@link RecordWritable} instances.<br>
     * <b>Note 1</b>: If contiguous keys are not required, using a sequence file instead is preferable from a performance
     * point of view. Contiguous keys are often only required for non-Spark use cases, such as with
     * {@link org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader}<br>
     * <b>Note 2</b>: This use a MapFile interval of {@link #DEFAULT_MAP_FILE_INTERVAL}, which is usually suitable for
     * use cases such as {@link org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader}. Use
     * {@link #saveMapFile(String, JavaRDD, int, Integer)} or {@link #saveMapFile(String, JavaRDD, Configuration, Integer)}
     * to customize this. <br>
     * <p>
     * Use {@link #restoreMapFile(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path Path to save the MapFile
     * @param rdd  RDD to save
     * @see #saveMapFileSequences(String, JavaRDD)
     * @see #saveSequenceFile(String, JavaRDD)
     */
    public static void saveMapFile(String path, JavaRDD<List<Writable>> rdd) {
        saveMapFile(path, rdd, DEFAULT_MAP_FILE_INTERVAL, null);
    }

    /**
     * Save a {@code JavaRDD<List<Writable>>} to a Hadoop {@link org.apache.hadoop.io.MapFile}. Each record is
     * given a <i>unique and contiguous</i> {@link LongWritable} key, and values are stored as
     * {@link RecordWritable} instances.<br>
     * <b>Note</b>: If contiguous keys are not required, using a sequence file instead is preferable from a performance
     * point of view. Contiguous keys are often only required for non-Spark use cases, such as with
     * {@link org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader}
     * <p>
     * Use {@link #restoreMapFileSequences(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path           Path to save the MapFile
     * @param rdd            RDD to save
     * @param interval       The map file index interval to use. Smaller values may result in the faster look up, at the
     *                       expense of more memory/disk use. However, usually the increase is relatively minor, due to
     *                       keys being stored as LongWritable objects
     * @param maxOutputFiles Nullable. If non-null: first coalesce the RDD to the specified size (number of partitions)
     *                       to limit the maximum number of output map files
     * @see #saveMapFileSequences(String, JavaRDD)
     * @see #saveSequenceFile(String, JavaRDD)
     */
    public static void saveMapFile(String path, JavaRDD<List<Writable>> rdd, int interval,
                     Integer maxOutputFiles) {
        Configuration c = new Configuration();
        c.set(MAP_FILE_INDEX_INTERVAL_KEY, String.valueOf(interval));
        saveMapFile(path, rdd, c, maxOutputFiles);
    }

    /**
     * Save a {@code JavaRDD<List<Writable>>} to a Hadoop {@link org.apache.hadoop.io.MapFile}. Each record is
     * given a <i>unique and contiguous</i> {@link LongWritable} key, and values are stored as
     * {@link RecordWritable} instances.<br>
     * <b>Note</b>: If contiguous keys are not required, using a sequence file instead is preferable from a performance
     * point of view. Contiguous keys are often only required for non-Spark use cases, such as with
     * {@link org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader}
     * <p>
     * Use {@link #restoreMapFileSequences(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path           Path to save the MapFile
     * @param rdd            RDD to save
     * @param c              Configuration object, used to customise options for the map file
     * @param maxOutputFiles Nullable. If non-null: first coalesce the RDD to the specified size (number of partitions)
     *                       to limit the maximum number of output map files
     * @see #saveMapFileSequences(String, JavaRDD)
     * @see #saveSequenceFile(String, JavaRDD)
     */
    public static void saveMapFile(String path, JavaRDD<List<Writable>> rdd, Configuration c,
                     Integer maxOutputFiles) {
        path = FilenameUtils.normalize(path, true);
        if (maxOutputFiles != null) {
            rdd = rdd.coalesce(maxOutputFiles);
        }
        JavaPairRDD<List<Writable>, Long> dataIndexPairs = rdd.zipWithIndex(); //Note: Long values are unique + contiguous, but requires a count
        JavaPairRDD<LongWritable, RecordWritable> keyedByIndex =
                        dataIndexPairs.mapToPair(new RecordSavePrepPairFunction());

        keyedByIndex.saveAsNewAPIHadoopFile(path, LongWritable.class, RecordWritable.class, MapFileOutputFormat.class,
                        c);
    }

    /**
     * Restore a {@code JavaPairRDD<Long,List<Writable>>} previously saved with {@link #saveMapFile(String, JavaRDD)}}<br>
     * Note that if the keys are not required, simply use {@code restoreMapFile(...).values()}
     *
     * @param path Path of the MapFile
     * @param sc   Spark context
     * @return The restored RDD, with their unique indices as the key
     */
    public static JavaPairRDD<Long, List<Writable>> restoreMapFile(String path, JavaSparkContext sc) {
        Configuration c = new Configuration();
        c.set(FileInputFormat.INPUT_DIR, FilenameUtils.normalize(path, true));
        JavaPairRDD<LongWritable, RecordWritable> pairRDD =
                        sc.newAPIHadoopRDD(c, SequenceFileInputFormat.class, LongWritable.class, RecordWritable.class);

        return pairRDD.mapToPair(new RecordLoadPairFunction());
    }

    /**
     * Save a {@code JavaRDD<List<List<Writable>>>} to a Hadoop {@link org.apache.hadoop.io.MapFile}. Each record is
     * given a <i>unique and contiguous</i> {@link LongWritable} key, and values are stored as
     * {@link SequenceRecordWritable} instances.<br>
     * <b>Note 1</b>: If contiguous keys are not required, using a sequence file instead is preferable from a performance
     * point of view. Contiguous keys are often only required for non-Spark use cases, such as with
     * {@link org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader}<br>
     * <b>Note 2</b>: This use a MapFile interval of {@link #DEFAULT_MAP_FILE_INTERVAL}, which is usually suitable for
     * use cases such as {@link org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader}. Use
     * {@link #saveMapFileSequences(String, JavaRDD, int, Integer)} or {@link #saveMapFileSequences(String, JavaRDD, Configuration, Integer)}
     * to customize this. <br>
     * <p>
     * Use {@link #restoreMapFileSequences(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path Path to save the MapFile
     * @param rdd  RDD to save
     * @see #saveMapFileSequences(String, JavaRDD)
     * @see #saveSequenceFile(String, JavaRDD)
     */
    public static void saveMapFileSequences(String path, JavaRDD<List<List<Writable>>> rdd) {
        saveMapFileSequences(path, rdd, DEFAULT_MAP_FILE_INTERVAL, null);
    }

    /**
     * Save a {@code JavaRDD<List<List<Writable>>>} to a Hadoop {@link org.apache.hadoop.io.MapFile}. Each record is
     * given a <i>unique and contiguous</i> {@link LongWritable} key, and values are stored as
     * {@link SequenceRecordWritable} instances.<br>
     * <b>Note</b>: If contiguous keys are not required, using a sequence file instead is preferable from a performance
     * point of view. Contiguous keys are often only required for non-Spark use cases, such as with
     * {@link org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader}<br>
     * <p>
     * Use {@link #restoreMapFileSequences(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path     Path to save the MapFile
     * @param rdd      RDD to save
     * @param interval The map file index interval to use. Smaller values may result in the faster look up, at the
     *                 expense of more memory/disk use. However, usually the increase is relatively minor, due to
     *                 keys being stored as LongWritable objects
     * @see #saveMapFileSequences(String, JavaRDD)
     * @see #saveSequenceFile(String, JavaRDD)
     */
    public static void saveMapFileSequences(String path, JavaRDD<List<List<Writable>>> rdd, int interval,
                     Integer maxOutputFiles) {
        Configuration c = new Configuration();
        c.set(MAP_FILE_INDEX_INTERVAL_KEY, String.valueOf(interval));
        saveMapFileSequences(path, rdd, c, maxOutputFiles);
    }

    /**
     * Save a {@code JavaRDD<List<List<Writable>>>} to a Hadoop {@link org.apache.hadoop.io.MapFile}. Each record is
     * given a <i>unique and contiguous</i> {@link LongWritable} key, and values are stored as
     * {@link SequenceRecordWritable} instances.<br>
     * <b>Note</b>: If contiguous keys are not required, using a sequence file instead is preferable from a performance
     * point of view. Contiguous keys are often only required for non-Spark use cases, such as with
     * {@link org.datavec.hadoop.records.reader.mapfile.MapFileSequenceRecordReader}<br>
     * <p>
     * Use {@link #restoreMapFileSequences(String, JavaSparkContext)} to restore values saved with this method.
     *
     * @param path Path to save the MapFile
     * @param rdd  RDD to save
     * @param c    Configuration object, used to customise options for the map file
     * @see #saveMapFileSequences(String, JavaRDD)
     * @see #saveSequenceFile(String, JavaRDD)
     */
    public static void saveMapFileSequences(String path, JavaRDD<List<List<Writable>>> rdd, Configuration c,
                     Integer maxOutputFiles) {
        path = FilenameUtils.normalize(path, true);
        if (maxOutputFiles != null) {
            rdd = rdd.coalesce(maxOutputFiles);
        }
        JavaPairRDD<List<List<Writable>>, Long> dataIndexPairs = rdd.zipWithIndex();
        JavaPairRDD<LongWritable, SequenceRecordWritable> keyedByIndex =
                        dataIndexPairs.mapToPair(new SequenceRecordSavePrepPairFunction());

        keyedByIndex.saveAsNewAPIHadoopFile(path, LongWritable.class, SequenceRecordWritable.class,
                        MapFileOutputFormat.class, c);
    }

    /**
     * Restore a {@code JavaPairRDD<Long,List<List<Writable>>>} previously saved with {@link #saveMapFile(String, JavaRDD)}}<br>
     * Note that if the keys are not required, simply use {@code restoreMapFileSequences(...).values()}
     *
     * @param path Path of the MapFile
     * @param sc   Spark context
     * @return The restored RDD, with their unique indices as the key
     */
    public static JavaPairRDD<Long, List<List<Writable>>> restoreMapFileSequences(String path, JavaSparkContext sc) {
        Configuration c = new Configuration();
        c.set(FileInputFormat.INPUT_DIR, FilenameUtils.normalize(path, true));
        JavaPairRDD<LongWritable, SequenceRecordWritable> pairRDD = sc.newAPIHadoopRDD(c, SequenceFileInputFormat.class,
                        LongWritable.class, SequenceRecordWritable.class);

        return pairRDD.mapToPair(new SequenceRecordLoadPairFunction());
    }

}
