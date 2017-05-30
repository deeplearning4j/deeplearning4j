/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

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
import org.datavec.spark.storage.functions.RecordLoadSequencePairFunction;
import org.datavec.spark.storage.functions.RecordSavePrepPairFunction;
import org.datavec.spark.storage.functions.SequenceRecordSavePrepPairFunction;

import java.util.List;

/**
 * Created by Alex on 30/05/2017.
 */
public class SparkStorageUtils {

    public static void saveSequenceFile(String path, JavaRDD<List<Writable>> rdd){
        path = FilenameUtils.normalize(path, true);
        JavaPairRDD<List<Writable>,Long> dataIndexPairs = rdd.zipWithUniqueId();    //Note: Long values are unique + NOT contiguous; more efficient than zipWithIndex
        JavaPairRDD<LongWritable,RecordWritable> keyedByIndex = dataIndexPairs.mapToPair(new RecordSavePrepPairFunction());

        keyedByIndex.saveAsNewAPIHadoopFile(path, LongWritable.class, RecordWritable.class, SequenceFileOutputFormat.class);
    }

    public static JavaRDD<List<Writable>> restoreSequenceFile(String path, JavaSparkContext sc){
        return restoreMapFile(path, sc).values();
    }

    public static void saveSequenceFileSequences(String path, JavaRDD<List<List<Writable>>> rdd){
        path = FilenameUtils.normalize(path, true);
        JavaPairRDD<List<List<Writable>>,Long> dataIndexPairs = rdd.zipWithUniqueId();    //Note: Long values are unique + NOT contiguous; more efficient than zipWithIndex
        JavaPairRDD<LongWritable,SequenceRecordWritable> keyedByIndex = dataIndexPairs.mapToPair(new SequenceRecordSavePrepPairFunction());

        keyedByIndex.saveAsNewAPIHadoopFile(path, LongWritable.class, SequenceRecordWritable.class, SequenceFileOutputFormat.class);
    }

    public static JavaRDD<List<List<Writable>>> restoreSequenceFileSequences(String path, JavaSparkContext sc){
        return restoreMapFileSequences(path, sc).values();
    }



    public static void saveMapFile(String path, JavaRDD<List<Writable>> rdd){
        path = FilenameUtils.normalize(path, true);
        JavaPairRDD<List<Writable>,Long> dataIndexPairs = rdd.zipWithIndex();   //Note: Long values are unique + contiguous, but requires a count
        JavaPairRDD<LongWritable,RecordWritable> keyedByIndex = dataIndexPairs.mapToPair(new RecordSavePrepPairFunction());

        keyedByIndex.saveAsNewAPIHadoopFile(path, LongWritable.class, RecordWritable.class, MapFileOutputFormat.class);
    }

    public static JavaPairRDD<Long,List<Writable>> restoreMapFile(String path, JavaSparkContext sc){
        Configuration c = new Configuration();
        c.set(FileInputFormat.INPUT_DIR, FilenameUtils.normalize(path, true));
        JavaPairRDD<LongWritable,RecordWritable> pairRDD = sc.newAPIHadoopRDD(c, SequenceFileInputFormat.class, LongWritable.class, RecordWritable.class);

        return pairRDD.mapToPair(new RecordLoadPairFunction());
    }


    public static void saveMapFileSequences(String path, JavaRDD<List<List<Writable>>> rdd){
        path = FilenameUtils.normalize(path, true);
        JavaPairRDD<List<List<Writable>>,Long> dataIndexPairs = rdd.zipWithIndex();
        JavaPairRDD<LongWritable,SequenceRecordWritable> keyedByIndex = dataIndexPairs.mapToPair(new SequenceRecordSavePrepPairFunction());

        keyedByIndex.saveAsNewAPIHadoopFile(path, LongWritable.class, SequenceRecordWritable.class, MapFileOutputFormat.class);
    }

    public static JavaPairRDD<Long,List<List<Writable>>> restoreMapFileSequences(String path, JavaSparkContext sc){
        Configuration c = new Configuration();
        c.set(FileInputFormat.INPUT_DIR, FilenameUtils.normalize(path, true));
        JavaPairRDD<LongWritable,SequenceRecordWritable> pairRDD = sc.newAPIHadoopRDD(c, SequenceFileInputFormat.class, LongWritable.class, SequenceRecordWritable.class);

        return pairRDD.mapToPair(new RecordLoadSequencePairFunction());
    }

}
