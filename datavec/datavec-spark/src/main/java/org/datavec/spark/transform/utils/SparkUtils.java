/*-
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

package org.datavec.spark.transform.utils;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.split.RandomSplit;
import org.datavec.api.transform.split.SplitStrategy;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.writable.*;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by Alex on 7/03/2016.
 */
public class SparkUtils {

    public static <T> List<JavaRDD<T>> splitData(SplitStrategy splitStrategy, JavaRDD<T> data, long seed) {

        if (splitStrategy instanceof RandomSplit) {

            RandomSplit rs = (RandomSplit) splitStrategy;

            double fractionTrain = rs.getFractionTrain();

            double[] splits = new double[] {fractionTrain, 1.0 - fractionTrain};

            JavaRDD<T>[] split = data.randomSplit(splits, seed);
            List<JavaRDD<T>> list = new ArrayList<>(2);
            Collections.addAll(list, split);

            return list;

        } else {
            throw new RuntimeException("Not yet implemented");
        }
    }

    /**
     * Write a String to a file (on HDFS or local) in UTF-8 format
     *
     * @param path       Path to write to
     * @param toWrite    String to write
     * @param sc         Spark context
     */
    public static void writeStringToFile(String path, String toWrite, JavaSparkContext sc) throws IOException {
        writeStringToFile(path, toWrite, sc.sc());
    }

    /**
     * Write a String to a file (on HDFS or local) in UTF-8 format
     *
     * @param path       Path to write to
     * @param toWrite    String to write
     * @param sc         Spark context
     */
    public static void writeStringToFile(String path, String toWrite, SparkContext sc) throws IOException {
        writeStringToFile(path, toWrite, sc.hadoopConfiguration());
    }

    /**
     * Write a String to a file (on HDFS or local) in UTF-8 format
     *
     * @param path         Path to write to
     * @param toWrite      String to write
     * @param hadoopConfig Hadoop configuration, for example from SparkContext.hadoopConfiguration()
     */
    public static void writeStringToFile(String path, String toWrite, Configuration hadoopConfig) throws IOException {
        FileSystem fileSystem = FileSystem.get(hadoopConfig);
        try (BufferedOutputStream bos = new BufferedOutputStream(fileSystem.create(new Path(path)))) {
            bos.write(toWrite.getBytes("UTF-8"));
        }
    }

    /**
     * Read a UTF-8 format String from HDFS (or local)
     *
     * @param path    Path to write the string
     * @param sc      Spark context
     */
    public static String readStringFromFile(String path, JavaSparkContext sc) throws IOException {
        return readStringFromFile(path, sc.sc());
    }

    /**
     * Read a UTF-8 format String from HDFS (or local)
     *
     * @param path    Path to write the string
     * @param sc      Spark context
     */
    public static String readStringFromFile(String path, SparkContext sc) throws IOException {
        return readStringFromFile(path, sc.hadoopConfiguration());
    }

    /**
     * Read a UTF-8 format String from HDFS (or local)
     *
     * @param path         Path to write the string
     * @param hadoopConfig Hadoop configuration, for example from SparkContext.hadoopConfiguration()
     */
    public static String readStringFromFile(String path, Configuration hadoopConfig) throws IOException {
        FileSystem fileSystem = FileSystem.get(hadoopConfig);
        try (BufferedInputStream bis = new BufferedInputStream(fileSystem.open(new Path(path)))) {
            byte[] asBytes = IOUtils.toByteArray(bis);
            return new String(asBytes, "UTF-8");
        }
    }

    /**
     * Write an object to HDFS (or local) using default Java object serialization
     *
     * @param path       Path to write the object to
     * @param toWrite    Object to write
     * @param sc         Spark context
     */
    public static void writeObjectToFile(String path, Object toWrite, JavaSparkContext sc) throws IOException {
        writeObjectToFile(path, toWrite, sc.sc());
    }

    /**
     * Write an object to HDFS (or local) using default Java object serialization
     *
     * @param path       Path to write the object to
     * @param toWrite    Object to write
     * @param sc         Spark context
     */
    public static void writeObjectToFile(String path, Object toWrite, SparkContext sc) throws IOException {
        writeObjectToFile(path, toWrite, sc.hadoopConfiguration());
    }

    /**
     * Write an object to HDFS (or local) using default Java object serialization
     *
     * @param path       Path to write the object to
     * @param toWrite    Object to write
     * @param hadoopConfig Hadoop configuration, for example from SparkContext.hadoopConfiguration()
     */
    public static void writeObjectToFile(String path, Object toWrite, Configuration hadoopConfig) throws IOException {
        FileSystem fileSystem = FileSystem.get(hadoopConfig);
        try (BufferedOutputStream bos = new BufferedOutputStream(fileSystem.create(new Path(path)))) {
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(toWrite);
        }
    }

    /**
     * Read an object from HDFS (or local) using default Java object serialization
     *
     * @param path    File to read
     * @param type    Class of the object to read
     * @param sc      Spark context
     * @param <T>     Type of the object to read
     */
    public static <T> T readObjectFromFile(String path, Class<T> type, JavaSparkContext sc) throws IOException {
        return readObjectFromFile(path, type, sc.sc());
    }

    /**
     * Read an object from HDFS (or local) using default Java object serialization
     *
     * @param path    File to read
     * @param type    Class of the object to read
     * @param sc      Spark context
     * @param <T>     Type of the object to read
     */
    public static <T> T readObjectFromFile(String path, Class<T> type, SparkContext sc) throws IOException {
        return readObjectFromFile(path, type, sc.hadoopConfiguration());
    }

    /**
     * Read an object from HDFS (or local) using default Java object serialization
     *
     * @param path         File to read
     * @param type         Class of the object to read
     * @param hadoopConfig Hadoop configuration, for example from SparkContext.hadoopConfiguration()
     * @param <T>          Type of the object to read
     */
    public static <T> T readObjectFromFile(String path, Class<T> type, Configuration hadoopConfig) throws IOException {
        FileSystem fileSystem = FileSystem.get(hadoopConfig);
        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(fileSystem.open(new Path(path))))) {
            Object o;
            try {
                o = ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }

            return (T) o;
        }
    }

    /**
     * Write a schema to a HDFS (or, local) file in a human-readable format
     *
     * @param outputPath    Output path to write to
     * @param schema        Schema to write
     * @param sc            Spark context
     */
    public static void writeSchema(String outputPath, Schema schema, JavaSparkContext sc) throws IOException {
        writeStringToFile(outputPath, schema.toString(), sc);
    }

    /**
     * Write a DataAnalysis to HDFS (or locally) as a HTML file
     *
     * @param outputPath      Output path
     * @param dataAnalysis    Analysis to generate HTML file for
     * @param sc              Spark context
     */
    public static void writeAnalysisHTMLToFile(String outputPath, DataAnalysis dataAnalysis, JavaSparkContext sc) {
        try {
            String analysisAsHtml = HtmlAnalysis.createHtmlAnalysisString(dataAnalysis);
            writeStringToFile(outputPath, analysisAsHtml, sc);
        } catch (Exception e) {
            throw new RuntimeException("Error generating or writing HTML analysis file (normalized data)", e);
        }
    }

    /**
     * Wlite a set of writables (or, sequence) to HDFS (or, locally).
     *
     * @param outputPath    Path to write the outptu
     * @param delim         Delimiter
     * @param writables     data to write
     * @param sc            Spark context
     */
    public static void writeWritablesToFile(String outputPath, String delim, List<List<Writable>> writables,
                    JavaSparkContext sc) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (List<Writable> list : writables) {
            boolean first = true;
            for (Writable w : list) {
                if (!first)
                    sb.append(delim);
                sb.append(w.toString());
                first = false;
            }
            sb.append("\n");
        }
        writeStringToFile(outputPath, sb.toString(), sc);
    }

    /**
     * Register the DataVec writable classes for Kryo
     */
    public static void registerKryoClasses(SparkConf conf) {
        List<Class<?>> classes = Arrays.<Class<?>>asList(BooleanWritable.class, ByteWritable.class,
                        DoubleWritable.class, FloatWritable.class, IntWritable.class, LongWritable.class,
                        NullWritable.class, Text.class);

        conf.registerKryoClasses((Class<?>[]) classes.toArray());
    }

    public static Class<? extends CompressionCodec> getCompressionCodeClass(String compressionCodecClass) {
        Class<?> tempClass;
        try {
            tempClass = Class.forName(compressionCodecClass);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Invalid class for compression codec: " + compressionCodecClass + " (not found)",
                            e);
        }
        if (!(CompressionCodec.class.isAssignableFrom(tempClass)))
            throw new RuntimeException("Invalid class for compression codec: " + compressionCodecClass
                            + " (not a CompressionCodec)");
        return (Class<? extends CompressionCodec>) tempClass;
    }
}
