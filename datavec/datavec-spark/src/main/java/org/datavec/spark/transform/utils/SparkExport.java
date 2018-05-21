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

import lombok.AllArgsConstructor;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.WritablesToStringFunction;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by Alex on 7/03/2016.
 */
public class SparkExport {

    //Quick and dirty CSV export (using Spark). Eventually, rework this to use DataVec record writers on Spark
    public static void exportCSVSpark(String directory, String delimiter, int outputSplits,
                    JavaRDD<List<Writable>> data) {
        exportCSVSpark(directory, delimiter, null, outputSplits, data);
    }

    public static void exportCSVSpark(String directory, String delimiter, String quote, int outputSplits,
                    JavaRDD<List<Writable>> data) {

        //NOTE: Order is probably not random here...
        JavaRDD<String> lines = data.map(new WritablesToStringFunction(delimiter, quote));
        lines.coalesce(outputSplits);

        lines.saveAsTextFile(directory);
    }

    //Another quick and dirty CSV export (local). Dumps all values into a single file
    public static void exportCSVLocal(File outputFile, String delimiter, JavaRDD<List<Writable>> data, int rngSeed)
                    throws Exception {
        exportCSVLocal(outputFile, delimiter, null, data, rngSeed);
    }

    public static void exportCSVLocal(File outputFile, String delimiter, String quote, JavaRDD<List<Writable>> data,
                    int rngSeed) throws Exception {

        JavaRDD<String> lines = data.map(new WritablesToStringFunction(delimiter, quote));
        List<String> linesList = lines.collect(); //Requires all data in memory
        if (!(linesList instanceof ArrayList))
            linesList = new ArrayList<>(linesList);
        Collections.shuffle(linesList, new Random(rngSeed));

        FileUtils.writeLines(outputFile, linesList);
    }

    //Another quick and dirty CSV export (local). Dumps all values into multiple files (specified number of files)
    public static void exportCSVLocal(String outputDir, String baseFileName, int numFiles, String delimiter,
                    JavaRDD<List<Writable>> data, int rngSeed) throws Exception {
        exportCSVLocal(outputDir, baseFileName, numFiles, delimiter, null, data, rngSeed);
    }

    public static void exportCSVLocal(String outputDir, String baseFileName, int numFiles, String delimiter,
                    String quote, JavaRDD<List<Writable>> data, int rngSeed) throws Exception {

        JavaRDD<String> lines = data.map(new WritablesToStringFunction(delimiter, quote));
        double[] split = new double[numFiles];
        for (int i = 0; i < split.length; i++)
            split[i] = 1.0 / numFiles;
        JavaRDD<String>[] splitData = lines.randomSplit(split);

        int count = 0;
        Random r = new Random(rngSeed);
        for (JavaRDD<String> subset : splitData) {
            String path = FilenameUtils.concat(outputDir, baseFileName + (count++) + ".csv");
            List<String> linesList = subset.collect();
            if (!(linesList instanceof ArrayList))
                linesList = new ArrayList<>(linesList);
            Collections.shuffle(linesList, r);
            FileUtils.writeLines(new File(path), linesList);
        }
    }

    // No shuffling
    public static void exportCSVLocal(String outputDir, String baseFileName, int numFiles, String delimiter,
                    JavaRDD<List<Writable>> data) throws Exception {
        exportCSVLocal(outputDir, baseFileName, numFiles, delimiter, null, data);
    }

    public static void exportCSVLocal(String outputDir, String baseFileName, int numFiles, String delimiter,
                    String quote, JavaRDD<List<Writable>> data) throws Exception {

        JavaRDD<String> lines = data.map(new WritablesToStringFunction(delimiter, quote));
        double[] split = new double[numFiles];
        for (int i = 0; i < split.length; i++)
            split[i] = 1.0 / numFiles;
        JavaRDD<String>[] splitData = lines.randomSplit(split);

        int count = 0;
        for (JavaRDD<String> subset : splitData) {
            String path = FilenameUtils.concat(outputDir, baseFileName + (count++) + ".csv");
            //            subset.saveAsTextFile(path);
            List<String> linesList = subset.collect();
            FileUtils.writeLines(new File(path), linesList);
        }
    }

    @AllArgsConstructor
    private static class SequenceToStringFunction implements Function<List<List<Writable>>, String> {

        private final String delim;

        @Override
        public String call(List<List<Writable>> sequence) throws Exception {

            StringBuilder sb = new StringBuilder();
            boolean firstTimeStep = true;
            for (List<Writable> c : sequence) {
                if (!firstTimeStep)
                    sb.append("\n");
                boolean first = true;
                for (Writable w : c) {
                    if (!first)
                        sb.append(delim);
                    sb.append(w.toString());
                    first = false;
                }
                firstTimeStep = false;
            }

            return sb.toString();
        }
    }



    //Another quick and dirty CSV export (local). Dumps all values into a single file
    public static void exportStringLocal(File outputFile, JavaRDD<String> data, int rngSeed) throws Exception {
        List<String> linesList = data.collect(); //Requires all data in memory
        if (!(linesList instanceof ArrayList))
            linesList = new ArrayList<>(linesList);
        Collections.shuffle(linesList, new Random(rngSeed));

        FileUtils.writeLines(outputFile, linesList);
    }

    //Quick and dirty CSV export: one file per sequence, with shuffling of the order of sequences
    public static void exportCSVSequenceLocal(File baseDir, JavaRDD<List<List<Writable>>> sequences, long seed)
                    throws Exception {
        baseDir.mkdirs();
        if (!baseDir.isDirectory())
            throw new IllegalArgumentException("File is not a directory: " + baseDir.toString());
        String baseDirStr = baseDir.toString();

        List<String> fileContents = sequences.map(new SequenceToStringFunction(",")).collect();
        if (!(fileContents instanceof ArrayList))
            fileContents = new ArrayList<>(fileContents);
        Collections.shuffle(fileContents, new Random(seed));

        int i = 0;
        for (String s : fileContents) {
            String path = FilenameUtils.concat(baseDirStr, i + ".csv");
            File f = new File(path);
            FileUtils.writeStringToFile(f, s);
            i++;
        }
    }

    //Quick and dirty CSV export: one file per sequence, without shuffling
    public static void exportCSVSequenceLocalNoShuffling(File baseDir, JavaRDD<List<List<Writable>>> sequences)
                    throws Exception {
        exportCSVSequenceLocalNoShuffling(baseDir, sequences, "", ",", "csv");
    }

    public static void exportCSVSequenceLocalNoShuffling(File baseDir, JavaRDD<List<List<Writable>>> sequences,
                    String delimiter, String filePrefix, String fileExtension) throws Exception {
        baseDir.mkdirs();
        if (!baseDir.isDirectory())
            throw new IllegalArgumentException("File is not a directory: " + baseDir.toString());
        String baseDirStr = baseDir.toString();

        List<String> fileContents = sequences.map(new SequenceToStringFunction(delimiter)).collect();
        if (!(fileContents instanceof ArrayList))
            fileContents = new ArrayList<>(fileContents);

        int i = 0;
        for (String s : fileContents) {
            String path = FilenameUtils.concat(baseDirStr, filePrefix + "_" + i + "." + fileExtension);
            File f = new File(path);
            FileUtils.writeStringToFile(f, s);
            i++;
        }
    }
}
