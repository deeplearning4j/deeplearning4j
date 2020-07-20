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

package org.deeplearning4j.spark.util;

import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.datavec.spark.util.SerializableHadoopConfig;
import org.deeplearning4j.core.loader.impl.RecordReaderFileBatchLoader;
import org.nd4j.common.loader.FileBatch;

import java.io.*;
import java.util.*;

/**
 * Utilities for handling data for Spark training
 *
 * @author Alex Black
 */
public class SparkDataUtils {

    private SparkDataUtils() {
    }

    /**
     * See {@link #createFileBatchesLocal(File, String[], boolean, File, int)}.<br>
     * The directory filtering (extensions arg) is null when calling this method.
     */
    public static void createFileBatchesLocal(File inputDirectory, boolean recursive, File outputDirectory, int batchSize) throws IOException {
        createFileBatchesLocal(inputDirectory, null, recursive, outputDirectory, batchSize);
    }

    /**
     * Create a number of {@link FileBatch} files from local files (in random order).<br>
     * Use cases: distributed training on compressed file formats such as images, that need to be loaded to a remote
     * file storage system such as HDFS. Local files can be created using this method and then copied to HDFS for training.<br>
     * FileBatch is also compressed (zip file format) so space may be saved in some cases (such as CSV sequences)
     * For example, if we were training with a minibatch size of 64 images, reading the raw images would result in 64
     * different disk reads (one for each file) - which could clearly be a bottleneck during training.<br>
     * Alternatively, we could create and save DataSet/INDArray objects containing a batch of images - however, storing
     * images in FP32 (or ever UINT8) format - effectively a bitmap - is still much less efficient than the raw image files.<br>
     * Instead, can create minibatches of {@link FileBatch} objects: these objects contain the raw file content for
     * multiple files (as byte[]s) along with their original paths, which can then be used for distributed training using
     * {@link RecordReaderFileBatchLoader}.<br>
     * This approach gives us the benefits of the original file format (i.e., small size, compression) along with
     * the benefits of a batched DataSet/INDArray format - i.e., disk reads are reduced by a factor of the minibatch size.<br>
     * <br>
     * See {@link #createFileBatchesSpark(JavaRDD, String, int, JavaSparkContext)} for the distributed (Spark) version of this method.<br>
     * <br>
     * Usage - image classification example - assume each FileBatch object contains a number of jpg/png etc image files
     * <pre>
     * {@code
     * JavaSparkContext sc = ...
     * SparkDl4jMultiLayer net = ...
     * String baseFileBatchDir = ...
     * JavaRDD<String> paths = org.deeplearning4j.spark.util.SparkUtils.listPaths(sc, baseFileBatchDir);
     *
     * //Image record reader:
     * PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
     * ImageRecordReader rr = new ImageRecordReader(32, 32, 1, labelMaker);
     * rr.setLabels(<labels here>);
     *
     * //Create DataSetLoader:
     * int batchSize = 32;
     * int numClasses = 1000;
     * DataSetLoader loader = RecordReaderFileBatchLoader(rr, batchSize, 1, numClasses);
     *
     * //Fit the network
     * net.fitPaths(paths, loader);
     * }
     * </pre>
     *
     * @param inputDirectory  Directory containing the files to convert
     * @param extensions      Optional (may be null). If non-null, only those files with the specified extension will be included
     * @param recursive       If true: convert the files recursively
     * @param outputDirectory Output directory to save the created FileBatch objects
     * @param batchSize       Batch size - i.e., minibatch size to be used for training, and the number of files to
     *                        include in each FileBatch object
     * @throws IOException If an error occurs while reading the files
     * @see #createFileBatchesSpark(JavaRDD, String, int, JavaSparkContext)
     * @see org.datavec.api.records.reader.impl.filebatch.FileBatchRecordReader FileBatchRecordReader for local training on these files, if required
     * @see org.datavec.api.records.reader.impl.filebatch.FileBatchSequenceRecordReader for local training on these files, if required
     */
    public static void createFileBatchesLocal(File inputDirectory, String[] extensions, boolean recursive, File outputDirectory, int batchSize) throws IOException {
        if(!outputDirectory.exists())
            outputDirectory.mkdirs();
        //Local version
        List<File> c = new ArrayList<>(FileUtils.listFiles(inputDirectory, extensions, recursive));
        Collections.shuffle(c);

        //Construct file batch
        List<String> list = new ArrayList<>();
        List<byte[]> bytes = new ArrayList<>();
        for (int i = 0; i < c.size(); i++) {
            list.add(c.get(i).toURI().toString());
            bytes.add(FileUtils.readFileToByteArray(c.get(i)));

            if (list.size() == batchSize) {
                process(list, bytes, outputDirectory);
            }
        }
        if (list.size() > 0) {
            process(list, bytes, outputDirectory);
        }
    }

    private static void process(List<String> paths, List<byte[]> bytes, File outputDirectory) throws IOException {
        FileBatch fb = new FileBatch(bytes, paths);
        String name = UUID.randomUUID().toString().replaceAll("-", "") + ".zip";
        File f = new File(outputDirectory, name);
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f))) {
            fb.writeAsZip(bos);
        }

        paths.clear();
        bytes.clear();
    }

    /**
     * Create a number of {@link FileBatch} files from files on network storage such as HDFS (in random order).<br>
     * Use cases: distributed training on compressed file formats such as images, that need to be loaded to a remote
     * file storage system such as HDFS.<br>
     * For example, if we were training with a minibatch size of 64 images, reading the raw images would result in 64
     * different disk reads (one for each file) - which could clearly be a bottleneck during training.<br>
     * Alternatively, we could create and save DataSet/INDArray objects containing a batch of images - however, storing
     * images in FP32 (or ever UINT8) format - effectively a bitmap - is still much less efficient than the raw image files.<br>
     * Instead, can create minibatches of {@link FileBatch} objects: these objects contain the raw file content for
     * multiple files (as byte[]s) along with their original paths, which can then be used for distributed training using
     * {@link RecordReaderFileBatchLoader}.<br>
     * This approach gives us the benefits of the original file format (i.e., small size, compression) along with
     * the benefits of a batched DataSet/INDArray format - i.e., disk reads are reduced by a factor of the minibatch size.<br>
     * <br>
     * See {@link #createFileBatchesLocal(File, String[], boolean, File, int)} for the local (non-Spark) version of this method.
     * <br>
     * Usage - image classification example - assume each FileBatch object contains a number of jpg/png etc image files
     * <pre>
     * {@code
     * JavaSparkContext sc = ...
     * SparkDl4jMultiLayer net = ...
     * String baseFileBatchDir = ...
     * JavaRDD<String> paths = org.deeplearning4j.spark.util.SparkUtils.listPaths(sc, baseFileBatchDir);
     *
     * //Image record reader:
     * PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
     * ImageRecordReader rr = new ImageRecordReader(32, 32, 1, labelMaker);
     * rr.setLabels(<labels here>);
     *
     * //Create DataSetLoader:
     * int batchSize = 32;
     * int numClasses = 1000;
     * DataSetLoader loader = RecordReaderFileBatchLoader(rr, batchSize, 1, numClasses);
     *
     * //Fit the network
     * net.fitPaths(paths, loader);
     * }
     * </pre>
     *
     * @param batchSize Batch size - i.e., minibatch size to be used for training, and the number of files to
     *                  include in each FileBatch object
     * @throws IOException If an error occurs while reading the files
     * @see #createFileBatchesLocal(File, String[], boolean, File, int)
     * @see org.datavec.api.records.reader.impl.filebatch.FileBatchRecordReader FileBatchRecordReader for local training on these files, if required
     * @see org.datavec.api.records.reader.impl.filebatch.FileBatchSequenceRecordReader for local training on these files, if required
     */
    public static void createFileBatchesSpark(JavaRDD<String> filePaths, final String rootOutputDir, final int batchSize, JavaSparkContext sc) {
        createFileBatchesSpark(filePaths, rootOutputDir, batchSize, sc.hadoopConfiguration());
    }

    /**
     * See {@link #createFileBatchesSpark(JavaRDD, String, int, JavaSparkContext)}
     */
    public static void createFileBatchesSpark(JavaRDD<String> filePaths, final String rootOutputDir, final int batchSize, @NonNull final org.apache.hadoop.conf.Configuration hadoopConfig) {
        final SerializableHadoopConfig conf = new SerializableHadoopConfig(hadoopConfig);
        //Here: assume input is images. We can't store them as Float32 arrays - that's too inefficient
        // instead: let's store the raw file content in a batch.
        long count = filePaths.count();
        long maxPartitions = count / batchSize;
        JavaRDD<String> repartitioned = filePaths.repartition(Math.max(filePaths.getNumPartitions(), (int) maxPartitions));
        repartitioned.foreachPartition(new VoidFunction<Iterator<String>>() {
            @Override
            public void call(Iterator<String> stringIterator) throws Exception {
                //Construct file batch
                List<String> list = new ArrayList<>();
                List<byte[]> bytes = new ArrayList<>();
                FileSystem fs = FileSystem.get(conf.getConfiguration());
                while (stringIterator.hasNext()) {
                    String inFile = stringIterator.next();
                    byte[] fileBytes;
                    try (BufferedInputStream bis = new BufferedInputStream(fs.open(new Path(inFile)))) {
                        fileBytes = IOUtils.toByteArray(bis);
                    }
                    list.add(inFile);
                    bytes.add(fileBytes);

                    if (list.size() == batchSize) {
                        process(list, bytes);
                    }
                }
                if (list.size() > 0) {
                    process(list, bytes);
                }
            }

            private void process(List<String> paths, List<byte[]> bytes) throws IOException {
                FileBatch fb = new FileBatch(bytes, paths);
                String name = UUID.randomUUID().toString().replaceAll("-", "") + ".zip";
                String outPath = FilenameUtils.concat(rootOutputDir, name);
                FileSystem fileSystem = FileSystem.get(conf.getConfiguration());
                try (BufferedOutputStream bos = new BufferedOutputStream(fileSystem.create(new Path(outPath)))) {
                    fb.writeAsZip(bos);
                }

                paths.clear();
                bytes.clear();
            }
        });
    }

}
