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

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.nd4j.api.loader.FileBatch;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;

public class SparkDataUtils {

    private SparkDataUtils(){ }


    public static void createFileBatchesLocal(File inputDirectory, boolean recursive, File outputDirectory, int batchSize) throws IOException {
        createFileBatchesLocal(inputDirectory, null, recursive, outputDirectory, batchSize);
    }

    public static void createFileBatchesLocal(File inputDirectory, String[] extensions, boolean recursive, File outputDirectory, int batchSize) throws IOException {
        //Local version
        List<File> c = new ArrayList<>(FileUtils.listFiles(inputDirectory, extensions, recursive));
        Collections.shuffle(c);

        //Construct file batch
        List<String> list = new ArrayList<>();
        List<byte[]> bytes = new ArrayList<>();
        for( int i=0; i<c.size(); i++ ){
            list.add(c.get(i).getPath());
            bytes.add(FileUtils.readFileToByteArray(c.get(i)));

            if(list.size() == batchSize){
                process(list, bytes, outputDirectory);
            }
        }
        if(list.size() > 0){
            process(list, bytes, outputDirectory);
        }
    }

    private static void process(List<String> paths, List<byte[]> bytes, File outputDirectory) throws IOException {
        FileBatch fb = new FileBatch(bytes, paths);
        String name = UUID.randomUUID().toString().replaceAll("-","") + ".zip";
        File f = new File(outputDirectory, name);
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f))) {
            fb.writeAsZip(bos);
        }

        paths.clear();
        bytes.clear();
    }

    public void createFileBatchesSpark(JavaRDD<String> filePaths, final String rootOutputDir, final int batchSize, JavaSparkContext sc){

        final org.apache.hadoop.conf.Configuration hadoopConfig = sc.hadoopConfiguration();
        //Here: assume input is images. We can't store them as Float32 arrays - that's too inefficient
        // instead: let's store the raw file content in a batch.
        long count = filePaths.count();
        long maxPartitions = count / batchSize;
        JavaRDD<String> repartitioned = filePaths.repartition(Math.max(filePaths.getNumPartitions(), (int)maxPartitions));
        repartitioned.foreachPartition(new VoidFunction<Iterator<String>>() {
            @Override
            public void call(Iterator<String> stringIterator) throws Exception {
                //Construct file batch
                List<String> list = new ArrayList<>();
                List<byte[]> bytes = new ArrayList<>();
                while(stringIterator.hasNext()){

                    if(list.size() == batchSize){
                        process(list, bytes);
                    }
                }
                if(list.size() > 0){
                    process(list, bytes);
                }
            }

            private void process(List<String> paths, List<byte[]> bytes) throws IOException {
                FileBatch fb = new FileBatch(bytes, paths);
                String name = UUID.randomUUID().toString().replaceAll("-","") + ".zip";
                String outPath = FilenameUtils.concat(rootOutputDir, name);
                FileSystem fileSystem = FileSystem.get(hadoopConfig);
                try (BufferedOutputStream bos = new BufferedOutputStream(fileSystem.create(new Path(outPath)))) {
                    fb.writeAsZip(bos);
                }

                paths.clear();
                bytes.clear();
            }
        });
    }

}
