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

package org.datavec.spark.functions;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.input.PortableDataStream;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.codec.reader.CodecRecordReader;
import org.datavec.spark.BaseSparkTest;
import org.datavec.spark.functions.data.FilesAsBytesFunction;
import org.datavec.spark.functions.data.SequenceRecordReaderBytesFunction;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestSequenceRecordReaderBytesFunction extends BaseSparkTest {

    @Test
    public void testRecordReaderBytesFunction() throws Exception {

        //Local file path
        ClassPathResource cpr = new ClassPathResource("/video/shapes_0.mp4");
        String path = cpr.getFile().getAbsolutePath();
        String folder = path.substring(0, path.length() - 12);
        path = folder + "*";

        //Load binary data from local file system, convert to a sequence file:
        //Load and convert
        JavaPairRDD<String, PortableDataStream> origData = sc.binaryFiles(path);
        JavaPairRDD<Text, BytesWritable> filesAsBytes = origData.mapToPair(new FilesAsBytesFunction());
        //Write the sequence file:
        Path p = Files.createTempDirectory("dl4j_rrbytesTest");
        p.toFile().deleteOnExit();
        String outPath = p.toString() + "/out";
        filesAsBytes.saveAsNewAPIHadoopFile(outPath, Text.class, BytesWritable.class, SequenceFileOutputFormat.class);

        //Load data from sequence file, parse via SequenceRecordReader:
        JavaPairRDD<Text, BytesWritable> fromSeqFile = sc.sequenceFile(outPath, Text.class, BytesWritable.class);
        SequenceRecordReader seqRR = new CodecRecordReader();
        Configuration conf = new Configuration();
        conf.set(CodecRecordReader.RAVEL, "true");
        conf.set(CodecRecordReader.START_FRAME, "0");
        conf.set(CodecRecordReader.TOTAL_FRAMES, "25");
        conf.set(CodecRecordReader.ROWS, "64");
        conf.set(CodecRecordReader.COLUMNS, "64");
        Configuration confCopy = new Configuration(conf);
        seqRR.setConf(conf);
        JavaRDD<List<List<Writable>>> dataVecData = fromSeqFile.map(new SequenceRecordReaderBytesFunction(seqRR));



        //Next: do the same thing locally, and compare the results
        InputSplit is = new FileSplit(new File(folder), new String[] {"mp4"}, true);
        SequenceRecordReader srr = new CodecRecordReader();
        srr.initialize(is);
        srr.setConf(confCopy);

        List<List<List<Writable>>> list = new ArrayList<>(4);
        while (srr.hasNext()) {
            list.add(srr.sequenceRecord());
        }
        assertEquals(4, list.size());

        List<List<List<Writable>>> fromSequenceFile = dataVecData.collect();

        assertEquals(4, list.size());
        assertEquals(4, fromSequenceFile.size());

        boolean[] found = new boolean[4];
        for (int i = 0; i < 4; i++) {
            int foundIndex = -1;
            List<List<Writable>> collection = fromSequenceFile.get(i);
            for (int j = 0; j < 4; j++) {
                if (collection.equals(list.get(j))) {
                    if (foundIndex != -1)
                        fail(); //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if (found[foundIndex])
                        fail(); //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true; //mark this one as seen before
                }
            }
        }
        int count = 0;
        for (boolean b : found)
            if (b)
                count++;
        assertEquals(4, count); //Expect all 4 and exactly 4 pairwise matches between spark and local versions
    }

}
