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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.datavec.codec.reader.CodecRecordReader;
import org.datavec.spark.BaseSparkTest;
import org.datavec.spark.functions.pairdata.BytesPairWritable;
import org.datavec.spark.functions.pairdata.PairSequenceRecordReaderBytesFunction;
import org.datavec.spark.functions.pairdata.PathToKeyConverter;
import org.datavec.spark.functions.pairdata.PathToKeyConverterFilename;
import org.datavec.spark.util.DataVecSparkUtil;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;
import scala.Tuple2;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestPairSequenceRecordReaderBytesFunction extends BaseSparkTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void test() throws Exception {
        //Goal: combine separate files together into a hadoop sequence file, for later parsing by a SequenceRecordReader
        //For example: use to combine input and labels data from separate files for training a RNN
        JavaSparkContext sc = getContext();

        File f = testDir.newFolder();
        new ClassPathResource("datavec-spark/video/").copyDirectory(f);
        String path = f.getAbsolutePath() + "/*";

        PathToKeyConverter pathConverter = new PathToKeyConverterFilename();
        JavaPairRDD<Text, BytesPairWritable> toWrite =
                        DataVecSparkUtil.combineFilesForSequenceFile(sc, path, path, pathConverter);

        Path p = Files.createTempDirectory("dl4j_rrbytesPairOut");
        p.toFile().deleteOnExit();
        String outPath = p.toString() + "/out";
        new File(outPath).deleteOnExit();
        toWrite.saveAsNewAPIHadoopFile(outPath, Text.class, BytesPairWritable.class, SequenceFileOutputFormat.class);

        //Load back into memory:
        JavaPairRDD<Text, BytesPairWritable> fromSeq = sc.sequenceFile(outPath, Text.class, BytesPairWritable.class);

        SequenceRecordReader srr1 = getReader();
        SequenceRecordReader srr2 = getReader();
        PairSequenceRecordReaderBytesFunction psrbf = new PairSequenceRecordReaderBytesFunction(srr1, srr2);

        JavaRDD<Tuple2<List<List<Writable>>, List<List<Writable>>>> writables = fromSeq.map(psrbf);
        List<Tuple2<List<List<Writable>>, List<List<Writable>>>> fromSequenceFile = writables.collect();

        //Load manually (single copy) and compare:
        InputSplit is = new FileSplit(f, new String[] {"mp4"}, true);
        SequenceRecordReader srr = getReader();
        srr.initialize(is);

        List<List<List<Writable>>> list = new ArrayList<>(4);
        while (srr.hasNext()) {
            list.add(srr.sequenceRecord());
        }

        assertEquals(4, list.size());
        assertEquals(4, fromSequenceFile.size());

        boolean[] found = new boolean[4];
        for (int i = 0; i < 4; i++) {
            int foundIndex = -1;
            Tuple2<List<List<Writable>>, List<List<Writable>>> tuple2 = fromSequenceFile.get(i);
            List<List<Writable>> seq1 = tuple2._1();
            List<List<Writable>> seq2 = tuple2._2();
            assertEquals(seq1, seq2);

            for (int j = 0; j < 4; j++) {
                if (seq1.equals(list.get(j))) {
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

    private static SequenceRecordReader getReader() {
        SequenceRecordReader seqRR = new CodecRecordReader();
        Configuration conf = new Configuration();
        conf.set(CodecRecordReader.RAVEL, "true");
        conf.set(CodecRecordReader.START_FRAME, "0");
        conf.set(CodecRecordReader.TOTAL_FRAMES, "25");
        conf.set(CodecRecordReader.ROWS, "64");
        conf.set(CodecRecordReader.COLUMNS, "64");
        seqRR.setConf(conf);
        return seqRR;
    }
}
