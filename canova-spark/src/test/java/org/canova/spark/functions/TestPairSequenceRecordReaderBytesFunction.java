package org.canova.spark.functions;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.canova.codec.reader.CodecRecordReader;
import org.canova.spark.BaseSparkTest;
import org.canova.spark.functions.pairdata.*;
import org.canova.spark.util.CanovaSparkUtil;
import org.junit.Test;
import scala.Tuple2;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestPairSequenceRecordReaderBytesFunction extends BaseSparkTest {

    @Test
    public void test() throws Exception {
        //Goal: combine separate files together into a hadoop sequence file, for later parsing by a SequenceRecordReader
        //For example: use to combine input and labels data from separate files for training a RNN
        JavaSparkContext sc = getContext();

        ClassPathResource cpr = new ClassPathResource("/video/shapes_0.mp4");
        String path = cpr.getFile().getAbsolutePath();
        String folder = path.substring(0, path.length() - 12);
        path = folder + "*";

        PathToKeyConverter pathConverter = new PathToKeyConverterFilename();
        JavaPairRDD<Text,BytesPairWritable> toWrite = CanovaSparkUtil.combineFilesForSequenceFile(sc, path, path, pathConverter);

        Path p = Files.createTempDirectory("dl4j_rrbytesPairOut");
        p.toFile().deleteOnExit();
        String outPath = p.toString() + "/out";
        new File(outPath).deleteOnExit();
        toWrite.saveAsNewAPIHadoopFile(outPath, Text.class, BytesPairWritable.class, SequenceFileOutputFormat.class);

        //Load back into memory:
        JavaPairRDD<Text,BytesPairWritable> fromSeq = sc.sequenceFile(outPath, Text.class, BytesPairWritable.class);

        SequenceRecordReader srr1 = getReader();
        SequenceRecordReader srr2 = getReader();
        PairSequenceRecordReaderBytesFunction psrbf = new PairSequenceRecordReaderBytesFunction(srr1,srr2);

        JavaRDD<Tuple2<Collection<Collection<Writable>>,Collection<Collection<Writable>>>> writables = fromSeq.map(psrbf);
        List<Tuple2<Collection<Collection<Writable>>,Collection<Collection<Writable>>>> fromSequenceFile = writables.collect();

        //Load manually (single copy) and compare:
        InputSplit is = new FileSplit(new File(folder),new String[]{"mp4"}, true);
        SequenceRecordReader srr = getReader();
        srr.initialize(is);

        List<Collection<Collection<Writable>>> list = new ArrayList<>(4);
        while(srr.hasNext()){
            list.add(srr.sequenceRecord());
        }

        assertEquals(4, list.size());
        assertEquals(4, fromSequenceFile.size());

        boolean[] found = new boolean[4];
        for( int i=0; i<4; i++ ){
            int foundIndex = -1;
            Tuple2<Collection<Collection<Writable>>,Collection<Collection<Writable>>> tuple2 = fromSequenceFile.get(i);
            Collection<Collection<Writable>> seq1 = tuple2._1();
            Collection<Collection<Writable>> seq2 = tuple2._2();
            assertEquals(seq1,seq2);

            for( int j=0; j<4; j++ ){
                if(seq1.equals(list.get(j))){
                    if(foundIndex != -1) fail();    //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if(found[foundIndex]) fail();   //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true;       //mark this one as seen before
                }
            }
        }
        int count = 0;
        for( boolean b : found ) if(b) count++;
        assertEquals(4, count);  //Expect all 4 and exactly 4 pairwise matches between spark and local versions

    }

    private static SequenceRecordReader getReader(){
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
