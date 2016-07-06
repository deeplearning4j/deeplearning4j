package org.canova.spark.functions;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageRecordReader;
import org.canova.spark.BaseSparkTest;
import org.canova.spark.functions.data.FilesAsBytesFunction;
import org.canova.spark.functions.data.RecordReaderBytesFunction;
import org.junit.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestRecordReaderBytesFunction extends BaseSparkTest{

    @Test
    public void testRecordReaderBytesFunction() throws Exception {
        JavaSparkContext sc = getContext();

        //Local file path
        ClassPathResource cpr = new ClassPathResource("/imagetest/0/a.bmp");
        String path = cpr.getFile().getAbsolutePath();
        String folder = path.substring(0, path.length() - 7);
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

        //Load data from sequence file, parse via RecordReader:
        JavaPairRDD<Text, BytesWritable> fromSeqFile = sc.sequenceFile(outPath, Text.class, BytesWritable.class);
        RecordReader rr = new ImageRecordReader(28, 28, 1, true, Arrays.asList("0", "1"));
        JavaRDD<Collection<Writable>> canovaData = fromSeqFile.map(new RecordReaderBytesFunction(rr));


        //Next: do the same thing locally, and compare the results
        InputSplit is = new FileSplit(new File(folder), new String[]{"bmp"}, true);
        ImageRecordReader irr = new ImageRecordReader(28, 28, 1, true);
        irr.initialize(is);

        List<Collection<Writable>> list = new ArrayList<>(4);
        while (irr.hasNext()) {
            list.add(irr.next());
        }

        List<Collection<Writable>> fromSequenceFile = canovaData.collect();

        assertEquals(4, list.size());
        assertEquals(4, fromSequenceFile.size());

        //Check that each of the values from Spark equals exactly one of the values doing it locally
        boolean[] found = new boolean[4];
        for (int i = 0; i < 4; i++) {
            int foundIndex = -1;
            Collection<Writable> collection = fromSequenceFile.get(i);
            for (int j = 0; j < 4; j++) {
                if (collection.equals(list.get(j))) {
                    if (foundIndex != -1)
                        fail();    //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if (found[foundIndex])
                        fail();   //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true;   //mark this one as seen before
                }
            }
        }
        int count = 0;
        for (boolean b : found) if (b) count++;
        assertEquals(4, count);  //Expect all 4 and exactly 4 pairwise matches between spark and local versions
        System.out.println("COUNT: " + count);
    }

}
