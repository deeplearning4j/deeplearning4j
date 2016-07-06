package org.canova.spark.functions;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.ArrayWritable;
import org.canova.api.writable.Writable;
import org.canova.image.recordreader.ImageRecordReader;
import org.canova.spark.BaseSparkTest;
import org.junit.Test;

import java.io.File;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class TestRecordReaderFunction extends BaseSparkTest {

    @Test
    public void testRecordReaderFunction() throws Exception {
        JavaSparkContext sc = getContext();

        ClassPathResource cpr = new ClassPathResource("/imagetest/0/a.bmp");
        List<String> labelsList = Arrays.asList("0", "1");   //Need this for Spark: can't infer without init call

        String path = cpr.getFile().getAbsolutePath();
        String folder = path.substring(0, path.length() - 7);
        path = folder + "*";

        JavaPairRDD<String,PortableDataStream> origData = sc.binaryFiles(path);
        assertEquals(4,origData.count());    //4 images

        RecordReaderFunction rrf = new RecordReaderFunction(new ImageRecordReader(28,28,1,true,labelsList));
        JavaRDD<Collection<Writable>> rdd = origData.map(rrf);
        List<Collection<Writable>> listSpark = rdd.collect();

        assertEquals(4,listSpark.size());
        for( int i=0; i<4; i++ ){
            assertEquals(1+1, listSpark.get(i).size());
            assertEquals(28*28, ((ArrayWritable)listSpark.get(i).iterator().next()).length());
        }

        //Load normally (i.e., not via Spark), and check that we get the same results (order not withstanding)
        InputSplit is = new FileSplit(new File(folder),new String[]{"bmp"}, true);
//        System.out.println("Locations: " + Arrays.toString(is.locations()));
        ImageRecordReader irr = new ImageRecordReader(28,28,1,true);
        irr.initialize(is);

        List<Collection<Writable>> list = new ArrayList<>(4);
        while(irr.hasNext()){
            list.add(irr.next());
        }
        assertEquals(4, list.size());

//        System.out.println("Spark list:");
//        for(Collection<Writable> c : listSpark ) System.out.println(c);
//        System.out.println("Local list:");
//        for(Collection<Writable> c : list ) System.out.println(c);

        //Check that each of the values from Spark equals exactly one of the values doing it locally
        boolean[] found = new boolean[4];
        for( int i=0; i<4; i++ ){
            int foundIndex = -1;
            Collection<Writable> collection = listSpark.get(i);
            for( int j=0; j<4; j++ ){
                if(collection.equals(list.get(j))){
                    if(foundIndex != -1) fail();    //Already found this value -> suggests this spark value equals two or more of local version? (Shouldn't happen)
                    foundIndex = j;
                    if(found[foundIndex]) fail();   //One of the other spark values was equal to this one -> suggests duplicates in Spark list
                    found[foundIndex] = true;   //mark this one as seen before
                }
            }
        }
        int count = 0;
        for( boolean b : found ) if(b) count++;
        assertEquals(4,count);  //Expect all 4 and exactly 4 pairwise matches between spark and local versions
    }

}