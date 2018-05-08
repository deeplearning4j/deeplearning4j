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

package org.datavec.image.recordreader;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.PathMultiLabelGenerator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.NDArrayRecordBatch;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Created by Alex on 27/09/2016.
 */
public class TestImageRecordReader {

    @Test(expected = IllegalArgumentException.class)
    public void testEmptySplit() throws IOException {
        InputSplit data = new CollectionInputSplit(new ArrayList<URI>());
        new ImageRecordReader().initialize(data, null);
    }

    @Test
    public void testMetaData() throws IOException {

        ClassPathResource cpr = new ClassPathResource("/testimages/class0/0.jpg");
        File parentDir = cpr.getFile().getParentFile().getParentFile();
        //        System.out.println(f.getAbsolutePath());
        //        System.out.println(f.getParentFile().getParentFile().getAbsolutePath());
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(32, 32, 3, labelMaker);
        rr.initialize(new FileSplit(parentDir));

        List<List<Writable>> out = new ArrayList<>();
        while (rr.hasNext()) {
            List<Writable> l = rr.next();
            out.add(l);
            assertEquals(2, l.size());
        }

        assertEquals(6, out.size());

        rr.reset();
        List<List<Writable>> out2 = new ArrayList<>();
        List<Record> out3 = new ArrayList<>();
        List<RecordMetaData> meta = new ArrayList<>();

        while (rr.hasNext()) {
            Record r = rr.nextRecord();
            out2.add(r.getRecord());
            out3.add(r);
            meta.add(r.getMetaData());
            //            System.out.println(r.getMetaData() + "\t" + r.getRecord().get(1));
        }

        assertEquals(out, out2);

        List<Record> fromMeta = rr.loadFromMetaData(meta);
        assertEquals(out3, fromMeta);
    }

    @Test
    public void testImageRecordReaderLabelsOrder() throws Exception {
        //Labels order should be consistent, regardless of file iteration order

        //Idea: labels order should be consistent regardless of input file order
        File f0 = new ClassPathResource("/testimages/class0/0.jpg").getFile();
        File f1 = new ClassPathResource("/testimages/class1/A.jpg").getFile();

        List<URI> order0 = Arrays.asList(f0.toURI(), f1.toURI());
        List<URI> order1 = Arrays.asList(f1.toURI(), f0.toURI());

        ParentPathLabelGenerator labelMaker0 = new ParentPathLabelGenerator();
        ImageRecordReader rr0 = new ImageRecordReader(32, 32, 3, labelMaker0);
        rr0.initialize(new CollectionInputSplit(order0));

        ParentPathLabelGenerator labelMaker1 = new ParentPathLabelGenerator();
        ImageRecordReader rr1 = new ImageRecordReader(32, 32, 3, labelMaker1);
        rr1.initialize(new CollectionInputSplit(order1));

        List<String> labels0 = rr0.getLabels();
        List<String> labels1 = rr1.getLabels();

        //        System.out.println(labels0);
        //        System.out.println(labels1);

        assertEquals(labels0, labels1);
    }


    @Test
    public void testImageRecordReaderRandomization() throws Exception {
        //Order of FileSplit+ImageRecordReader should be different after reset

        //Idea: labels order should be consistent regardless of input file order
        File f0 = new ClassPathResource("/testimages/").getFile();

        FileSplit fs = new FileSplit(f0, new Random(12345));

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(32, 32, 3, labelMaker);
        rr.initialize(fs);

        List<List<Writable>> out1 = new ArrayList<>();
        List<File> order1 = new ArrayList<>();
        while (rr.hasNext()) {
            out1.add(rr.next());
            order1.add(rr.getCurrentFile());
        }
        assertEquals(6, out1.size());
        assertEquals(6, order1.size());

        rr.reset();
        List<List<Writable>> out2 = new ArrayList<>();
        List<File> order2 = new ArrayList<>();
        while (rr.hasNext()) {
            out2.add(rr.next());
            order2.add(rr.getCurrentFile());
        }
        assertEquals(6, out2.size());
        assertEquals(6, order2.size());

        assertNotEquals(out1, out2);
        assertNotEquals(order1, order2);

        //Check that different seed gives different order for the initial iteration
        FileSplit fs2 = new FileSplit(f0, new Random(999999999));

        ParentPathLabelGenerator labelMaker2 = new ParentPathLabelGenerator();
        ImageRecordReader rr2 = new ImageRecordReader(32, 32, 3, labelMaker2);
        rr2.initialize(fs2);

        List<File> order3 = new ArrayList<>();
        while (rr2.hasNext()) {
            rr2.next();
            order3.add(rr2.getCurrentFile());
        }
        assertEquals(6, order3.size());

        assertNotEquals(order1, order3);
    }


    @Test
    public void testImageRecordReaderRegression() throws Exception {

        PathLabelGenerator regressionLabelGen = new TestRegressionLabelGen();

        ImageRecordReader rr = new ImageRecordReader(28, 28, 3, regressionLabelGen);

        File rootDir = new ClassPathResource("/testimages/").getFile();
        FileSplit fs = new FileSplit(rootDir);
        rr.initialize(fs);
        URI[] arr = fs.locations();

        assertTrue(rr.getLabels() == null || rr.getLabels().isEmpty());

        List<Writable> expLabels = new ArrayList<>();
        for(URI u : arr){
            String path = u.getPath();
            expLabels.add(testLabel(path.substring(path.length()-5, path.length())));
        }

        int count = 0;
        while(rr.hasNext()){
            List<Writable> l = rr.next();

            assertEquals(2, l.size());
            assertEquals(expLabels.get(count), l.get(1));

            count++;
        }
        assertEquals(6, count);

        //Test batch ops:
        rr.reset();

        List<List<Writable>> b1 = rr.next(3);
        List<List<Writable>> b2 = rr.next(3);
        assertFalse(rr.hasNext());

        NDArrayRecordBatch b1a = (NDArrayRecordBatch)b1;
        NDArrayRecordBatch b2a = (NDArrayRecordBatch)b2;
        assertEquals(2, b1a.getArrays().size());
        assertEquals(2, b2a.getArrays().size());

        NDArrayWritable l1 = new NDArrayWritable(Nd4j.create(new double[]{expLabels.get(0).toDouble(),
                expLabels.get(1).toDouble(), expLabels.get(2).toDouble()}, new int[]{3,1}));
        NDArrayWritable l2 = new NDArrayWritable(Nd4j.create(new double[]{expLabels.get(3).toDouble(),
                expLabels.get(4).toDouble(), expLabels.get(5).toDouble()}, new int[]{3,1}));

        INDArray act1 = b1a.getArrays().get(1);
        INDArray act2 = b2a.getArrays().get(1);
        assertEquals(l1.get(), act1);
        assertEquals(l2.get(), act2);
    }


    private static class TestRegressionLabelGen implements PathLabelGenerator {

        @Override
        public Writable getLabelForPath(String path) {
            String filename = path.substring(path.length()-5, path.length());
            return testLabel(filename);
        }

        @Override
        public Writable getLabelForPath(URI uri) {
            return getLabelForPath(uri.toString());
        }

        @Override
        public boolean inferLabelClasses() {
            return false;
        }
    }

    private static Writable testLabel(String filename){
        switch(filename){
            case "0.jpg":
                return new DoubleWritable(0.0);
            case "1.png":
                return new DoubleWritable(1.0);
            case "2.jpg":
                return new DoubleWritable(2.0);
            case "A.jpg":
                return new DoubleWritable(10);
            case "B.png":
                return new DoubleWritable(11);
            case "C.jpg":
                return new DoubleWritable(12);
            default:
                throw new RuntimeException(filename);
        }
    }


    @Test
    public void testImageRecordReaderPathMultiLabelGenerator() throws Exception {
        //Assumption: 2 multi-class (one hot) classification labels: 2 and 3 classes respectively
        // PLUS single value (Writable) regression label

        PathMultiLabelGenerator multiLabelGen = new TestPathMultiLabelGenerator();

        ImageRecordReader rr = new ImageRecordReader(28, 28, 3, multiLabelGen);

        File rootDir = new ClassPathResource("/testimages/").getFile();
        FileSplit fs = new FileSplit(rootDir);
        rr.initialize(fs);
        URI[] arr = fs.locations();

        assertTrue(rr.getLabels() == null || rr.getLabels().isEmpty());

        List<List<Writable>> expLabels = new ArrayList<>();
        for(URI u : arr){
            String path = u.getPath();
            expLabels.add(testMultiLabel(path.substring(path.length()-5, path.length())));
        }

        int count = 0;
        while(rr.hasNext()){
            List<Writable> l = rr.next();
            assertEquals(4, l.size());
            for( int i=0; i<3; i++ ){
                assertEquals(expLabels.get(count).get(i), l.get(i+1));
            }
            count++;
        }
        assertEquals(6, count);

        //Test batch ops:
        rr.reset();
        List<List<Writable>> b1 = rr.next(3);
        List<List<Writable>> b2 = rr.next(3);
        assertFalse(rr.hasNext());

        NDArrayRecordBatch b1a = (NDArrayRecordBatch)b1;
        NDArrayRecordBatch b2a = (NDArrayRecordBatch)b2;
        assertEquals(4, b1a.getArrays().size());
        assertEquals(4, b2a.getArrays().size());

        NDArrayWritable l1a = new NDArrayWritable(Nd4j.vstack(
                ((NDArrayWritable)expLabels.get(0).get(0)).get(),
                ((NDArrayWritable)expLabels.get(1).get(0)).get(),
                ((NDArrayWritable)expLabels.get(2).get(0)).get()));
        NDArrayWritable l1b = new NDArrayWritable(Nd4j.vstack(
                ((NDArrayWritable)expLabels.get(0).get(1)).get(),
                ((NDArrayWritable)expLabels.get(1).get(1)).get(),
                ((NDArrayWritable)expLabels.get(2).get(1)).get()));
        NDArrayWritable l1c = new NDArrayWritable(Nd4j.create(new double[]{
                expLabels.get(0).get(2).toDouble(),
                expLabels.get(1).get(2).toDouble(),
                expLabels.get(2).get(2).toDouble()}));


        NDArrayWritable l2a = new NDArrayWritable(Nd4j.vstack(
                ((NDArrayWritable)expLabels.get(3).get(0)).get(),
                ((NDArrayWritable)expLabels.get(4).get(0)).get(),
                ((NDArrayWritable)expLabels.get(5).get(0)).get()));
        NDArrayWritable l2b = new NDArrayWritable(Nd4j.vstack(
                ((NDArrayWritable)expLabels.get(3).get(1)).get(),
                ((NDArrayWritable)expLabels.get(4).get(1)).get(),
                ((NDArrayWritable)expLabels.get(5).get(1)).get()));
        NDArrayWritable l2c = new NDArrayWritable(Nd4j.create(new double[]{
                expLabels.get(3).get(2).toDouble(),
                expLabels.get(4).get(2).toDouble(),
                expLabels.get(5).get(2).toDouble()}));



        assertEquals(l1a.get(), b1a.getArrays().get(1));
        assertEquals(l1b.get(), b1a.getArrays().get(2));
        assertEquals(l1c.get(), b1a.getArrays().get(3));

        assertEquals(l2a.get(), b2a.getArrays().get(1));
        assertEquals(l2b.get(), b2a.getArrays().get(2));
        assertEquals(l2c.get(), b2a.getArrays().get(3));
    }

    private static class TestPathMultiLabelGenerator implements PathMultiLabelGenerator {

        @Override
        public List<Writable> getLabels(String uriPath) {
            String filename = uriPath.substring(uriPath.length()-5);
            return testMultiLabel(filename);
        }
    }

    private static List<Writable> testMultiLabel(String filename){
        switch(filename){
            case "0.jpg":
                return Arrays.<Writable>asList(new NDArrayWritable(Nd4j.create(new double[]{1,0})),
                        new NDArrayWritable(Nd4j.create(new double[]{1,0,0})), new DoubleWritable(0.0));
            case "1.png":
                return Arrays.<Writable>asList(new NDArrayWritable(Nd4j.create(new double[]{1,0})),
                        new NDArrayWritable(Nd4j.create(new double[]{0,1,0})), new DoubleWritable(1.0));
            case "2.jpg":
                return Arrays.<Writable>asList(new NDArrayWritable(Nd4j.create(new double[]{1,0})),
                        new NDArrayWritable(Nd4j.create(new double[]{0,0,1})), new DoubleWritable(2.0));
            case "A.jpg":
                return Arrays.<Writable>asList(new NDArrayWritable(Nd4j.create(new double[]{0,1})),
                        new NDArrayWritable(Nd4j.create(new double[]{1,0,0})), new DoubleWritable(3.0));
            case "B.png":
                return Arrays.<Writable>asList(new NDArrayWritable(Nd4j.create(new double[]{0,1})),
                        new NDArrayWritable(Nd4j.create(new double[]{0,1,0})), new DoubleWritable(4.0));
            case "C.jpg":
                return Arrays.<Writable>asList(new NDArrayWritable(Nd4j.create(new double[]{0,1})),
                        new NDArrayWritable(Nd4j.create(new double[]{0,0,1})), new DoubleWritable(5.0));
            default:
                throw new RuntimeException(filename);
        }
    }
}

