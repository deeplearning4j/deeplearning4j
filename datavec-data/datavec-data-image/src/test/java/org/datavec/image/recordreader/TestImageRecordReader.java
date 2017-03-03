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
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Created by Alex on 27/09/2016.
 */
public class TestImageRecordReader {

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
}
