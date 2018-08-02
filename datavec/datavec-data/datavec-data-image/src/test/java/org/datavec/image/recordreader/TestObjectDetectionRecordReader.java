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

package org.datavec.image.recordreader;

import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;

import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.*;

public class TestObjectDetectionRecordReader {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void test() throws Exception {
        ImageObjectLabelProvider lp = new TestImageObjectDetectionLabelProvider();

        File f = testDir.newFolder();
        new ClassPathResource("datavec-data-image/objdetect/").copyDirectory(f);

        String path = new File(f, "000012.jpg").getParent();

        int h = 32;
        int w = 32;
        int c = 3;
        int gW = 13;
        int gH = 10;

        //Enforce consistent iteration order for tests
        URI[] u = new FileSplit(new File(path)).locations();
        Arrays.sort(u);

        RecordReader rr = new ObjectDetectionRecordReader(h, w, c, gH, gW, lp);
        rr.initialize(new CollectionInputSplit(u));

        RecordReader imgRR = new ImageRecordReader(h, w, c);
        imgRR.initialize(new CollectionInputSplit(u));

        List<String> labels = rr.getLabels();
        assertEquals(Arrays.asList("car", "cat"), labels);


        //000012.jpg - originally 500x333
        //000019.jpg - originally 500x375
        double[] origW = new double[]{500, 500};
        double[] origH = new double[]{333, 375};
        List<List<ImageObject>> l = Arrays.asList(
                Collections.singletonList(new ImageObject(156, 97, 351, 270, "car")),
                Arrays.asList(new ImageObject(11, 113, 266, 259, "cat"), new ImageObject(231, 88, 483, 256, "cat"))
        );

        for (int idx = 0; idx < 2; idx++) {
            assertTrue(rr.hasNext());
            List<Writable> next = rr.next();
            List<Writable> nextImgRR = imgRR.next();

            //Check features:
            assertEquals(next.get(0), nextImgRR.get(0));

            //Check labels
            assertEquals(2, next.size());
            assertTrue(next.get(0) instanceof NDArrayWritable);
            assertTrue(next.get(1) instanceof NDArrayWritable);

            List<ImageObject> objects = l.get(idx);

            INDArray expLabels = Nd4j.create(1, 4 + 2, gH, gW);
            for (ImageObject io : objects) {
                double fracImageX1 = io.getX1() / origW[idx];
                double fracImageY1 = io.getY1() / origH[idx];
                double fracImageX2 = io.getX2() / origW[idx];
                double fracImageY2 = io.getY2() / origH[idx];

                double x1C = (fracImageX1 + fracImageX2) / 2.0;
                double y1C = (fracImageY1 + fracImageY2) / 2.0;

                int labelGridX = (int) (x1C * gW);
                int labelGridY = (int) (y1C * gH);

                int labelIdx;
                if (io.getLabel().equals("car")) {
                    labelIdx = 4;
                } else {
                    labelIdx = 5;
                }
                expLabels.putScalar(0, labelIdx, labelGridY, labelGridX, 1.0);

                expLabels.putScalar(0, 0, labelGridY, labelGridX, fracImageX1 * gW);
                expLabels.putScalar(0, 1, labelGridY, labelGridX, fracImageY1 * gH);
                expLabels.putScalar(0, 2, labelGridY, labelGridX, fracImageX2 * gW);
                expLabels.putScalar(0, 3, labelGridY, labelGridX, fracImageY2 * gH);
            }

            INDArray lArr = ((NDArrayWritable) next.get(1)).get();
            assertArrayEquals(new long[]{1, 4 + 2, gH, gW}, lArr.shape());
            assertEquals(expLabels, lArr);
        }

        rr.reset();
        Record record = rr.nextRecord();
        RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)record.getMetaData();
        assertEquals(new File(path, "000012.jpg"), new File(metadata.getURI()));
        assertEquals(3, metadata.getOrigC());
        assertEquals((int)origH[0], metadata.getOrigH());
        assertEquals((int)origW[0], metadata.getOrigW());

        List<Record> out = new ArrayList<>();
        List<RecordMetaData> meta = new ArrayList<>();
        out.add(record);
        meta.add(metadata);
        record = rr.nextRecord();
        metadata = (RecordMetaDataImageURI)record.getMetaData();
        out.add(record);
        meta.add(metadata);

        List<Record> fromMeta = rr.loadFromMetaData(meta);
        assertEquals(out, fromMeta);

        // make sure we don't lose objects just by explicitly resizing
        int i = 0;
        int[] nonzeroCount = {5, 10};

        ImageTransform transform = new ResizeImageTransform(37, 42);
        RecordReader rrTransform = new ObjectDetectionRecordReader(42, 37, c, gH, gW, lp, transform);
        rrTransform.initialize(new CollectionInputSplit(u));
        i = 0;
        while (rrTransform.hasNext()) {
            List<Writable> next = rrTransform.next();
            assertEquals(37, transform.getCurrentImage().getWidth());
            assertEquals(42, transform.getCurrentImage().getHeight());
            INDArray labelArray = ((NDArrayWritable)next.get(1)).get();
            BooleanIndexing.applyWhere(labelArray, Conditions.notEquals(0), new Value(1));
            assertEquals(nonzeroCount[i++], labelArray.ravel().sum(1).getInt(0));
        }

        ImageTransform transform2 = new ResizeImageTransform(1024, 2048);
        RecordReader rrTransform2 = new ObjectDetectionRecordReader(2048, 1024, c, gH, gW, lp, transform2);
        rrTransform2.initialize(new CollectionInputSplit(u));
        i = 0;
        while (rrTransform2.hasNext()) {
            List<Writable> next = rrTransform2.next();
            assertEquals(1024, transform2.getCurrentImage().getWidth());
            assertEquals(2048, transform2.getCurrentImage().getHeight());
            INDArray labelArray = ((NDArrayWritable)next.get(1)).get();
            BooleanIndexing.applyWhere(labelArray, Conditions.notEquals(0), new Value(1));
            assertEquals(nonzeroCount[i++], labelArray.ravel().sum(1).getInt(0));
        }
        
        //Make sure image flip does not break labels and are correct for new image size dimensions:
        ImageTransform transform3 = new PipelineImageTransform(
                new ResizeImageTransform(2048, 4096),
                new FlipImageTransform(-1)
        );
        RecordReader rrTransform3 = new ObjectDetectionRecordReader(2048, 1024, c, gH, gW, lp, transform3);
        rrTransform3.initialize(new CollectionInputSplit(u));
        i = 0;
        while (rrTransform3.hasNext()) {
            List<Writable> next = rrTransform3.next();
            INDArray labelArray = ((NDArrayWritable)next.get(1)).get();
            BooleanIndexing.applyWhere(labelArray, Conditions.notEquals(0), new Value(1));
            assertEquals(nonzeroCount[i++], labelArray.ravel().sum(1).getInt(0));
        }
        
        //Test that doing a downscale with the native image loader directly instead of a transform does not cause an exception:
        ImageTransform transform4 = new FlipImageTransform(-1);
        RecordReader rrTransform4 = new ObjectDetectionRecordReader(128, 128, c, gH, gW, lp, transform4);
        rrTransform4.initialize(new CollectionInputSplit(u));
        i = 0;
        while (rrTransform4.hasNext()) {
            List<Writable> next = rrTransform4.next();
            
            assertEquals((int) origW[i], transform4.getCurrentImage().getWidth());
            assertEquals((int) origH[i], transform4.getCurrentImage().getHeight());
            
            INDArray labelArray = ((NDArrayWritable)next.get(1)).get();
            BooleanIndexing.applyWhere(labelArray, Conditions.notEquals(0), new Value(1));
            assertEquals(nonzeroCount[i++], labelArray.ravel().sum(1).getInt(0));
        }
    }

    //2 images: 000012.jpg and 000019.jpg
    private static class TestImageObjectDetectionLabelProvider implements ImageObjectLabelProvider {

        @Override
        public List<ImageObject> getImageObjectsForPath(URI uri) {
            return getImageObjectsForPath(uri.getPath());
        }

        @Override
        public List<ImageObject> getImageObjectsForPath(String path) {
            if (path.endsWith("000012.jpg")) {
                return Collections.singletonList(new ImageObject(156, 97, 351, 270, "car"));
            } else if (path.endsWith("000019.jpg")) {
                return Arrays.asList(
                        new ImageObject(11, 113, 266, 259, "cat"),
                        new ImageObject(231, 88, 483, 256, "cat"));
            } else {
                throw new RuntimeException();
            }
        }
    }
}
