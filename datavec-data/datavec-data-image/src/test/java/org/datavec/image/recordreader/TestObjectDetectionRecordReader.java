/*
 *  * Copyright 2017 Skymind, Inc.
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

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.junit.Test;

import java.io.File;
import java.net.URI;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestObjectDetectionRecordReader {

    @Test
    public void test() throws Exception {

        ImageObjectLabelProvider lp = new TestImageObjectDetectionLabelProvider();
        String path = new ClassPathResource("objdetect/000012.jpg").getFile().getParent();

        int gW = 13;
        int gH = 13;

        RecordReader rr = new ObjectDetectionRecordReader(64, 64, 3, gH, gW, lp);
        rr.initialize(new FileSplit(new File(path)));

        RecordReader imgRR = new ImageRecordReader(64, 64, 3);
        imgRR.initialize(new FileSplit(new File(path)));

        List<String> labels = rr.getLabels();
        assertEquals(Arrays.asList("car", "cat"), labels);

        assertEquals(imgRR.getLabels(), rr.getLabels());

        assertTrue(rr.hasNext());
        List<Writable> first = rr.next();
        assertEquals(2, first.size());
        assertTrue(first.get(0) instanceof NDArrayWritable);
        assertTrue(first.get(1) instanceof NDArrayWritable);

        //000012.jpg - originally 500x333
        double fracImageX1 = 156 / 500.0;
        double fracImageY1 = 97 / 333.0;
        double fracImageX2 = 351 / 500.0;
        double fracImageY2 = 270 / 333.0;

        double x1C = (fracImageX1 + fracImageX2)/2.0;
        double y1C = (fracImageY1 + fracImageY2)/2.0;

        int gridX = (int)(x1C * gW);
        int gridY = (int)(y1C * gH);

        //Check labels:


    }

    //2 images: 000012.jpg and 000019.jpg
    private static class TestImageObjectDetectionLabelProvider implements ImageObjectLabelProvider {

        @Override
        public List<ImageObject> getImageObjectsForPath(URI uri) {
            return getImageObjectsForPath(uri.getPath());
        }

        @Override
        public List<ImageObject> getImageObjectsForPath(String path) {
            if(path.endsWith("000012.jpg")){
                return Collections.singletonList(new ImageObject(156, 97, 351, 270, "car"));
            } else if(path.equals("000019.jpg")){
                return Arrays.asList(
                        new ImageObject(11, 113, 266, 259, "cat"),
                        new ImageObject(231, 88, 483, 256, "cat"));
            } else {
                throw new RuntimeException();
            }
        }
    }
}
