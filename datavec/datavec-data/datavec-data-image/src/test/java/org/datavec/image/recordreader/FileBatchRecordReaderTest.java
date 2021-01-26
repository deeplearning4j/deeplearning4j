/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.image.recordreader;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.reader.impl.filebatch.FileBatchRecordReader;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.common.loader.FileBatch;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.util.*;

import static org.junit.Assert.*;

public class FileBatchRecordReaderTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testCsv() throws Exception {
        File extractedSourceDir = testDir.newFolder();
        new ClassPathResource("datavec-data-image/testimages").copyDirectory(extractedSourceDir);
        File baseDir = testDir.newFolder();


        List<File> c = new ArrayList<>(FileUtils.listFiles(extractedSourceDir, null, true));
        assertEquals(6, c.size());

        Collections.sort(c, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                return o1.getPath().compareTo(o2.getPath());
            }
        });


        FileBatch fb = FileBatch.forFiles(c);
        File saveFile = new File(baseDir, "saved.zip");
        fb.writeAsZip(saveFile);
        fb = FileBatch.readFromZip(saveFile);

        PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(32, 32, 1, labelMaker);
        rr.setLabels(Arrays.asList("class0", "class1"));
        FileBatchRecordReader fbrr = new FileBatchRecordReader(rr, fb);


        NativeImageLoader il = new NativeImageLoader(32, 32, 1);
        for( int test=0; test<3; test++) {
            for (int i = 0; i < 6; i++) {
                assertTrue(fbrr.hasNext());
                List<Writable> next = fbrr.next();
                assertEquals(2, next.size());

                INDArray exp;
                switch (i){
                    case 0:
                        exp = il.asMatrix(new File(extractedSourceDir, "class0/0.jpg"));
                        break;
                    case 1:
                        exp = il.asMatrix(new File(extractedSourceDir, "class0/1.png"));
                        break;
                    case 2:
                        exp = il.asMatrix(new File(extractedSourceDir, "class0/2.jpg"));
                        break;
                    case 3:
                        exp = il.asMatrix(new File(extractedSourceDir, "class1/A.jpg"));
                        break;
                    case 4:
                        exp = il.asMatrix(new File(extractedSourceDir, "class1/B.png"));
                        break;
                    case 5:
                        exp = il.asMatrix(new File(extractedSourceDir, "class1/C.jpg"));
                        break;
                    default:
                        throw new RuntimeException();
                }
                Writable expLabel = (i < 3 ? new IntWritable(0) : new IntWritable(1));

                assertEquals(((NDArrayWritable)next.get(0)).get(), exp);
                assertEquals(expLabel, next.get(1));
            }
            assertFalse(fbrr.hasNext());
            assertTrue(fbrr.resetSupported());
            fbrr.reset();
        }
    }

}
