/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.datavec.api.records.reader.impl;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.impl.filebatch.FileBatchRecordReader;
import org.datavec.api.records.reader.impl.filebatch.FileBatchSequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.api.loader.FileBatch;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class FileBatchRecordReaderTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testCsv() throws Exception {

        //This is an unrealistic use case - one line/record per CSV
        File baseDir = testDir.newFolder();

        List<File> fileList = new ArrayList<>();
        for( int i=0; i<10; i++ ){
            String s = "file_" + i + "," + i + "," + i;
            File f = new File(baseDir, "origFile" + i + ".csv");
            FileUtils.writeStringToFile(f, s, StandardCharsets.UTF_8);
            fileList.add(f);
        }

        FileBatch fb = FileBatch.forFiles(fileList);

        RecordReader rr = new CSVRecordReader();
        FileBatchRecordReader fbrr = new FileBatchRecordReader(rr, fb);


        for( int test=0; test<3; test++) {
            for (int i = 0; i < 10; i++) {
                assertTrue(fbrr.hasNext());
                List<Writable> next = fbrr.next();
                assertEquals(3, next.size());
                String s1 = "file_" + i;
                assertEquals(s1, next.get(0).toString());
                assertEquals(String.valueOf(i), next.get(1).toString());
                assertEquals(String.valueOf(i), next.get(2).toString());
            }
            assertFalse(fbrr.hasNext());
            assertTrue(fbrr.resetSupported());
            fbrr.reset();
        }
    }

    @Test
    public void testCsvSequence() throws Exception {
        //CSV sequence - 3 lines per file, 10 files
        File baseDir = testDir.newFolder();

        List<File> fileList = new ArrayList<>();
        for( int i=0; i<10; i++ ){
            StringBuilder sb = new StringBuilder();
            for( int j=0; j<3; j++ ){
                if(j > 0)
                    sb.append("\n");
                sb.append("file_" + i + "," + i + "," + j);
            }
            File f = new File(baseDir, "origFile" + i + ".csv");
            FileUtils.writeStringToFile(f, sb.toString(), StandardCharsets.UTF_8);
            fileList.add(f);
        }

        FileBatch fb = FileBatch.forFiles(fileList);
        SequenceRecordReader rr = new CSVSequenceRecordReader();
        FileBatchSequenceRecordReader fbrr = new FileBatchSequenceRecordReader(rr, fb);


        for( int test=0; test<3; test++) {
            for (int i = 0; i < 10; i++) {
                assertTrue(fbrr.hasNext());
                List<List<Writable>> next = fbrr.sequenceRecord();
                assertEquals(3, next.size());
                int count = 0;
                for(List<Writable> step : next ){
                    String s1 = "file_" + i;
                    assertEquals(s1, step.get(0).toString());
                    assertEquals(String.valueOf(i), step.get(1).toString());
                    assertEquals(String.valueOf(count++), step.get(2).toString());
                }
            }
            assertFalse(fbrr.hasNext());
            assertTrue(fbrr.resetSupported());
            fbrr.reset();
        }
    }

}
