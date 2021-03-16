/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.datavec.api.records.reader.impl;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.impl.filebatch.FileBatchRecordReader;
import org.datavec.api.records.reader.impl.filebatch.FileBatchSequenceRecordReader;
import org.datavec.api.writable.Writable;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.loader.FileBatch;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("File Batch Record Reader Test")
class FileBatchRecordReaderTest extends BaseND4JTest {

    @Test
    @DisplayName("Test Csv")
    void testCsv(@TempDir  Path testDir) throws Exception {
        // This is an unrealistic use case - one line/record per CSV
        File baseDir = testDir.toFile();
        List<File> fileList = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            String s = "file_" + i + "," + i + "," + i;
            File f = new File(baseDir, "origFile" + i + ".csv");
            FileUtils.writeStringToFile(f, s, StandardCharsets.UTF_8);
            fileList.add(f);
        }
        FileBatch fb = FileBatch.forFiles(fileList);
        RecordReader rr = new CSVRecordReader();
        FileBatchRecordReader fbrr = new FileBatchRecordReader(rr, fb);
        for (int test = 0; test < 3; test++) {
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
    @DisplayName("Test Csv Sequence")
    void testCsvSequence(@TempDir  Path testDir) throws Exception {
        // CSV sequence - 3 lines per file, 10 files
        File baseDir = testDir.toFile();
        List<File> fileList = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < 3; j++) {
                if (j > 0)
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
        for (int test = 0; test < 3; test++) {
            for (int i = 0; i < 10; i++) {
                assertTrue(fbrr.hasNext());
                List<List<Writable>> next = fbrr.sequenceRecord();
                assertEquals(3, next.size());
                int count = 0;
                for (List<Writable> step : next) {
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
