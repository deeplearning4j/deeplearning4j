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

package org.datavec.api.records.reader.impl;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 11/17/14.
 */
public class LineReaderTest {

    private static Logger log = LoggerFactory.getLogger(LineReaderTest.class);

    @Test
    public void testLineReader() throws Exception {
        String tempDir = System.getProperty("java.io.tmpdir");
        File tmpdir = new File(tempDir, "tmpdir-testLineReader");
        if (tmpdir.exists())
            tmpdir.delete();
        tmpdir.mkdir();

        File tmp1 = new File(FilenameUtils.concat(tmpdir.getPath(), "tmp1.txt"));
        File tmp2 = new File(FilenameUtils.concat(tmpdir.getPath(), "tmp2.txt"));
        File tmp3 = new File(FilenameUtils.concat(tmpdir.getPath(), "tmp3.txt"));

        FileUtils.writeLines(tmp1, Arrays.asList("1", "2", "3"));
        FileUtils.writeLines(tmp2, Arrays.asList("4", "5", "6"));
        FileUtils.writeLines(tmp3, Arrays.asList("7", "8", "9"));

        InputSplit split = new FileSplit(tmpdir);

        RecordReader reader = new LineRecordReader();
        reader.initialize(split);

        int count = 0;
        List<List<Writable>> list = new ArrayList<>();
        while (reader.hasNext()) {
            List<Writable> l = reader.next();
            assertEquals(1, l.size());
            list.add(l);
            count++;
        }

        assertEquals(9, count);

        try {
            FileUtils.deleteDirectory(tmpdir);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testLineReaderMetaData() throws Exception {
        String tempDir = System.getProperty("java.io.tmpdir");
        File tmpdir = new File(tempDir, "tmpdir-testLineReader");
        if (tmpdir.exists())
            tmpdir.delete();
        tmpdir.mkdir();

        File tmp1 = new File(FilenameUtils.concat(tmpdir.getPath(), "tmp1.txt"));
        File tmp2 = new File(FilenameUtils.concat(tmpdir.getPath(), "tmp2.txt"));
        File tmp3 = new File(FilenameUtils.concat(tmpdir.getPath(), "tmp3.txt"));

        FileUtils.writeLines(tmp1, Arrays.asList("1", "2", "3"));
        FileUtils.writeLines(tmp2, Arrays.asList("4", "5", "6"));
        FileUtils.writeLines(tmp3, Arrays.asList("7", "8", "9"));

        InputSplit split = new FileSplit(tmpdir);

        RecordReader reader = new LineRecordReader();
        reader.initialize(split);

        List<List<Writable>> list = new ArrayList<>();
        while (reader.hasNext()) {
            list.add(reader.next());
        }
        assertEquals(9, list.size());


        List<List<Writable>> out2 = new ArrayList<>();
        List<Record> out3 = new ArrayList<>();
        List<RecordMetaData> meta = new ArrayList<>();
        reader.reset();
        int count = 0;
        while (reader.hasNext()) {
            Record r = reader.nextRecord();
            out2.add(r.getRecord());
            out3.add(r);
            meta.add(r.getMetaData());
            int fileIdx = count / 3;
            URI uri = r.getMetaData().getURI();
            assertEquals(uri, split.locations()[fileIdx]);
            count++;
        }

        assertEquals(list, out2);

        List<Record> fromMeta = reader.loadFromMetaData(meta);
        assertEquals(out3, fromMeta);

        //try: second line of second and third files only...
        List<RecordMetaData> subsetMeta = new ArrayList<>();
        subsetMeta.add(meta.get(4));
        subsetMeta.add(meta.get(7));
        List<Record> subset = reader.loadFromMetaData(subsetMeta);
        assertEquals(2, subset.size());
        assertEquals(out3.get(4), subset.get(0));
        assertEquals(out3.get(7), subset.get(1));


        try {
            FileUtils.deleteDirectory(tmpdir);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testLineReaderWithInputStreamInputSplit() throws Exception {
        String tempDir = System.getProperty("java.io.tmpdir");
        File tmpdir = new File(tempDir, "tmpdir");
        tmpdir.mkdir();

        File tmp1 = new File(tmpdir, "tmp1.txt.gz");

        OutputStream os = new GZIPOutputStream(new FileOutputStream(tmp1, false));
        IOUtils.writeLines(Arrays.asList("1", "2", "3", "4", "5", "6", "7", "8", "9"), null, os);
        os.flush();
        os.close();

        InputSplit split = new InputStreamInputSplit(new GZIPInputStream(new FileInputStream(tmp1)));

        RecordReader reader = new LineRecordReader();
        reader.initialize(split);

        int count = 0;
        while (reader.hasNext()) {
            assertEquals(1, reader.next().size());
            count++;
        }

        assertEquals(9, count);

        try {
            FileUtils.deleteDirectory(tmpdir);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
