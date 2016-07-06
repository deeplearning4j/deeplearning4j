package org.canova.api.records.writer.impl;

import org.canova.api.io.data.Text;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class CSVRecordWriterTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testWrite() throws Exception {
        File tempFile = File.createTempFile("canova","writer");
        tempFile.deleteOnExit();

        CSVRecordWriter writer = new CSVRecordWriter(tempFile);

        List<Writable> collection = new ArrayList<>();
        collection.add(new Text("12"));
        collection.add(new Text("13"));
        collection.add(new Text("14"));

        writer.write(collection);

        CSVRecordReader reader = new CSVRecordReader(0);
        reader.initialize(new FileSplit(tempFile));
        int cnt = 0;
        while (reader.hasNext()) {
            List<Writable> line = new ArrayList<>(reader.next());
            assertEquals(3, line.size());

            assertEquals(12, line.get(0).toInt());
            assertEquals(13, line.get(1).toInt());
            assertEquals(14, line.get(2).toInt());
            cnt++;
        }
        assertEquals(1, cnt);
    }
}