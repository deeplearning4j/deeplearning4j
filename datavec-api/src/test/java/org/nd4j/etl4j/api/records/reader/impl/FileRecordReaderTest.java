package org.nd4j.etl4j.api.records.reader.impl;

import org.nd4j.etl4j.api.split.FileSplit;
import org.nd4j.etl4j.api.writable.Writable;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by nyghtowl on 11/14/15.
 */
public class FileRecordReaderTest {

    @Test
    public void testReset() throws Exception {
        FileRecordReader rr = new FileRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("iris.dat").getFile()));

        int nResets = 5;
        for( int i=0; i < nResets; i++ ){

            int lineCount = 0;
            while(rr.hasNext()){
                Collection<Writable> line = rr.next();
                assertEquals(1, line.size());
                lineCount++;
            }
            assertFalse(rr.hasNext());
            assertEquals(1, lineCount);
            rr.reset();
        }
    }

}
