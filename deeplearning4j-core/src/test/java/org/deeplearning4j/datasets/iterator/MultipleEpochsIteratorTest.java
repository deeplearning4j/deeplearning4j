package org.deeplearning4j.datasets.iterator;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by nyghtowl on 11/14/15.
 */

public class MultipleEpochsIteratorTest {

    protected File file1, file2, file3, file4, newPath;
    protected static String localPath = System.getProperty("java.io.tmpdir") + File.separator;
    protected static String testPath = localPath + "test-folder" + File.separator;


    @Before
    @Ignore
    public void doBefore() throws IOException {
        newPath = new File(testPath);

        newPath.mkdir();

        file1 = File.createTempFile("myfile_1", ".jpg", newPath);
        file2 = File.createTempFile("myfile_2", ".txt", newPath);
        file3 = File.createTempFile("myfile_3", ".jpg", newPath);
        file4 = File.createTempFile("treehouse_4", ".jpg", newPath);
    }


    //TODO structure needs work to make this class function using reset for total dataset
    // Will work with ImageNetRecordReader only at this time

    @Test
    @Ignore
    public void testNextAndReset() throws Exception{
        int epochs = 2;
        int batchSize = 2;

        RecordReader recordReader = new FileRecordReader();
        recordReader.initialize(new FileSplit(new File(testPath)));
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize);
        MultipleEpochsIterator multiIter = new MultipleEpochsIterator(epochs, iter);

        assertTrue(multiIter.hasNext());
        while(multiIter.hasNext()){
            multiIter.next();
        }
        assertEquals(epochs, multiIter.numPasses, 0.0);

    }


    @After
    @Ignore
    public void doAfter(){
        file1.delete();
        file2.delete();
        file3.delete();
        file4.delete();
        newPath.delete();
    }



}
