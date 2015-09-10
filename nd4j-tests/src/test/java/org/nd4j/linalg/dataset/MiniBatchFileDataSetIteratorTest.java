package org.nd4j.linalg.dataset;

import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;


/**
 * Created by agibsonccc on 9/10/15.
 */
public class MiniBatchFileDataSetIteratorTest extends BaseNd4jTest {
    @Test
    public void testMiniBatches() throws Exception {
        DataSet load = new IrisDataSetIterator(150,150).next();
        final MiniBatchFileDataSetIterator iter = new MiniBatchFileDataSetIterator(load,10,false);
        while(iter.hasNext())
            assertEquals(10,iter.next().numExamples());

        DataSetIterator existing = new ExistingMiniBatchDataSetIterator(iter.getRootDir());
        while(iter.hasNext())
            assertEquals(10,existing.next().numExamples());


        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    FileUtils.deleteDirectory(iter.getRootDir());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }));

    }

    @Override
    public char ordering() {
        return 'f';
    }
}

