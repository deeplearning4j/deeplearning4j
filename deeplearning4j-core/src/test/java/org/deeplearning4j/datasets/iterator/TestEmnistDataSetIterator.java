package org.deeplearning4j.datasets.iterator;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.base.EmnistFetcher;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;

import static org.junit.Assert.*;

/**
 * Created by Alex on 28/07/2017.
 */
@Slf4j
public class TestEmnistDataSetIterator {

    @Test
    public void testEmnistDataSetIterator() throws Exception {

        EmnistFetcher fetcher = new EmnistFetcher(EmnistDataSetIterator.Set.COMPLETE);
        File baseEmnistDir = fetcher.getFILE_DIR();
        if(baseEmnistDir.exists()){
            FileUtils.deleteDirectory(baseEmnistDir);
        }
        assertFalse(baseEmnistDir.exists());


        int batchSize = 128;

        for(EmnistDataSetIterator.Set s : EmnistDataSetIterator.Set.values()){
            for(boolean train : new boolean[]{true, false}){
                log.info("Starting test: {}, {}", s, (train ? "train" : "test"));
                EmnistDataSetIterator iter = new EmnistDataSetIterator(s, batchSize, train, 12345);

                assertTrue(iter.asyncSupported());
                assertTrue(iter.resetSupported());

                int expNumExamples;
                if(train){
                    expNumExamples = EmnistDataSetIterator.numExamplesTrain(s);
                } else {
                    expNumExamples = EmnistDataSetIterator.numExamplesTest(s);
                }

                assertEquals(expNumExamples, iter.numExamples());

                int numLabels = EmnistDataSetIterator.numLabels(s);

                assertEquals(numLabels, iter.getLabels().size());
                assertEquals(numLabels, iter.getLabelsArrays().length);

                char[] labelArr = iter.getLabelsArrays();
                for(char c : labelArr){
                    boolean isExpected = (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
                    assertTrue(isExpected);
                }

                int totalCount = 0;
                while(iter.hasNext()){
                    DataSet ds = iter.next();
                    assertNotNull(ds.getFeatures());
                    assertNotNull(ds.getLabels());
                    assertEquals(ds.getFeatures().size(0), ds.getLabels().size(0));

                    totalCount += ds.getFeatures().size(0);

                    assertEquals(784, ds.getFeatures().size(1));
                    assertEquals(numLabels, ds.getLabels().size(1));
                }

                assertEquals(expNumExamples, totalCount);
            }
        }
    }

}
