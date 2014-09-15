package org.deeplearning4j.util;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Created by mjk on 9/15/14.
 */
public class ReadWriteObjectTest {
    @Test
    public void testWriteRead() {
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        String irisData = new String("irisData.dat");

        DataSet freshDataSet = iter.next(150);

        SerializationUtils.saveObject(freshDataSet, new File(irisData));

        DataSet readDataSet = SerializationUtils.readObject(new File(irisData));

        assertEquals(freshDataSet.getFeatureMatrix(),readDataSet.getFeatureMatrix());
        assertEquals(freshDataSet.getLabels(), readDataSet.getLabels());
        try {
            FileUtils.forceDelete(new File(irisData));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
