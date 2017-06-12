package org.datavec.spark.transform;

import org.datavec.spark.transform.model.CSVRecord;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by agibsonccc on 2/12/17.
 */
public class CSVRecordTest {

    @Test(expected = IllegalArgumentException.class)
    public void testVectorAssertion() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(1, 1));
        CSVRecord csvRecord = CSVRecord.fromRow(dataSet);
        fail(csvRecord.toString() + " should have thrown an exception");
    }

    @Test
    public void testVectorOneHotLabel() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(new double[][] {{0, 1}, {1, 0}}));

        //assert
        CSVRecord csvRecord = CSVRecord.fromRow(dataSet.get(0));
        assertEquals(3, csvRecord.getValues().length);

    }

    @Test
    public void testVectorRegression() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(new double[][] {{1, 1}, {1, 1}}));

        //assert
        CSVRecord csvRecord = CSVRecord.fromRow(dataSet.get(0));
        assertEquals(4, csvRecord.getValues().length);

    }

}
