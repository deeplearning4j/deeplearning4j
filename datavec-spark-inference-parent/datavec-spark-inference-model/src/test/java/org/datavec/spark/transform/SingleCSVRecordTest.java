package org.datavec.spark.transform;

import org.datavec.spark.transform.model.SingleCSVRecord;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by agibsonccc on 2/12/17.
 */
public class SingleCSVRecordTest {

    @Test(expected = IllegalArgumentException.class)
    public void testVectorAssertion() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(1, 1));
        SingleCSVRecord singleCsvRecord = SingleCSVRecord.fromRow(dataSet);
        fail(singleCsvRecord.toString() + " should have thrown an exception");
    }

    @Test
    public void testVectorOneHotLabel() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(new double[][] {{0, 1}, {1, 0}}));

        //assert
        SingleCSVRecord singleCsvRecord = SingleCSVRecord.fromRow(dataSet.get(0));
        assertEquals(3, singleCsvRecord.getValues().size());

    }

    @Test
    public void testVectorRegression() {
        DataSet dataSet = new DataSet(Nd4j.create(2, 2), Nd4j.create(new double[][] {{1, 1}, {1, 1}}));

        //assert
        SingleCSVRecord singleCsvRecord = SingleCSVRecord.fromRow(dataSet.get(0));
        assertEquals(4, singleCsvRecord.getValues().size());

    }

}
