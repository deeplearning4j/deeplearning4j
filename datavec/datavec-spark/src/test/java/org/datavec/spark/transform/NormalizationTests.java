package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Row;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/22/16.
 */
public class NormalizationTests extends BaseSparkTest {


    @Test
    public void testMeanStdZeros() {
        List<List<Writable>> data = new ArrayList<>();
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 6;
        for (int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));

        for (int i = 0; i < 5; i++) {
            List<Writable> record = new ArrayList<>(numColumns);
            data.add(record);
            for (int j = 0; j < numColumns; j++) {
                record.add(new DoubleWritable(1.0));
            }

        }

        INDArray arr = RecordConverter.toMatrix(data);

        Schema schema = builder.build();
        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        DataRowsFacade dataFrame = DataFrames.toDataFrame(schema, rdd);

        //assert equivalent to the ndarray pre processing
        NormalizerStandardize standardScaler = new NormalizerStandardize();
        standardScaler.fit(new DataSet(arr.dup(), arr.dup()));
        INDArray standardScalered = arr.dup();
        standardScaler.transform(new DataSet(standardScalered, standardScalered));
        DataNormalization zeroToOne = new NormalizerMinMaxScaler();
        zeroToOne.fit(new DataSet(arr.dup(), arr.dup()));
        INDArray zeroToOnes = arr.dup();
        zeroToOne.transform(new DataSet(zeroToOnes, zeroToOnes));
        List<Row> rows = Normalization.stdDevMeanColumns(dataFrame, dataFrame.get().columns());
        INDArray assertion = DataFrames.toMatrix(rows);
        //compare standard deviation
        assertTrue(standardScaler.getStd().equalsWithEps(assertion.getRow(0), 1e-1));
        //compare mean
        assertTrue(standardScaler.getMean().equalsWithEps(assertion.getRow(1), 1e-1));

    }



    @Test
    public void normalizationTests() {
        List<List<Writable>> data = new ArrayList<>();
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 6;
        for (int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));

        for (int i = 0; i < 5; i++) {
            List<Writable> record = new ArrayList<>(numColumns);
            data.add(record);
            for (int j = 0; j < numColumns; j++) {
                record.add(new DoubleWritable(1.0));
            }

        }

        INDArray arr = RecordConverter.toMatrix(data);

        Schema schema = builder.build();
        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        assertEquals(schema, DataFrames.fromStructType(DataFrames.fromSchema(schema)));
        assertEquals(rdd.collect(), DataFrames.toRecords(DataFrames.toDataFrame(schema, rdd)).getSecond().collect());

        DataRowsFacade dataFrame = DataFrames.toDataFrame(schema, rdd);
        dataFrame.get().show();
        Normalization.zeromeanUnitVariance(dataFrame).get().show();
        Normalization.normalize(dataFrame).get().show();

        //assert equivalent to the ndarray pre processing
        NormalizerStandardize standardScaler = new NormalizerStandardize();
        standardScaler.fit(new DataSet(arr.dup(), arr.dup()));
        INDArray standardScalered = arr.dup();
        standardScaler.transform(new DataSet(standardScalered, standardScalered));
        DataNormalization zeroToOne = new NormalizerMinMaxScaler();
        zeroToOne.fit(new DataSet(arr.dup(), arr.dup()));
        INDArray zeroToOnes = arr.dup();
        zeroToOne.transform(new DataSet(zeroToOnes, zeroToOnes));

        INDArray zeroMeanUnitVarianceDataFrame =
                        RecordConverter.toMatrix(Normalization.zeromeanUnitVariance(schema, rdd).collect());
        INDArray zeroMeanUnitVarianceDataFrameZeroToOne =
                        RecordConverter.toMatrix(Normalization.normalize(schema, rdd).collect());
        assertEquals(standardScalered, zeroMeanUnitVarianceDataFrame);
        assertTrue(zeroToOnes.equalsWithEps(zeroMeanUnitVarianceDataFrameZeroToOne, 1e-1));

    }

}
