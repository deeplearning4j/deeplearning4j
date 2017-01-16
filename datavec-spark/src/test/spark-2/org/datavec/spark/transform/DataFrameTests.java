package org.datavec.spark.transform;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/21/16.
 */
public class DataFrameTests extends BaseSparkTest {


    @Test
    public void testMinMax() {
        INDArray arr = Nd4j.linspace(1,10,10).broadcast(10,10);
        for(int i = 0; i < arr.rows(); i++)
            arr.getRow(i).addi(i);

        List<List<Writable>> records = RecordConverter.toRecords(arr);
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 10;
        for(int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));
        Schema schema = builder.build();
        Dataset<Row> dataFrame = DataFrames.toDataFrame(schema,sc.parallelize(records));
        dataFrame.show();
        dataFrame.describe(DataFrames.toArray(schema.getColumnNames())).show();
//        System.out.println(Normalization.minMaxColumns(dataFrame,schema.getColumnNames()));
//        System.out.println(Normalization.stdDevMeanColumns(dataFrame,schema.getColumnNames()));

    }


    @Test
    public void testDataFrameConversions() {
        List<List<Writable>> data = new ArrayList<>();
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 6;
        for(int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));

        for(int i = 0; i < 5; i++) {
            List<Writable> record = new ArrayList<>(numColumns);
            data.add(record);
            for(int j = 0; j < numColumns; j++) {
                record.add(new DoubleWritable(1.0));
            }

        }

        Schema schema = builder.build();
        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        assertEquals(schema,DataFrames.fromStructType(DataFrames.fromSchema(schema)));
        assertEquals(rdd.collect(),DataFrames.toRecords(DataFrames.toDataFrame(schema,rdd)).getSecond().collect());

        Dataset<Row> dataFrame = DataFrames.toDataFrame(schema,rdd);
        dataFrame.show();
        Column mean = DataFrames.mean(dataFrame,"0");
        Column std = DataFrames.std(dataFrame,"0");
        dataFrame.withColumn("0",dataFrame.col("0").minus(mean)).show();
        dataFrame.withColumn("0",dataFrame.col("0").divide(std)).show();

        /*   DataFrame desc = dataFrame.describe(dataFrame.columns());
        dataFrame.show();
        System.out.println(dataFrame.agg(avg("0"), dataFrame.col("0")));
        dataFrame.withColumn("0",dataFrame.col("0").minus(avg(dataFrame.col("0"))));
        dataFrame.show();


        for(String column : dataFrame.columns()) {
            System.out.println(DataFrames.mean(desc,column));
            System.out.println(DataFrames.min(desc,column));
            System.out.println(DataFrames.max(desc,column));
            System.out.println(DataFrames.std(desc,column));

        }*/
    }



}
