package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.RowFactory;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import static org.apache.spark.sql.functions.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/21/16.
 */
public class DataFrameTests extends BaseSparkTest {



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

        DataFrame dataFrame = DataFrames.toDataFrame(schema,rdd);
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

    @Test
    public void testNormalize(){

        List<List<Writable>> data = new ArrayList<>();

        data.add(Arrays.<Writable>asList(new IntWritable(1), new DoubleWritable(10)));
        data.add(Arrays.<Writable>asList(new IntWritable(2), new DoubleWritable(20)));
        data.add(Arrays.<Writable>asList(new IntWritable(3), new DoubleWritable(30)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(data);

        Schema schema = new Schema.Builder()
                .addColumnInteger("c0")
                .addColumnDouble("c1")
                .build();


        JavaRDD<List<Writable>> normalized = Normalization.normalize(schema, rdd);
        JavaRDD<List<Writable>> standardize = Normalization.zeromeanUnitVariance(schema, rdd);

        System.out.println("Normalized:");
        System.out.println(normalized.collect());
        System.out.println("Standardized:");
        System.out.println(standardize.collect());
    }


    @Test
    public void testDataFrameSequenceNormalization(){
        List<List<List<Writable>>> sequences = new ArrayList<>();

        List<List<Writable>> seq1 = new ArrayList<>();
        seq1.add(Arrays.<Writable>asList(new IntWritable(1), new DoubleWritable(1), new LongWritable(1)));
        seq1.add(Arrays.<Writable>asList(new IntWritable(2), new DoubleWritable(2), new LongWritable(2)));
        seq1.add(Arrays.<Writable>asList(new IntWritable(3), new DoubleWritable(3), new LongWritable(3)));

        List<List<Writable>> seq2 = new ArrayList<>();
        seq2.add(Arrays.<Writable>asList(new IntWritable(4), new DoubleWritable(4), new LongWritable(4)));
        seq2.add(Arrays.<Writable>asList(new IntWritable(5), new DoubleWritable(5), new LongWritable(5)));

        sequences.add(seq1);
        sequences.add(seq2);

        Schema schema = new Schema.Builder()
                .addColumnInteger("c0")
                .addColumnDouble("c1")
                .addColumnLong("c2")
                .build();

        JavaRDD<List<List<Writable>>> rdd = sc.parallelize(sequences);

        JavaRDD<List<List<Writable>>> normalized = Normalization.normalizeSequence(schema, rdd, 0, 1);

        List<List<List<Writable>>> norm = normalized.collect();


        assertEquals(2, norm.size());

        for(List<List<Writable>> l : norm){
            System.out.println(l);
            System.out.println();
        }

    }


}
