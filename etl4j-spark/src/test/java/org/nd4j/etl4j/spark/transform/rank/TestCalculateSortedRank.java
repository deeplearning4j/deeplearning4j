package org.nd4j.etl4j.spark.transform.rank;

import io.skymind.echidna.spark.BaseSparkTest;
import org.apache.spark.api.java.JavaRDD;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.ColumnType;
import io.skymind.echidna.api.TransformProcess;
import org.nd4j.etl4j.api.transform.comparator.DoubleWritableComparator;
import org.nd4j.etl4j.api.transform.schema.Schema;
import io.skymind.echidna.spark.SparkTransformExecutor;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 1/06/2016.
 */
public class TestCalculateSortedRank extends BaseSparkTest {

    @Test
    public void testCalculateSortedRank(){

        List<List<Writable>> data = new ArrayList<>();
        data.add(Arrays.asList((Writable)new Text("0"),new DoubleWritable(0.0)));
        data.add(Arrays.asList((Writable)new Text("3"),new DoubleWritable(0.3)));
        data.add(Arrays.asList((Writable)new Text("2"),new DoubleWritable(0.2)));
        data.add(Arrays.asList((Writable)new Text("1"),new DoubleWritable(0.1)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(data);

        Schema schema = new Schema.Builder()
                .addColumnsString("TextCol")
                .addColumnDouble("DoubleCol")
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .calculateSortedRank("rank","DoubleCol",new DoubleWritableComparator())
                .build();

        Schema outSchema = tp.getFinalSchema();
        assertEquals(3, outSchema.numColumns());
        assertEquals(Arrays.asList("TextCol","DoubleCol","rank"), outSchema.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.String, ColumnType.Double, ColumnType.Long), outSchema.getColumnTypes());

        SparkTransformExecutor exec = new SparkTransformExecutor();
        JavaRDD<List<Writable>> out = exec.execute(rdd, tp);

        List<List<Writable>> collected = out.collect();
        assertEquals(4, collected.size());
        for( int i=0; i<4; i++ ) assertEquals(3, collected.get(i).size());

        for(List<Writable> example : collected){
            int exampleNum = example.get(0).toInt();
            int rank = example.get(2).toInt();
            assertEquals(exampleNum, rank);
        }
    }

}
