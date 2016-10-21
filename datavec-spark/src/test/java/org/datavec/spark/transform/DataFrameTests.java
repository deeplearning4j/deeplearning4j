package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
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
        for(int i = 0; i < 5; i++) {
            data.add(new ArrayList<Writable>());
            for(int j = 0; j < 6; j++) {
                data.get(i).add(new DoubleWritable(1.0));
            }

            builder.addColumnDouble(String.valueOf(i));
        }

        Schema schema = builder.build();
        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        assertEquals(schema,DataFrames.fromStructType(DataFrames.fromSchema(schema)));
        assertEquals(rdd.collect(),DataFrames.toRecords(DataFrames.toDataFrame(schema,rdd)).getSecond().collect());

    }


}
