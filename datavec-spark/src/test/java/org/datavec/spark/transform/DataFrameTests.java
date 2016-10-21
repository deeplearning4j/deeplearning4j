package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
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
        DataFrame desc = dataFrame.describe(dataFrame.columns());
        desc.show();
        System.out.println(desc.col("0").minus(DataFrames.mean(desc,"0")).desc())   ;
        for(String column : dataFrame.columns()) {
            System.out.println(DataFrames.mean(desc,column));
            System.out.println(DataFrames.min(desc,column));
            System.out.println(DataFrames.max(desc,column));
            System.out.println(DataFrames.std(desc,column));

        }


    }


}
