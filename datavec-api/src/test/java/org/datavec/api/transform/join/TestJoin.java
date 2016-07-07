package org.datavec.api.transform.join;

import org.datavec.api.io.data.NullWritable;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.io.data.IntWritable;
import org.datavec.api.io.data.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 18/04/2016.
 */
public class TestJoin {

    @Test
    public void testJoin(){

        Schema firstSchema = new Schema.Builder()
                .addColumnString("keyColumn")
                .addColumnsInteger("first0","first1")
                .build();

        Schema secondSchema = new Schema.Builder()
                .addColumnString("keyColumn")
                .addColumnsInteger("second0")
                .build();

        List<List<Writable>> first = new ArrayList<>();
        first.add(Arrays.asList((Writable)new Text("key0"), new IntWritable(0), new IntWritable(1)));
        first.add(Arrays.asList((Writable)new Text("key1"), new IntWritable(10), new IntWritable(11)));

        List<List<Writable>> second = new ArrayList<>();
        second.add(Arrays.asList((Writable)new Text("key0"), new IntWritable(100)));
        second.add(Arrays.asList((Writable)new Text("key1"), new IntWritable(110)));

        Join join = new Join.Builder(Join.JoinType.Inner)
                .setKeyColumns("keyColumn")
                .setSchemas(firstSchema, secondSchema)
                .build();

        List<List<Writable>> expected = new ArrayList<>();
        expected.add(Arrays.asList((Writable)new Text("key0"), new IntWritable(0), new IntWritable(1), new IntWritable(100)));
        expected.add(Arrays.asList((Writable)new Text("key1"), new IntWritable(10), new IntWritable(11), new IntWritable(110)));


        //Check schema:
        Schema joinedSchema = join.getOutputSchema();
        assertEquals(4,joinedSchema.numColumns());
        assertEquals(Arrays.asList("keyColumn","first0","first1","second0"), joinedSchema.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.String, ColumnType.Integer, ColumnType.Integer, ColumnType.Integer), joinedSchema.getColumnTypes());


        //Check joining with null values:
        expected = new ArrayList<>();
        expected.add(Arrays.asList((Writable)new Text("key0"), new IntWritable(0), new IntWritable(1), NullWritable.INSTANCE));
        expected.add(Arrays.asList((Writable)new Text("key1"), new IntWritable(10), new IntWritable(11), NullWritable.INSTANCE));
        for( int i=0; i<first.size(); i++ ){
            List<Writable> out = join.joinExamples(first.get(i),null);
            assertEquals(expected.get(i), out);
        }

        expected = new ArrayList<>();
        expected.add(Arrays.asList((Writable)new Text("key0"), NullWritable.INSTANCE, NullWritable.INSTANCE, new IntWritable(100)));
        expected.add(Arrays.asList((Writable)new Text("key1"), NullWritable.INSTANCE, NullWritable.INSTANCE, new IntWritable(110)));
        for( int i=0; i<first.size(); i++ ){
            List<Writable> out = join.joinExamples(null,second.get(i));
            assertEquals(expected.get(i), out);
        }
    }

}
