package org.nd4j.etl4j.api.transform.condition;

import org.nd4j.etl4j.api.transform.ColumnType;
import org.nd4j.etl4j.api.transform.condition.column.*;
import org.nd4j.etl4j.api.transform.condition.string.StringRegexColumnCondition;
import org.nd4j.etl4j.api.transform.schema.Schema;
import org.nd4j.etl4j.api.transform.transform.TestTransforms;
import org.nd4j.etl4j.api.io.data.*;
import org.nd4j.etl4j.api.writable.Writable;
import org.junit.Test;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 24/03/2016.
 */
public class TestConditions {

    @Test
    public void testIntegerCondition(){
        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessThan, 0);
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new IntWritable(-1))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new IntWritable(-2))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(0))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(1))));

        Set<Integer> set = new HashSet<>();
        set.add(0);
        set.add(3);
        condition = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable)new IntWritable(0))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new IntWritable(3))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(1))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(2))));
    }

    @Test
    public void testLongCondition(){
        Schema schema = TestTransforms.getSchema(ColumnType.Long);

        Condition condition = new LongColumnCondition("column", SequenceConditionMode.Or, ConditionOp.NotEqual, 5L);
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new LongWritable(0))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new LongWritable(1))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new LongWritable(5))));

        Set<Long> set = new HashSet<>();
        set.add(0L);
        set.add(3L);
        condition = new LongColumnCondition("column", SequenceConditionMode.Or, ConditionOp.NotInSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable)new LongWritable(5))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new LongWritable(10))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new LongWritable(0))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new LongWritable(3))));
    }

    @Test
    public void testDoubleCondition(){
        Schema schema = TestTransforms.getSchema(ColumnType.Double);

        Condition condition = new DoubleColumnCondition("column", SequenceConditionMode.Or, ConditionOp.GreaterOrEqual, 0);
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new DoubleWritable(0.0))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new DoubleWritable(0.5))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new DoubleWritable(-0.5))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new DoubleWritable(-1))));

        Set<Double> set = new HashSet<>();
        set.add(0.0);
        set.add(3.0);
        condition = new DoubleColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable)new DoubleWritable(0.0))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new DoubleWritable(3.0))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new DoubleWritable(1.0))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new DoubleWritable(2.0))));
    }

    @Test
    public void testStringCondition(){
        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition = new StringColumnCondition("column", SequenceConditionMode.Or, ConditionOp.Equal, "value");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("value"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text("not_value"))));

        Set<String> set = new HashSet<>();
        set.add("in set");
        set.add("also in set");
        condition = new StringColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("in set"))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("also in set"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text("not in the set"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text(":)"))));
    }

    @Test
    public void testCategoricalCondition(){
        Schema schema = new Schema.Builder()
                .addColumnCategorical("column", "alpha", "beta", "gamma")
                .build();

        Condition condition = new CategoricalColumnCondition("column", SequenceConditionMode.Or, ConditionOp.Equal, "alpha");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("alpha"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text("beta"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text("gamma"))));

        Set<String> set = new HashSet<>();
        set.add("alpha");
        set.add("beta");
        condition = new StringColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("alpha"))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("beta"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text("gamma"))));
    }

    @Test
    public void testTimeCondition(){
        Schema schema = TestTransforms.getSchema(ColumnType.Time);

        //1451606400000 = 01/01/2016 00:00:00 GMT
        Condition condition = new TimeColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessOrEqual, 1451606400000L);
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new LongWritable(1451606400000L))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new LongWritable(1451606400000L - 1L))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new LongWritable(1451606400000L + 1L))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new LongWritable(1451606400000L + 1000L))));

        Set<Long> set = new HashSet<>();
        set.add(1451606400000L);
        condition = new TimeColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable)new LongWritable(1451606400000L))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new LongWritable(1451606400000L + 1L))));
    }

    @Test
    public void testStringRegexCondition(){

        Schema schema = TestTransforms.getSchema(ColumnType.String);

        //Condition: String value starts with "abc"
        Condition condition = new StringRegexColumnCondition("column", "abc.*");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("abc"))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("abcdefghijk"))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new Text("abc more text \tetc"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text("ab"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text("also doesn't match"))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text(" abc"))));

        //Check application on non-String columns
        schema = TestTransforms.getSchema(ColumnType.Integer);
        condition = new StringRegexColumnCondition("column", "123\\d*");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new IntWritable(123))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new IntWritable(123456))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(-123))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(456789))));
    }

    @Test
    public void testNullWritableColumnCondition(){
        Schema schema = TestTransforms.getSchema(ColumnType.Time);

        Condition condition = new NullWritableColumnCondition("column");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) NullWritable.INSTANCE)));
        assertTrue(condition.condition(Collections.singletonList((Writable) new NullWritable() )));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(0))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new Text("1"))));
    }

    @Test
    public void testBooleanConditionNot(){

        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessThan, 0);
        condition.setInputSchema(schema);

        Condition notCondition = BooleanCondition.NOT(condition);
        notCondition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable)new IntWritable(-1))));
        assertTrue(condition.condition(Collections.singletonList((Writable)new IntWritable(-2))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(0))));
        assertFalse(condition.condition(Collections.singletonList((Writable)new IntWritable(1))));

        //Expect opposite for not condition:
        assertFalse(notCondition.condition(Collections.singletonList((Writable)new IntWritable(-1))));
        assertFalse(notCondition.condition(Collections.singletonList((Writable)new IntWritable(-2))));
        assertTrue(notCondition.condition(Collections.singletonList((Writable)new IntWritable(0))));
        assertTrue(notCondition.condition(Collections.singletonList((Writable)new IntWritable(1))));
    }

    @Test
    public void testBooleanConditionAnd(){

        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition1 = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessThan, 0);
        condition1.setInputSchema(schema);

        Condition condition2 = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessThan, -1);
        condition2.setInputSchema(schema);

        Condition andCondition = BooleanCondition.AND(condition1,condition2);
        andCondition.setInputSchema(schema);

        assertFalse(andCondition.condition(Collections.singletonList((Writable)new IntWritable(-1))));
        assertTrue(andCondition.condition(Collections.singletonList((Writable)new IntWritable(-2))));
        assertFalse(andCondition.condition(Collections.singletonList((Writable)new IntWritable(0))));
        assertFalse(andCondition.condition(Collections.singletonList((Writable)new IntWritable(1))));
    }
}
