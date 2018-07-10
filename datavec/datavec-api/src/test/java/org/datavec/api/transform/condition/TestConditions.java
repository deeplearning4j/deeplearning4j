/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.transform.condition;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.condition.column.*;
import org.datavec.api.transform.condition.sequence.SequenceLengthCondition;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.TestTransforms;
import org.datavec.api.writable.*;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 24/03/2016.
 */
public class TestConditions {

    @Test
    public void testIntegerCondition() {
        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessThan, 0);
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new IntWritable(-1))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new IntWritable(-2))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(0))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(1))));

        Set<Integer> set = new HashSet<>();
        set.add(0);
        set.add(3);
        condition = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable) new IntWritable(0))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new IntWritable(3))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(1))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(2))));
    }

    @Test
    public void testLongCondition() {
        Schema schema = TestTransforms.getSchema(ColumnType.Long);

        Condition condition = new LongColumnCondition("column", SequenceConditionMode.Or, ConditionOp.NotEqual, 5L);
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new LongWritable(0))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new LongWritable(1))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new LongWritable(5))));

        Set<Long> set = new HashSet<>();
        set.add(0L);
        set.add(3L);
        condition = new LongColumnCondition("column", SequenceConditionMode.Or, ConditionOp.NotInSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable) new LongWritable(5))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new LongWritable(10))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new LongWritable(0))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new LongWritable(3))));
    }

    @Test
    public void testDoubleCondition() {
        Schema schema = TestTransforms.getSchema(ColumnType.Double);

        Condition condition =
                        new DoubleColumnCondition("column", SequenceConditionMode.Or, ConditionOp.GreaterOrEqual, 0);
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new DoubleWritable(0.0))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new DoubleWritable(0.5))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new DoubleWritable(-0.5))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new DoubleWritable(-1))));

        Set<Double> set = new HashSet<>();
        set.add(0.0);
        set.add(3.0);
        condition = new DoubleColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable) new DoubleWritable(0.0))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new DoubleWritable(3.0))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new DoubleWritable(1.0))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new DoubleWritable(2.0))));
    }

    @Test
    public void testStringCondition() {
        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition = new StringColumnCondition("column", SequenceConditionMode.Or, ConditionOp.Equal, "value");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("value"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("not_value"))));

        Set<String> set = new HashSet<>();
        set.add("in set");
        set.add("also in set");
        condition = new StringColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("in set"))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("also in set"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("not in the set"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text(":)"))));
    }

    @Test
    public void testCategoricalCondition() {
        Schema schema = new Schema.Builder().addColumnCategorical("column", "alpha", "beta", "gamma").build();

        Condition condition =
                        new CategoricalColumnCondition("column", SequenceConditionMode.Or, ConditionOp.Equal, "alpha");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("alpha"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("beta"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("gamma"))));

        Set<String> set = new HashSet<>();
        set.add("alpha");
        set.add("beta");
        condition = new StringColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("alpha"))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("beta"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("gamma"))));
    }

    @Test
    public void testTimeCondition() {
        Schema schema = TestTransforms.getSchema(ColumnType.Time);

        //1451606400000 = 01/01/2016 00:00:00 GMT
        Condition condition = new TimeColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessOrEqual,
                        1451606400000L);
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new LongWritable(1451606400000L))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new LongWritable(1451606400000L - 1L))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new LongWritable(1451606400000L + 1L))));
        assertFalse(condition
                        .condition(Collections.singletonList((Writable) new LongWritable(1451606400000L + 1000L))));

        Set<Long> set = new HashSet<>();
        set.add(1451606400000L);
        condition = new TimeColumnCondition("column", SequenceConditionMode.Or, ConditionOp.InSet, set);
        condition.setInputSchema(schema);
        assertTrue(condition.condition(Collections.singletonList((Writable) new LongWritable(1451606400000L))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new LongWritable(1451606400000L + 1L))));
    }

    @Test
    public void testStringRegexCondition() {

        Schema schema = TestTransforms.getSchema(ColumnType.String);

        //Condition: String value starts with "abc"
        Condition condition = new StringRegexColumnCondition("column", "abc.*");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("abc"))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("abcdefghijk"))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("abc more text \tetc"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("ab"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("also doesn't match"))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text(" abc"))));

        //Check application on non-String columns
        schema = TestTransforms.getSchema(ColumnType.Integer);
        condition = new StringRegexColumnCondition("column", "123\\d*");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new IntWritable(123))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new IntWritable(123456))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(-123))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(456789))));
    }

    @Test
    public void testNullWritableColumnCondition() {
        Schema schema = TestTransforms.getSchema(ColumnType.Time);

        Condition condition = new NullWritableColumnCondition("column");
        condition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) NullWritable.INSTANCE)));
        assertTrue(condition.condition(Collections.singletonList((Writable) new NullWritable())));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(0))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("1"))));
    }

    @Test
    public void testBooleanConditionNot() {

        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessThan, 0);
        condition.setInputSchema(schema);

        Condition notCondition = BooleanCondition.NOT(condition);
        notCondition.setInputSchema(schema);

        assertTrue(condition.condition(Collections.singletonList((Writable) new IntWritable(-1))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new IntWritable(-2))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(0))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(1))));

        //Expect opposite for not condition:
        assertFalse(notCondition.condition(Collections.singletonList((Writable) new IntWritable(-1))));
        assertFalse(notCondition.condition(Collections.singletonList((Writable) new IntWritable(-2))));
        assertTrue(notCondition.condition(Collections.singletonList((Writable) new IntWritable(0))));
        assertTrue(notCondition.condition(Collections.singletonList((Writable) new IntWritable(1))));
    }

    @Test
    public void testBooleanConditionAnd() {

        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition1 = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessThan, 0);
        condition1.setInputSchema(schema);

        Condition condition2 = new IntegerColumnCondition("column", SequenceConditionMode.Or, ConditionOp.LessThan, -1);
        condition2.setInputSchema(schema);

        Condition andCondition = BooleanCondition.AND(condition1, condition2);
        andCondition.setInputSchema(schema);

        assertFalse(andCondition.condition(Collections.singletonList((Writable) new IntWritable(-1))));
        assertTrue(andCondition.condition(Collections.singletonList((Writable) new IntWritable(-2))));
        assertFalse(andCondition.condition(Collections.singletonList((Writable) new IntWritable(0))));
        assertFalse(andCondition.condition(Collections.singletonList((Writable) new IntWritable(1))));
    }


    @Test
    public void testInvalidValueColumnConditionCondition() {
        Schema schema = TestTransforms.getSchema(ColumnType.Integer);

        Condition condition = new InvalidValueColumnCondition("column");
        condition.setInputSchema(schema);

        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(-1)))); //Not invalid -> condition does not apply
        assertFalse(condition.condition(Collections.singletonList((Writable) new IntWritable(-2))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new LongWritable(1000))));
        assertFalse(condition.condition(Collections.singletonList((Writable) new Text("1000"))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("text"))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new Text("NaN"))));
        assertTrue(condition.condition(
                        Collections.singletonList((Writable) new LongWritable(1L + (long) Integer.MAX_VALUE))));
        assertTrue(condition.condition(Collections.singletonList((Writable) new DoubleWritable(3.14159))));
    }

    @Test
    public void testSequenceLengthCondition() {

        Condition c = new SequenceLengthCondition(ConditionOp.LessThan, 2);

        List<List<Writable>> l1 = Arrays.asList(Collections.<Writable>singletonList(NullWritable.INSTANCE));

        List<List<Writable>> l2 = Arrays.asList(Collections.<Writable>singletonList(NullWritable.INSTANCE),
                        Collections.<Writable>singletonList(NullWritable.INSTANCE));

        List<List<Writable>> l3 = Arrays.asList(Collections.<Writable>singletonList(NullWritable.INSTANCE),
                        Collections.<Writable>singletonList(NullWritable.INSTANCE),
                        Collections.<Writable>singletonList(NullWritable.INSTANCE));

        assertTrue(c.conditionSequence(l1));
        assertFalse(c.conditionSequence(l2));
        assertFalse(c.conditionSequence(l3));

        Set<Integer> set = new HashSet<>();
        set.add(2);
        c = new SequenceLengthCondition(ConditionOp.InSet, set);
        assertFalse(c.conditionSequence(l1));
        assertTrue(c.conditionSequence(l2));
        assertFalse(c.conditionSequence(l3));

    }
}
