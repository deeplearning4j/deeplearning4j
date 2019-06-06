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

package org.datavec.api.transform.reduce;

import lombok.Getter;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.ops.AggregableMultiOp;
import org.datavec.api.transform.ops.IAggregableReduceOp;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 21/03/2016.
 */
public class TestMultiOpReduce {

    @Test
    public void testMultiOpReducerDouble() {

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new DoubleWritable(0)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new DoubleWritable(1)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new DoubleWritable(2)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new DoubleWritable(2)));

        Map<ReduceOp, Double> exp = new LinkedHashMap<>();
        exp.put(ReduceOp.Min, 0.0);
        exp.put(ReduceOp.Max, 2.0);
        exp.put(ReduceOp.Range, 2.0);
        exp.put(ReduceOp.Sum, 5.0);
        exp.put(ReduceOp.Mean, 1.25);
        exp.put(ReduceOp.Stdev, 0.957427108);
        exp.put(ReduceOp.Count, 4.0);
        exp.put(ReduceOp.CountUnique, 3.0);
        exp.put(ReduceOp.TakeFirst, 0.0);
        exp.put(ReduceOp.TakeLast, 2.0);

        for (ReduceOp op : exp.keySet()) {

            Schema schema = new Schema.Builder().addColumnString("key").addColumnDouble("column").build();

            Reducer reducer = new Reducer.Builder(op).keyColumns("key").build();

            reducer.setInputSchema(schema);
            IAggregableReduceOp<List<Writable>, List<Writable>> accumulator = reducer.aggregableReducer();

            for (int i = 0; i < inputs.size(); i++) {
                accumulator.accept(inputs.get(i));
            }
            List<Writable> out = accumulator.get();

            assertEquals(2, out.size());

            assertEquals(out.get(0), new Text("someKey"));

            String msg = op.toString();
            assertEquals(msg, exp.get(op), out.get(1).toDouble(), 1e-5);
        }
    }

    @Test
    public void testReducerInteger() {

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(0)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(1)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(2)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(2)));

        Map<ReduceOp, Double> exp = new LinkedHashMap<>();
        exp.put(ReduceOp.Min, 0.0);
        exp.put(ReduceOp.Max, 2.0);
        exp.put(ReduceOp.Range, 2.0);
        exp.put(ReduceOp.Sum, 5.0);
        exp.put(ReduceOp.Mean, 1.25);
        exp.put(ReduceOp.Stdev, 0.957427108);
        exp.put(ReduceOp.Count, 4.0);
        exp.put(ReduceOp.CountUnique, 3.0);
        exp.put(ReduceOp.TakeFirst, 0.0);
        exp.put(ReduceOp.TakeLast, 2.0);

        for (ReduceOp op : exp.keySet()) {

            Schema schema = new Schema.Builder().addColumnString("key").addColumnInteger("column").build();

            Reducer reducer = new Reducer.Builder(op).keyColumns("key").build();

            reducer.setInputSchema(schema);
            IAggregableReduceOp<List<Writable>, List<Writable>> accumulator = reducer.aggregableReducer();

            for (int i = 0; i < inputs.size(); i++) {
                accumulator.accept(inputs.get(i));
            }
            List<Writable> out = accumulator.get();

            assertEquals(2, out.size());

            assertEquals(out.get(0), new Text("someKey"));

            String msg = op.toString();
            assertEquals(msg, exp.get(op), out.get(1).toDouble(), 1e-5);
        }
    }


    @Test
    public void testReduceString() {

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("1")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("2")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("3")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("4")));

        Map<ReduceOp, String> exp = new LinkedHashMap<>();
        exp.put(ReduceOp.Append, "1234");
        exp.put(ReduceOp.Prepend, "4321");

        for (ReduceOp op : exp.keySet()) {

            Schema schema = new Schema.Builder().addColumnString("key").addColumnsString("column").build();

            Reducer reducer = new Reducer.Builder(op).keyColumns("key").build();

            reducer.setInputSchema(schema);
            IAggregableReduceOp<List<Writable>, List<Writable>> accumulator = reducer.aggregableReducer();

            for (int i = 0; i < inputs.size(); i++) {
                accumulator.accept(inputs.get(i));
            }
            List<Writable> out = accumulator.get();

            assertEquals(2, out.size());

            assertEquals(out.get(0), new Text("someKey"));

            String msg = op.toString();
            assertEquals(msg, exp.get(op), out.get(1).toString());
        }
    }

    @Test
    public void testReduceIntegerIgnoreInvalidValues() {

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("0")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("1")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(2)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("ignore me")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("also ignore me")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("2")));


        Map<ReduceOp, Double> exp = new LinkedHashMap<>();
        exp.put(ReduceOp.Min, 0.0);
        exp.put(ReduceOp.Max, 2.0);
        exp.put(ReduceOp.Range, 2.0);
        exp.put(ReduceOp.Sum, 5.0);
        exp.put(ReduceOp.Mean, 1.25);
        exp.put(ReduceOp.Stdev, 0.957427108);
        exp.put(ReduceOp.Count, 4.0);
        exp.put(ReduceOp.CountUnique, 3.0);
        exp.put(ReduceOp.TakeFirst, 0.0);
        exp.put(ReduceOp.TakeLast, 2.0);

        for (ReduceOp op : exp.keySet()) {
            Schema schema = new Schema.Builder().addColumnString("key").addColumnInteger("column").build();

            Reducer reducer = new Reducer.Builder(op).keyColumns("key").setIgnoreInvalid("column").build();

            reducer.setInputSchema(schema);

            IAggregableReduceOp<List<Writable>, List<Writable>> accumulator = reducer.aggregableReducer();

            for (int i = 0; i < inputs.size(); i++) {
                accumulator.accept(inputs.get(i));
            }
            List<Writable> out = accumulator.get();

            assertEquals(2, out.size());

            assertEquals(out.get(0), new Text("someKey"));

            String msg = op.toString();
            assertEquals(msg, exp.get(op), out.get(1).toDouble(), 1e-5);
        }

        for (ReduceOp op : Arrays.asList(ReduceOp.Min, ReduceOp.Max, ReduceOp.Range, ReduceOp.Sum, ReduceOp.Mean,
                        ReduceOp.Stdev)) {
            //Try the same thing WITHOUT setIgnoreInvalid -> expect exception

            Schema schema = new Schema.Builder().addColumnString("key").addColumnInteger("column").build();

            Reducer reducer = new Reducer.Builder(op).keyColumns("key").build();
            reducer.setInputSchema(schema);
            IAggregableReduceOp<List<Writable>, List<Writable>> accu = reducer.aggregableReducer();

            try {
                for (List<Writable> i : inputs)
                    accu.accept(i);
                fail("No exception thrown for invalid input: op=" + op);
            } catch (NumberFormatException e) {
                //ok
            }
        }
    }


    @Test
    public void testCustomReductions() {

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(1), new Text("zero"),
                        new DoubleWritable(0)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(2), new Text("one"),
                        new DoubleWritable(1)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(3), new Text("two"),
                        new DoubleWritable(2)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(4), new Text("three"),
                        new DoubleWritable(3)));

        List<Writable> expected = Arrays.asList((Writable) new Text("someKey"), new IntWritable(10), new Text("one"),
                        new DoubleWritable(1));


        Schema schema = new Schema.Builder().addColumnString("key").addColumnInteger("intCol")
                        .addColumnString("textCol").addColumnString("doubleCol").build();

        Reducer reducer = new Reducer.Builder(ReduceOp.Sum).keyColumns("key")
                        .customReduction("textCol", new CustomReduceTakeSecond())
                        .customReduction("doubleCol", new CustomReduceTakeSecond()).build();

        reducer.setInputSchema(schema);


        IAggregableReduceOp<List<Writable>, List<Writable>> accumulator = reducer.aggregableReducer();

        for (int i = 0; i < inputs.size(); i++) {
            accumulator.accept(inputs.get(i));
        }
        List<Writable> out = accumulator.get();

        assertEquals(4, out.size());
        assertEquals(expected, out);

        //Check schema:
        String[] expNames = new String[] {"key", "sum(intCol)", "myCustomReduce(textCol)", "myCustomReduce(doubleCol)"};
        ColumnType[] expTypes =
                        new ColumnType[] {ColumnType.String, ColumnType.Integer, ColumnType.String, ColumnType.String};
        Schema outSchema = reducer.transform(schema);

        assertEquals(4, outSchema.numColumns());
        for (int i = 0; i < 4; i++) {
            assertEquals(expNames[i], outSchema.getName(i));
            assertEquals(expTypes[i], outSchema.getType(i));
        }
    }

    @Test
    public void testCustomReductionsWithCondition() {

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(1), new Text("zero"),
                new DoubleWritable(0)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(2), new Text("one"),
                new DoubleWritable(1)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(3), new Text("two"),
                new DoubleWritable(2)));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new IntWritable(4), new Text("three"),
                new DoubleWritable(3)));

        List<Writable> expected = Arrays.asList((Writable) new Text("someKey"), new IntWritable(10), new IntWritable(3),
                new DoubleWritable(1));


        Schema schema = new Schema.Builder().addColumnString("key").addColumnInteger("intCol")
                .addColumnString("textCol").addColumnString("doubleCol").build();

        Reducer reducer = new Reducer.Builder(ReduceOp.Sum).keyColumns("key")
                .conditionalReduction("textCol", "condTextCol",
                        ReduceOp.Count, new StringColumnCondition("textCol", ConditionOp.NotEqual, "three"))
                .customReduction("doubleCol", new CustomReduceTakeSecond()).build();

        reducer.setInputSchema(schema);


        IAggregableReduceOp<List<Writable>, List<Writable>> accumulator = reducer.aggregableReducer();

        for (int i = 0; i < inputs.size(); i++) {
            accumulator.accept(inputs.get(i));
        }
        List<Writable> out = accumulator.get();

        assertEquals(4, out.size());
        assertEquals(expected, out);

        //Check schema:
        String[] expNames = new String[] {"key", "sum(intCol)", "condTextCol", "myCustomReduce(doubleCol)"};
        ColumnType[] expTypes =
                new ColumnType[] {ColumnType.String, ColumnType.Integer, ColumnType.Long, ColumnType.String};
        Schema outSchema = reducer.transform(schema);

        assertEquals(4, outSchema.numColumns());
        for (int i = 0; i < 4; i++) {
            assertEquals(expNames[i], outSchema.getName(i));
            assertEquals(expTypes[i], outSchema.getType(i));
        }
    }

    private static class CustomReduceTakeSecond implements AggregableColumnReduction {

        @Override
        public IAggregableReduceOp<Writable, List<Writable>> reduceOp() {
            //For testing: let's take the second value
            return new AggregableMultiOp<>(Collections
                            .<IAggregableReduceOp<Writable, Writable>>singletonList(new AggregableSecond<Writable>()));
        }

        @Override
        public List<String> getColumnsOutputName(String columnInputName) {
            return Collections.singletonList("myCustomReduce(" + columnInputName + ")");
        }

        @Override
        public List<ColumnMetaData> getColumnOutputMetaData(List<String> newColumnName,
                        ColumnMetaData columnInputMeta) {
            ColumnMetaData thiscolumnMeta = new StringMetaData(newColumnName.get(0));
            return Collections.singletonList(thiscolumnMeta);
        }

        public static class AggregableSecond<T> implements IAggregableReduceOp<T, Writable> {

            @Getter
            private T firstMet = null;
            @Getter
            private T elem = null;

            @Override
            public void accept(T element) {
                if (firstMet == null)
                    firstMet = element;
                else {
                    if (elem == null)
                        elem = element;
                }
            }

            @Override
            public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
                if (accu instanceof AggregableSecond && elem == null) {
                    if (firstMet == null) { // this accumulator is empty, import accu
                        AggregableSecond<T> accumulator = (AggregableSecond) accu;
                        T otherFirst = accumulator.getFirstMet();
                        T otherElement = accumulator.getElem();
                        if (otherFirst != null)
                            firstMet = otherFirst;
                        if (otherElement != null)
                            elem = otherElement;
                    } else { // we have the first element, they may have the rest
                        AggregableSecond<T> accumulator = (AggregableSecond) accu;
                        T otherFirst = accumulator.getFirstMet();
                        if (otherFirst != null)
                            elem = otherFirst;
                    }
                }
            }

            @Override
            public Writable get() {
                return UnsafeWritableInjector.inject(elem);
            }
        }

        /**
         * Get the output schema for this transformation, given an input schema
         *
         * @param inputSchema
         */
        @Override
        public Schema transform(Schema inputSchema) {
            return null;
        }

        /**
         * Set the input schema.
         *
         * @param inputSchema
         */
        @Override
        public void setInputSchema(Schema inputSchema) {

        }

        /**
         * Getter for input schema
         *
         * @return
         */
        @Override
        public Schema getInputSchema() {
            return null;
        }

        /**
         * The output column name
         * after the operation has been applied
         *
         * @return the output column name
         */
        @Override
        public String outputColumnName() {
            return null;
        }

        /**
         * The output column names
         * This will often be the same as the input
         *
         * @return the output column names
         */
        @Override
        public String[] outputColumnNames() {
            return new String[0];
        }

        /**
         * Returns column names
         * this op is meant to run on
         *
         * @return
         */
        @Override
        public String[] columnNames() {
            return new String[0];
        }

        /**
         * Returns a singular column name
         * this op is meant to run on
         *
         * @return
         */
        @Override
        public String columnName() {
            return null;
        }
    }



    @Test
    public void testConditionalReduction() {

        Schema schema = new Schema.Builder().addColumnString("key").addColumnInteger("intCol")
                        .addColumnString("filterCol").addColumnString("textCol").build();

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.<Writable>asList(new Text("someKey"), new IntWritable(1), new Text("a"), new Text("zero")));
        inputs.add(Arrays.<Writable>asList(new Text("someKey"), new IntWritable(2), new Text("b"), new Text("one")));
        inputs.add(Arrays.<Writable>asList(new Text("someKey"), new IntWritable(3), new Text("a"), new Text("two")));
        inputs.add(Arrays.<Writable>asList(new Text("someKey"), new IntWritable(4), new Text("b"), new Text("three")));
        inputs.add(Arrays.<Writable>asList(new Text("someKey"), new IntWritable(5), new Text("a"), new Text("three")));
        inputs.add(Arrays.<Writable>asList(new Text("someKey"), new IntWritable(6), new Text("b"), new Text("three")));

        Condition condition = new StringColumnCondition("filterCol", ConditionOp.Equal, "a");

        Reducer reducer = new Reducer.Builder(ReduceOp.Stdev).keyColumns("key")
                        .conditionalReduction("intCol", "sumOfAs", ReduceOp.Sum, condition) //Sum, only where 'filterCol' == "a"
                        .countUniqueColumns("filterCol", "textCol").build();

        reducer.setInputSchema(schema);

        IAggregableReduceOp<List<Writable>, List<Writable>> accumulator = reducer.aggregableReducer();

        for (int i = 0; i < inputs.size(); i++) {
            accumulator.accept(inputs.get(i));
        }
        List<Writable> out = accumulator.get();
        List<Writable> expected = Arrays.<Writable>asList(new Text("someKey"), new IntWritable(1 + 3 + 5),
                        new LongWritable(2), new LongWritable(4));

        assertEquals(4, out.size());
        assertEquals(expected, out);

        Schema outSchema = reducer.transform(schema);
        assertEquals(4, outSchema.numColumns());
        assertEquals(Arrays.asList("key", "sumOfAs", "countunique(filterCol)", "countunique(textCol)"),
                        outSchema.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.String, ColumnType.Integer, ColumnType.Long, ColumnType.Long),
                        outSchema.getColumnTypes());
    }
}
