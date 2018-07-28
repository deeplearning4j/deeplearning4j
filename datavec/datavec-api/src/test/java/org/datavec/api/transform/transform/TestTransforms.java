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

package org.datavec.api.transform.transform;

import junit.framework.TestCase;
import org.datavec.api.transform.*;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.IntegerColumnCondition;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.metadata.LongMetaData;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.sequence.ReduceSequenceTransform;
import org.datavec.api.transform.sequence.trim.SequenceTrimTransform;
import org.datavec.api.transform.serde.JsonMappers;
import org.datavec.api.transform.transform.categorical.*;
import org.datavec.api.transform.transform.column.*;
import org.datavec.api.transform.transform.condition.ConditionalCopyValueTransform;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransform;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransformWithDefault;
import org.datavec.api.transform.transform.doubletransform.*;
import org.datavec.api.transform.transform.integer.*;
import org.datavec.api.transform.transform.longtransform.LongColumnsMathOpTransform;
import org.datavec.api.transform.transform.longtransform.LongMathOpTransform;
import org.datavec.api.transform.transform.nlp.TextToCharacterIndexTransform;
import org.datavec.api.transform.transform.nlp.TextToTermIndexSequenceTransform;
import org.datavec.api.transform.transform.sequence.SequenceDifferenceTransform;
import org.datavec.api.transform.transform.sequence.SequenceMovingWindowReduceTransform;
import org.datavec.api.transform.transform.sequence.SequenceOffsetTransform;
import org.datavec.api.transform.transform.string.*;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.transform.transform.time.TimeMathOpTransform;
import org.datavec.api.writable.*;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.*;

/**
 * Created by Alex on 21/03/2016.
 */
public class TestTransforms {

    public static Schema getSchema(ColumnType type, String... colNames) {

        Schema.Builder schema = new Schema.Builder();

        switch (type) {
            case String:
                schema.addColumnString("column");
                break;
            case Integer:
                schema.addColumnInteger("column");
                break;
            case Long:
                schema.addColumnLong("column");
                break;
            case Double:
                schema.addColumnDouble("column");
                break;
            case Float:
                schema.addColumnFloat("column");
            case Categorical:
                schema.addColumnCategorical("column", colNames);
                break;
            case Time:
                schema.addColumnTime("column", DateTimeZone.UTC);
                break;
            default:
                throw new RuntimeException();
        }
        return schema.build();
    }

    @Test
    public void testCategoricalToInteger() {
        Schema schema = getSchema(ColumnType.Categorical, "zero", "one", "two");

        Transform transform = new CategoricalToIntegerTransform("column");
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);


        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(0).getColumnType());
        IntegerMetaData meta = (IntegerMetaData) out.getMetaData(0);
        assertNotNull(meta.getMinAllowedValue());
        assertEquals(0, (int) meta.getMinAllowedValue());

        assertNotNull(meta.getMaxAllowedValue());
        assertEquals(2, (int) meta.getMaxAllowedValue());

        assertEquals(0, transform.map(Collections.singletonList((Writable) new Text("zero"))).get(0).toInt());
        assertEquals(1, transform.map(Collections.singletonList((Writable) new Text("one"))).get(0).toInt());
        assertEquals(2, transform.map(Collections.singletonList((Writable) new Text("two"))).get(0).toInt());
    }

    @Test
    public void testCategoricalToOneHotTransform() {
        Schema schema = getSchema(ColumnType.Categorical, "zero", "one", "two");

        Transform transform = new CategoricalToOneHotTransform("column");
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(3, out.getColumnMetaData().size());
        for (int i = 0; i < 3; i++) {
            TestCase.assertEquals(ColumnType.Integer, out.getMetaData(i).getColumnType());
            IntegerMetaData meta = (IntegerMetaData) out.getMetaData(i);
            assertNotNull(meta.getMinAllowedValue());
            assertEquals(0, (int) meta.getMinAllowedValue());

            assertNotNull(meta.getMaxAllowedValue());
            assertEquals(1, (int) meta.getMaxAllowedValue());
        }

        assertEquals(Arrays.asList(new IntWritable(1), new IntWritable(0), new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new Text("zero"))));
        assertEquals(Arrays.asList(new IntWritable(0), new IntWritable(1), new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new Text("one"))));
        assertEquals(Arrays.asList(new IntWritable(0), new IntWritable(0), new IntWritable(1)),
                transform.map(Collections.singletonList((Writable) new Text("two"))));
    }

    @Test
    public void testPivotTransform(){
        Schema schema = new Schema.Builder()
                .addColumnString("otherCol")
                .addColumnCategorical("key", Arrays.asList("first","second","third"))
                .addColumnDouble("value")
                .addColumnDouble("otherCol2")
                .build();

        Transform t = new PivotTransform("key","value");
        t.setInputSchema(schema);
        Schema out = t.transform(schema);

        List<String> expNames = Arrays.asList("otherCol", "key[first]", "key[second]", "key[third]", "otherCol2");
        List<String> actNames = out.getColumnNames();

        assertEquals(expNames, actNames);

        List<ColumnType> columnTypesExp = Arrays.asList(ColumnType.String, ColumnType.Double, ColumnType.Double,
                ColumnType.Double, ColumnType.Double);
        assertEquals(columnTypesExp, out.getColumnTypes());

        //Expand (second,100) into (0,100,0). Leave the remaining columns as is
        List<Writable> e1 = Arrays.<Writable>asList(new DoubleWritable(1), new DoubleWritable(0), new DoubleWritable(100),
                new DoubleWritable(0), new DoubleWritable(-1));
        List<Writable> a1 = t.map(Arrays.<Writable>asList(new DoubleWritable(1), new Text("second"), new DoubleWritable(100),
                new DoubleWritable(-1)));
        assertEquals(e1,a1);

        //Expand (third,200) into (0,0,200). Leave the remaining columns as is
        List<Writable> e2 = Arrays.<Writable>asList(new DoubleWritable(1), new DoubleWritable(0), new DoubleWritable(0),
                new DoubleWritable(200), new DoubleWritable(-1));
        List<Writable> a2 = t.map(Arrays.<Writable>asList(new DoubleWritable(1), new Text("third"), new DoubleWritable(200),
                new DoubleWritable(-1)));
        assertEquals(e2,a2);
    }

    @Test
    public void testIntegerToCategoricalTransform() {
        Schema schema = getSchema(ColumnType.Integer);

        Transform transform = new IntegerToCategoricalTransform("column", Arrays.asList("zero", "one", "two"));
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        assertEquals(ColumnType.Categorical, out.getMetaData(0).getColumnType());
        CategoricalMetaData meta = (CategoricalMetaData) out.getMetaData(0);
        assertEquals(Arrays.asList("zero", "one", "two"), meta.getStateNames());

        assertEquals(Collections.singletonList((Writable) new Text("zero")),
                transform.map(Collections.singletonList((Writable) new IntWritable(0))));
        assertEquals(Collections.singletonList((Writable) new Text("one")),
                transform.map(Collections.singletonList((Writable) new IntWritable(1))));
        assertEquals(Collections.singletonList((Writable) new Text("two")),
                transform.map(Collections.singletonList((Writable) new IntWritable(2))));
    }

    @Test
    public void testIntegerToOneHotTransform() {
        Schema schema = getSchema(ColumnType.Integer);

        Transform transform = new IntegerToOneHotTransform("column", 3, 5);
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(3, out.getColumnMetaData().size());
        assertEquals(ColumnType.Integer, out.getMetaData(0).getColumnType());
        assertEquals(ColumnType.Integer, out.getMetaData(1).getColumnType());
        assertEquals(ColumnType.Integer, out.getMetaData(2).getColumnType());

        assertEquals(Arrays.asList("column[3]", "column[4]", "column[5]"), out.getColumnNames());

        assertEquals(Arrays.<Writable>asList(new IntWritable(1), new IntWritable(0), new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(3))));
        assertEquals(Arrays.<Writable>asList(new IntWritable(0), new IntWritable(1), new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(4))));
        assertEquals(Arrays.<Writable>asList(new IntWritable(0), new IntWritable(0), new IntWritable(1)),
                transform.map(Collections.singletonList((Writable) new IntWritable(5))));
    }

    @Test
    public void testStringToCategoricalTransform() {
        Schema schema = getSchema(ColumnType.String);

        Transform transform = new StringToCategoricalTransform("column", Arrays.asList("zero", "one", "two"));
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Categorical, out.getMetaData(0).getColumnType());
        CategoricalMetaData meta = (CategoricalMetaData) out.getMetaData(0);
        assertEquals(Arrays.asList("zero", "one", "two"), meta.getStateNames());

        assertEquals(Collections.singletonList((Writable) new Text("zero")),
                transform.map(Collections.singletonList((Writable) new Text("zero"))));
        assertEquals(Collections.singletonList((Writable) new Text("one")),
                transform.map(Collections.singletonList((Writable) new Text("one"))));
        assertEquals(Collections.singletonList((Writable) new Text("two")),
                transform.map(Collections.singletonList((Writable) new Text("two"))));
    }

    @Test
    public void testConcatenateStringColumnsTransform() throws Exception {
        final String DELIMITER = " ";
        final String NEW_COLUMN = "NewColumn";
        final List<String> CONCAT_COLUMNS = Arrays.asList("ConcatenatedColumn1", "ConcatenatedColumn2", "ConcatenatedColumn3");
        final List<String> ALL_COLUMNS = Arrays.asList("ConcatenatedColumn1", "OtherColumn4", "ConcatenatedColumn2",
                "OtherColumn5", "ConcatenatedColumn3", "OtherColumn6");
        final List<Text> COLUMN_VALUES = Arrays.asList(new Text("string1"), new Text("other4"),
                new Text("string2"), new Text("other5"),
                new Text("string3"), new Text("other6"));
        final String NEW_COLUMN_VALUE = "string1 string2 string3";

        Transform transform = new ConcatenateStringColumns(NEW_COLUMN, DELIMITER, CONCAT_COLUMNS);
        String[] allColumns = ALL_COLUMNS.toArray(new String[ALL_COLUMNS.size()]);
        Schema schema = new Schema.Builder().addColumnsString(allColumns).build();

        List<String> outputColumns = new ArrayList<>(ALL_COLUMNS);
        outputColumns.add(NEW_COLUMN);
        Schema newSchema = transform.transform(schema);
        Assert.assertEquals(outputColumns, newSchema.getColumnNames());

        List<Writable> input = new ArrayList<>();
        for (Writable value : COLUMN_VALUES)
            input.add(value);

        transform.setInputSchema(schema);
        List<Writable> transformed = transform.map(input);
        Assert.assertEquals(NEW_COLUMN_VALUE, transformed.get(transformed.size() - 1).toString());

        List<Text> outputColumnValues = new ArrayList<>(COLUMN_VALUES);
        outputColumnValues.add(new Text(NEW_COLUMN_VALUE));
        Assert.assertEquals(outputColumnValues, transformed);

        String s = JsonMappers.getMapper().writeValueAsString(transform);
        Transform transform2 = JsonMappers.getMapper().readValue(s, ConcatenateStringColumns.class);
        Assert.assertEquals(transform, transform2);
    }

    @Test
    public void testChangeCaseStringTransform() throws Exception {
        final String STRING_COLUMN = "StringColumn";
        final List<String> ALL_COLUMNS = Arrays.asList(STRING_COLUMN, "OtherColumn");
        final String TEXT_MIXED_CASE = "UPPER lower MiXeD";
        final String TEXT_UPPER_CASE = TEXT_MIXED_CASE.toUpperCase();
        final String TEXT_LOWER_CASE = TEXT_MIXED_CASE.toLowerCase();

        Transform transform = new ChangeCaseStringTransform(STRING_COLUMN);
        String[] allColumns = ALL_COLUMNS.toArray(new String[ALL_COLUMNS.size()]);
        Schema schema = new Schema.Builder().addColumnsString(allColumns).build();
        transform.setInputSchema(schema);
        Schema newSchema = transform.transform(schema);
        List<String> outputColumns = new ArrayList<>(ALL_COLUMNS);
        Assert.assertEquals(outputColumns, newSchema.getColumnNames());

        transform = new ChangeCaseStringTransform(STRING_COLUMN, ChangeCaseStringTransform.CaseType.LOWER);
        transform.setInputSchema(schema);
        List<Writable> input = new ArrayList<>();
        input.add(new Text(TEXT_MIXED_CASE));
        input.add(new Text(TEXT_MIXED_CASE));
        List<Writable> output = new ArrayList<>();
        output.add(new Text(TEXT_LOWER_CASE));
        output.add(new Text(TEXT_MIXED_CASE));
        List<Writable> transformed = transform.map(input);
        Assert.assertEquals(transformed.get(0).toString(), TEXT_LOWER_CASE);
        Assert.assertEquals(transformed, output);

        transform = new ChangeCaseStringTransform(STRING_COLUMN, ChangeCaseStringTransform.CaseType.UPPER);
        transform.setInputSchema(schema);
        output.clear();
        output.add(new Text(TEXT_UPPER_CASE));
        output.add(new Text(TEXT_MIXED_CASE));
        transformed = transform.map(input);
        Assert.assertEquals(transformed.get(0).toString(), TEXT_UPPER_CASE);
        Assert.assertEquals(transformed, output);

        String s = JsonMappers.getMapper().writeValueAsString(transform);
        Transform transform2 = JsonMappers.getMapper().readValue(s, ChangeCaseStringTransform.class);
        Assert.assertEquals(transform, transform2);
    }

    @Test
    public void testRemoveColumnsTransform() {
        Schema schema = new Schema.Builder().addColumnDouble("first").addColumnString("second")
                .addColumnInteger("third").addColumnLong("fourth").build();

        Transform transform = new RemoveColumnsTransform("first", "fourth");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(2, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());
        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(1).getColumnType());

        assertEquals(Arrays.asList(new Text("one"), new IntWritable(1)),
                transform.map(Arrays.asList((Writable) new DoubleWritable(1.0), new Text("one"),
                        new IntWritable(1), new LongWritable(1L))));
    }

    @Test
    public void testRemoveAllColumnsExceptForTransform() {
        Schema schema = new Schema.Builder().addColumnDouble("first").addColumnString("second")
                .addColumnInteger("third").addColumnLong("fourth").build();

        Transform transform = new RemoveAllColumnsExceptForTransform("second", "third");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(2, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());
        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(1).getColumnType());

        assertEquals(Arrays.asList(new Text("one"), new IntWritable(1)),
                transform.map(Arrays.asList((Writable) new DoubleWritable(1.0), new Text("one"),
                        new IntWritable(1), new LongWritable(1L))));

    }

    @Test
    public void testReplaceEmptyIntegerWithValueTransform() {
        Schema schema = getSchema(ColumnType.Integer);

        Transform transform = new ReplaceEmptyIntegerWithValueTransform("column", 1000);
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(0))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(1)),
                transform.map(Collections.singletonList((Writable) new IntWritable(1))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(1000)),
                transform.map(Collections.singletonList((Writable) new Text(""))));
    }

    @Test
    public void testReplaceInvalidWithIntegerTransform() {
        Schema schema = getSchema(ColumnType.Integer);

        Transform transform = new ReplaceInvalidWithIntegerTransform("column", 1000);
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(0))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(1)),
                transform.map(Collections.singletonList((Writable) new IntWritable(1))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(1000)),
                transform.map(Collections.singletonList((Writable) new Text(""))));
    }

    @Test
    public void testLog2Normalizer() {
        Schema schema = getSchema(ColumnType.Double);

        double mu = 2.0;
        double min = 1.0;
        double scale = 0.5;

        Transform transform = new Log2Normalizer("column", mu, min, scale);
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Double, out.getMetaData(0).getColumnType());
        DoubleMetaData meta = (DoubleMetaData) out.getMetaData(0);
        assertNotNull(meta.getMinAllowedValue());
        assertEquals(0, meta.getMinAllowedValue(), 1e-6);
        assertNull(meta.getMaxAllowedValue());

        double loge2 = Math.log(2);
        assertEquals(0.0,
                transform.map(Collections.singletonList((Writable) new DoubleWritable(min))).get(0).toDouble(),
                1e-6);
        double d = scale * Math.log((10 - min) / (mu - min) + 1) / loge2;
        assertEquals(d, transform.map(Collections.singletonList((Writable) new DoubleWritable(10))).get(0).toDouble(),
                1e-6);
        d = scale * Math.log((3 - min) / (mu - min) + 1) / loge2;
        assertEquals(d, transform.map(Collections.singletonList((Writable) new DoubleWritable(3))).get(0).toDouble(),
                1e-6);
    }

    @Test
    public void testDoubleMinMaxNormalizerTransform() {
        Schema schema = getSchema(ColumnType.Double);

        Transform transform = new MinMaxNormalizer("column", 0, 100);
        Transform transform2 = new MinMaxNormalizer("column", 0, 100, -1, 1);
        transform.setInputSchema(schema);
        transform2.setInputSchema(schema);

        Schema out = transform.transform(schema);
        Schema out2 = transform2.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Double, out.getMetaData(0).getColumnType());
        DoubleMetaData meta = (DoubleMetaData) out.getMetaData(0);
        DoubleMetaData meta2 = (DoubleMetaData) out2.getMetaData(0);
        assertEquals(0, meta.getMinAllowedValue(), 1e-6);
        assertEquals(1, meta.getMaxAllowedValue(), 1e-6);
        assertEquals(-1, meta2.getMinAllowedValue(), 1e-6);
        assertEquals(1, meta2.getMaxAllowedValue(), 1e-6);


        assertEquals(0.0, transform.map(Collections.singletonList((Writable) new DoubleWritable(0))).get(0).toDouble(),
                1e-6);
        assertEquals(1.0,
                transform.map(Collections.singletonList((Writable) new DoubleWritable(100))).get(0).toDouble(),
                1e-6);
        assertEquals(0.5, transform.map(Collections.singletonList((Writable) new DoubleWritable(50))).get(0).toDouble(),
                1e-6);

        assertEquals(-1.0,
                transform2.map(Collections.singletonList((Writable) new DoubleWritable(0))).get(0).toDouble(),
                1e-6);
        assertEquals(1.0,
                transform2.map(Collections.singletonList((Writable) new DoubleWritable(100))).get(0).toDouble(),
                1e-6);
        assertEquals(0.0,
                transform2.map(Collections.singletonList((Writable) new DoubleWritable(50))).get(0).toDouble(),
                1e-6);
    }

    @Test
    public void testStandardizeNormalizer() {
        Schema schema = getSchema(ColumnType.Double);

        double mu = 1.0;
        double sigma = 2.0;

        Transform transform = new StandardizeNormalizer("column", mu, sigma);
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Double, out.getMetaData(0).getColumnType());
        DoubleMetaData meta = (DoubleMetaData) out.getMetaData(0);
        assertNull(meta.getMinAllowedValue());
        assertNull(meta.getMaxAllowedValue());


        assertEquals(0.0, transform.map(Collections.singletonList((Writable) new DoubleWritable(mu))).get(0).toDouble(),
                1e-6);
        double d = (10 - mu) / sigma;
        assertEquals(d, transform.map(Collections.singletonList((Writable) new DoubleWritable(10))).get(0).toDouble(),
                1e-6);
        d = (-2 - mu) / sigma;
        assertEquals(d, transform.map(Collections.singletonList((Writable) new DoubleWritable(-2))).get(0).toDouble(),
                1e-6);
    }

    @Test
    public void testSubtractMeanNormalizer() {
        Schema schema = getSchema(ColumnType.Double);

        double mu = 1.0;

        Transform transform = new SubtractMeanNormalizer("column", mu);
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Double, out.getMetaData(0).getColumnType());
        DoubleMetaData meta = (DoubleMetaData) out.getMetaData(0);
        assertNull(meta.getMinAllowedValue());
        assertNull(meta.getMaxAllowedValue());


        assertEquals(0.0, transform.map(Collections.singletonList((Writable) new DoubleWritable(mu))).get(0).toDouble(),
                1e-6);
        assertEquals(10 - mu,
                transform.map(Collections.singletonList((Writable) new DoubleWritable(10))).get(0).toDouble(),
                1e-6);
    }

    @Test
    public void testMapAllStringsExceptListTransform() {
        Schema schema = getSchema(ColumnType.String);

        Transform transform = new MapAllStringsExceptListTransform("column", "replacement",
                Arrays.asList("one", "two", "three"));
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new Text("one")),
                transform.map(Collections.singletonList((Writable) new Text("one"))));
        assertEquals(Collections.singletonList((Writable) new Text("two")),
                transform.map(Collections.singletonList((Writable) new Text("two"))));
        assertEquals(Collections.singletonList((Writable) new Text("replacement")),
                transform.map(Collections.singletonList((Writable) new Text("this should be replaced"))));
    }

    @Test
    public void testRemoveWhitespaceTransform() {
        Schema schema = getSchema(ColumnType.String);

        Transform transform = new RemoveWhiteSpaceTransform("column");
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new Text("one")),
                transform.map(Collections.singletonList((Writable) new Text("one "))));
        assertEquals(Collections.singletonList((Writable) new Text("two")),
                transform.map(Collections.singletonList((Writable) new Text("two\t"))));
        assertEquals(Collections.singletonList((Writable) new Text("three")),
                transform.map(Collections.singletonList((Writable) new Text("three\n"))));
        assertEquals(Collections.singletonList((Writable) new Text("one")),
                transform.map(Collections.singletonList((Writable) new Text(" o n e\t"))));
    }

    @Test
    public void testReplaceEmptyStringTransform() {
        Schema schema = getSchema(ColumnType.String);

        Transform transform = new ReplaceEmptyStringTransform("column", "newvalue");
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new Text("one")),
                transform.map(Collections.singletonList((Writable) new Text("one"))));
        assertEquals(Collections.singletonList((Writable) new Text("newvalue")),
                transform.map(Collections.singletonList((Writable) new Text(""))));
        assertEquals(Collections.singletonList((Writable) new Text("three")),
                transform.map(Collections.singletonList((Writable) new Text("three"))));
    }

    @Test
    public void testAppendStringColumnTransform() {
        Schema schema = getSchema(ColumnType.String);

        Transform transform = new AppendStringColumnTransform("column", "_AppendThis");
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new Text("one_AppendThis")),
                transform.map(Collections.singletonList((Writable) new Text("one"))));
        assertEquals(Collections.singletonList((Writable) new Text("two_AppendThis")),
                transform.map(Collections.singletonList((Writable) new Text("two"))));
        assertEquals(Collections.singletonList((Writable) new Text("three_AppendThis")),
                transform.map(Collections.singletonList((Writable) new Text("three"))));
    }

    @Test
    public void testStringListToCategoricalSetTransform() {
        //Idea: String list to a set of categories... "a,c" for categories {a,b,c} -> "true","false","true"

        Schema schema = getSchema(ColumnType.String);

        Transform transform = new StringListToCategoricalSetTransform("column", Arrays.asList("a", "b", "c"),
                Arrays.asList("a", "b", "c"), ",");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(3, out.getColumnMetaData().size());
        for (int i = 0; i < 3; i++) {
            TestCase.assertEquals(ColumnType.Categorical, out.getType(i));
            CategoricalMetaData meta = (CategoricalMetaData) out.getMetaData(i);
            assertEquals(Arrays.asList("true", "false"), meta.getStateNames());
        }

        assertEquals(Arrays.asList(new Text("false"), new Text("false"), new Text("false")),
                transform.map(Collections.singletonList((Writable) new Text(""))));
        assertEquals(Arrays.asList(new Text("true"), new Text("false"), new Text("false")),
                transform.map(Collections.singletonList((Writable) new Text("a"))));
        assertEquals(Arrays.asList(new Text("false"), new Text("true"), new Text("false")),
                transform.map(Collections.singletonList((Writable) new Text("b"))));
        assertEquals(Arrays.asList(new Text("false"), new Text("false"), new Text("true")),
                transform.map(Collections.singletonList((Writable) new Text("c"))));
        assertEquals(Arrays.asList(new Text("true"), new Text("false"), new Text("true")),
                transform.map(Collections.singletonList((Writable) new Text("a,c"))));
        assertEquals(Arrays.asList(new Text("true"), new Text("true"), new Text("true")),
                transform.map(Collections.singletonList((Writable) new Text("a,b,c"))));
    }

    @Test
    public void testStringMapTransform() {
        Schema schema = getSchema(ColumnType.String);

        Map<String, String> map = new HashMap<>();
        map.put("one", "ONE");
        map.put("two", "TWO");
        Transform transform = new StringMapTransform("column", map);
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new Text("ONE")),
                transform.map(Collections.singletonList((Writable) new Text("one"))));
        assertEquals(Collections.singletonList((Writable) new Text("TWO")),
                transform.map(Collections.singletonList((Writable) new Text("two"))));
        assertEquals(Collections.singletonList((Writable) new Text("three")),
                transform.map(Collections.singletonList((Writable) new Text("three"))));
    }


    @Test
    public void testStringToTimeTransform() throws Exception {
        testStringToDateTime("YYYY-MM-dd HH:mm:ss");
    }



    @Test
    public void testStringToTimeTransformNoDateTime() throws Exception {

        Schema schema = getSchema(ColumnType.String);
        String dateTime = "2017-09-21T17:06:29.064687";
        String dateTime2 = "2007-12-30";
        String dateTime3 = "12/1/2010 11:21";

        //http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
        StringToTimeTransform transform = new StringToTimeTransform("column", null, DateTimeZone.forID("UTC"));
        transform.setInputSchema(schema);
        transform.map(new Text(dateTime3));
        transform.map(new Text(dateTime));
        transform.map(new Text(dateTime2));
        testStringToDateTime(null);




    }


    private void testStringToDateTime(String timeFormat) throws Exception {
        Schema schema = getSchema(ColumnType.String);

        //http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
        Transform transform = new StringToTimeTransform("column", timeFormat, DateTimeZone.forID("UTC"));
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Time, out.getMetaData(0).getColumnType());

        String in1 = "2016-01-01 12:30:45";
        long out1 = 1451651445000L;

        String in2 = "2015-06-30 23:59:59";
        long out2 = 1435708799000L;

        assertEquals(Collections.singletonList((Writable) new LongWritable(out1)),
                transform.map(Collections.singletonList((Writable) new Text(in1))));
        assertEquals(Collections.singletonList((Writable) new LongWritable(out2)),
                transform.map(Collections.singletonList((Writable) new Text(in2))));

        //Check serialization: things like DateTimeFormatter etc aren't serializable, hence we need custom serialization :/
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(transform);

        byte[] bytes = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        ObjectInputStream ois = new ObjectInputStream(bais);

        Transform deserialized = (Transform) ois.readObject();
        assertEquals(Collections.singletonList((Writable) new LongWritable(out1)),
                deserialized.map(Collections.singletonList((Writable) new Text(in1))));
        assertEquals(Collections.singletonList((Writable) new LongWritable(out2)),
                deserialized.map(Collections.singletonList((Writable) new Text(in2))));
    }


    @Test
    public void testDeriveColumnsFromTimeTransform() throws Exception {
        Schema schema = new Schema.Builder().addColumnTime("column", DateTimeZone.forID("UTC"))
                .addColumnString("otherColumn").build();

        Transform transform = new DeriveColumnsFromTimeTransform.Builder("column").insertAfter("otherColumn")
                .addIntegerDerivedColumn("hour", DateTimeFieldType.hourOfDay())
                .addIntegerDerivedColumn("day", DateTimeFieldType.dayOfMonth())
                .addIntegerDerivedColumn("second", DateTimeFieldType.secondOfMinute())
                .addStringDerivedColumn("humanReadable", "YYYY-MM-dd HH:mm:ss", DateTimeZone.UTC).build();

        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(6, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Time, out.getMetaData(0).getColumnType());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(1).getColumnType());
        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(2).getColumnType());
        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(3).getColumnType());
        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(4).getColumnType());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(5).getColumnType());

        assertEquals("column", out.getName(0));
        assertEquals("otherColumn", out.getName(1));
        assertEquals("hour", out.getName(2));
        assertEquals("day", out.getName(3));
        assertEquals("second", out.getName(4));
        assertEquals("humanReadable", out.getName(5));

        long in1 = 1451651445000L; //"2016-01-01 12:30:45" GMT

        List<Writable> out1 = new ArrayList<>();
        out1.add(new LongWritable(in1));
        out1.add(new Text("otherColumnValue"));
        out1.add(new IntWritable(12)); //hour
        out1.add(new IntWritable(1)); //day
        out1.add(new IntWritable(45)); //second
        out1.add(new Text("2016-01-01 12:30:45"));

        long in2 = 1435708799000L; //"2015-06-30 23:59:59" GMT
        List<Writable> out2 = new ArrayList<>();
        out2.add(new LongWritable(in2));
        out2.add(new Text("otherColumnValue"));
        out2.add(new IntWritable(23)); //hour
        out2.add(new IntWritable(30)); //day
        out2.add(new IntWritable(59)); //second
        out2.add(new Text("2015-06-30 23:59:59"));

        assertEquals(out1,
                transform.map(Arrays.asList((Writable) new LongWritable(in1), new Text("otherColumnValue"))));
        assertEquals(out2,
                transform.map(Arrays.asList((Writable) new LongWritable(in2), new Text("otherColumnValue"))));



        //Check serialization: things like DateTimeFormatter etc aren't serializable, hence we need custom serialization :/
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(transform);

        byte[] bytes = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        ObjectInputStream ois = new ObjectInputStream(bais);

        Transform deserialized = (Transform) ois.readObject();
        assertEquals(out1, deserialized
                .map(Arrays.asList((Writable) new LongWritable(in1), new Text("otherColumnValue"))));
        assertEquals(out2, deserialized
                .map(Arrays.asList((Writable) new LongWritable(in2), new Text("otherColumnValue"))));
    }


    @Test
    public void testDuplicateColumnsTransform() {

        Schema schema = new Schema.Builder().addColumnString("stringCol").addColumnInteger("intCol")
                .addColumnLong("longCol").build();

        List<String> toDup = Arrays.asList("intCol", "longCol");
        List<String> newNames = Arrays.asList("dup_intCol", "dup_longCol");

        Transform transform = new DuplicateColumnsTransform(toDup, newNames);
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(5, out.getColumnMetaData().size());

        List<String> expOutNames = Arrays.asList("stringCol", "intCol", "dup_intCol", "longCol", "dup_longCol");
        List<ColumnType> expOutTypes = Arrays.asList(ColumnType.String, ColumnType.Integer, ColumnType.Integer,
                ColumnType.Long, ColumnType.Long);
        for (int i = 0; i < 5; i++) {
            assertEquals(expOutNames.get(i), out.getName(i));
            TestCase.assertEquals(expOutTypes.get(i), out.getType(i));
        }

        List<Writable> inList = Arrays.asList((Writable) new Text("one"), new IntWritable(2), new LongWritable(3L));
        List<Writable> outList = Arrays.asList((Writable) new Text("one"), new IntWritable(2), new IntWritable(2),
                new LongWritable(3L), new LongWritable(3L));

        assertEquals(outList, transform.map(inList));
    }

    @Test
    public void testIntegerMathOpTransform() {
        Schema schema = new Schema.Builder().addColumnInteger("column", -1, 1).build();

        Transform transform = new IntegerMathOpTransform("column", MathOp.Multiply, 5);
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Integer, out.getType(0));
        IntegerMetaData meta = (IntegerMetaData) out.getMetaData(0);
        assertEquals(-5, (int) meta.getMinAllowedValue());
        assertEquals(5, (int) meta.getMaxAllowedValue());

        assertEquals(Collections.singletonList((Writable) new IntWritable(-5)),
                transform.map(Collections.singletonList((Writable) new IntWritable(-1))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(0))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(5)),
                transform.map(Collections.singletonList((Writable) new IntWritable(1))));
    }

    @Test
    public void testIntegerColumnsMathOpTransform() {
        Schema schema = new Schema.Builder().addColumnInteger("first").addColumnString("second")
                .addColumnInteger("third").build();

        Transform transform = new IntegerColumnsMathOpTransform("out", MathOp.Add, "first", "third");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(4, out.numColumns());
        assertEquals(Arrays.asList("first", "second", "third", "out"), out.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.Integer, ColumnType.String, ColumnType.Integer, ColumnType.Integer),
                out.getColumnTypes());


        assertEquals(Arrays.asList((Writable) new IntWritable(1), new Text("something"), new IntWritable(2),
                new IntWritable(3)),
                transform.map(Arrays.asList((Writable) new IntWritable(1), new Text("something"),
                        new IntWritable(2))));
        assertEquals(Arrays.asList((Writable) new IntWritable(100), new Text("something2"), new IntWritable(21),
                new IntWritable(121)),
                transform.map(Arrays.asList((Writable) new IntWritable(100), new Text("something2"),
                        new IntWritable(21))));
    }

    @Test
    public void testLongMathOpTransform() {
        Schema schema = new Schema.Builder().addColumnLong("column", -1L, 1L).build();

        Transform transform = new LongMathOpTransform("column", MathOp.Multiply, 5);
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Long, out.getType(0));
        LongMetaData meta = (LongMetaData) out.getMetaData(0);
        assertEquals(-5, (long) meta.getMinAllowedValue());
        assertEquals(5, (long) meta.getMaxAllowedValue());

        assertEquals(Collections.singletonList((Writable) new LongWritable(-5)),
                transform.map(Collections.singletonList((Writable) new LongWritable(-1))));
        assertEquals(Collections.singletonList((Writable) new LongWritable(0)),
                transform.map(Collections.singletonList((Writable) new LongWritable(0))));
        assertEquals(Collections.singletonList((Writable) new LongWritable(5)),
                transform.map(Collections.singletonList((Writable) new LongWritable(1))));
    }

    @Test
    public void testLongColumnsMathOpTransform() {
        Schema schema = new Schema.Builder().addColumnLong("first").addColumnString("second").addColumnLong("third")
                .build();

        Transform transform = new LongColumnsMathOpTransform("out", MathOp.Add, "first", "third");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(4, out.numColumns());
        assertEquals(Arrays.asList("first", "second", "third", "out"), out.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.Long, ColumnType.String, ColumnType.Long, ColumnType.Long),
                out.getColumnTypes());


        assertEquals(Arrays.asList((Writable) new LongWritable(1), new Text("something"), new LongWritable(2),
                new LongWritable(3)),
                transform.map(Arrays.asList((Writable) new LongWritable(1), new Text("something"),
                        new LongWritable(2))));
        assertEquals(Arrays.asList((Writable) new LongWritable(100), new Text("something2"), new LongWritable(21),
                new LongWritable(121)),
                transform.map(Arrays.asList((Writable) new LongWritable(100), new Text("something2"),
                        new LongWritable(21))));
    }

    @Test
    public void testTimeMathOpTransform() {
        Schema schema = new Schema.Builder().addColumnTime("column", DateTimeZone.UTC).build();

        Transform transform = new TimeMathOpTransform("column", MathOp.Add, 12, TimeUnit.HOURS); //12 hours: 43200000 milliseconds
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Time, out.getType(0));

        assertEquals(Collections.singletonList((Writable) new LongWritable(1000 + 43200000)),
                transform.map(Collections.singletonList((Writable) new LongWritable(1000))));
        assertEquals(Collections.singletonList((Writable) new LongWritable(1452441600000L + 43200000)),
                transform.map(Collections.singletonList((Writable) new LongWritable(1452441600000L))));
    }

    @Test
    public void testDoubleMathOpTransform() {
        Schema schema = new Schema.Builder().addColumnDouble("column", -1.0, 1.0).build();

        Transform transform = new DoubleMathOpTransform("column", MathOp.Multiply, 5.0);
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Double, out.getType(0));
        DoubleMetaData meta = (DoubleMetaData) out.getMetaData(0);
        assertEquals(-5.0, meta.getMinAllowedValue(), 1e-6);
        assertEquals(5.0, meta.getMaxAllowedValue(), 1e-6);

        assertEquals(Collections.singletonList((Writable) new DoubleWritable(-5)),
                transform.map(Collections.singletonList((Writable) new DoubleWritable(-1))));
        assertEquals(Collections.singletonList((Writable) new DoubleWritable(0)),
                transform.map(Collections.singletonList((Writable) new DoubleWritable(0))));
        assertEquals(Collections.singletonList((Writable) new DoubleWritable(5)),
                transform.map(Collections.singletonList((Writable) new DoubleWritable(1))));
    }

    @Test
    public void testDoubleMathFunctionTransform() {
        Schema schema = new Schema.Builder().addColumnDouble("column").addColumnString("strCol").build();

        Transform transform = new DoubleMathFunctionTransform("column", MathFunction.SIN);
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(2, out.getColumnMetaData().size());
        assertEquals(ColumnType.Double, out.getType(0));
        assertEquals(ColumnType.String, out.getType(1));

        assertEquals(Arrays.<Writable>asList(new DoubleWritable(Math.sin(1)), new Text("0")),
                transform.map(Arrays.<Writable>asList(new DoubleWritable(1), new Text("0"))));
        assertEquals(Arrays.<Writable>asList(new DoubleWritable(Math.sin(2)), new Text("1")),
                transform.map(Arrays.<Writable>asList(new DoubleWritable(2), new Text("1"))));
        assertEquals(Arrays.<Writable>asList(new DoubleWritable(Math.sin(3)), new Text("2")),
                transform.map(Arrays.<Writable>asList(new DoubleWritable(3), new Text("2"))));
    }

    @Test
    public void testDoubleColumnsMathOpTransform() {
        Schema schema = new Schema.Builder().addColumnString("first").addColumnDouble("second").addColumnDouble("third")
                .build();

        Transform transform = new DoubleColumnsMathOpTransform("out", MathOp.Add, "second", "third");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(4, out.numColumns());
        assertEquals(Arrays.asList("first", "second", "third", "out"), out.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.String, ColumnType.Double, ColumnType.Double, ColumnType.Double),
                out.getColumnTypes());


        assertEquals(Arrays.asList((Writable) new Text("something"), new DoubleWritable(1.0), new DoubleWritable(2.1),
                new DoubleWritable(3.1)),
                transform.map(Arrays.asList((Writable) new Text("something"), new DoubleWritable(1.0),
                        new DoubleWritable(2.1))));
        assertEquals(Arrays.asList((Writable) new Text("something2"), new DoubleWritable(100.0),
                new DoubleWritable(21.1), new DoubleWritable(121.1)),
                transform.map(Arrays.asList((Writable) new Text("something2"), new DoubleWritable(100.0),
                        new DoubleWritable(21.1))));
    }

    @Test
    public void testRenameColumnsTransform() {

        Schema schema = new Schema.Builder().addColumnDouble("col1").addColumnString("col2").addColumnInteger("col3")
                .build();

        Transform transform =
                new RenameColumnsTransform(Arrays.asList("col1", "col3"), Arrays.asList("column1", "column3"));
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(3, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.Double, out.getMetaData(0).getColumnType());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(1).getColumnType());
        TestCase.assertEquals(ColumnType.Integer, out.getMetaData(2).getColumnType());

        assertEquals("column1", out.getName(0));
        assertEquals("col2", out.getName(1));
        assertEquals("column3", out.getName(2));
    }

    @Test
    public void testReorderColumnsTransform() {
        Schema schema = new Schema.Builder().addColumnDouble("col1").addColumnString("col2").addColumnInteger("col3")
                .build();

        Transform transform = new ReorderColumnsTransform("col3", "col2");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(3, out.numColumns());
        assertEquals(Arrays.asList("col3", "col2", "col1"), out.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.Integer, ColumnType.String, ColumnType.Double), out.getColumnTypes());

        assertEquals(Arrays.asList((Writable) new IntWritable(1), new Text("one"), new DoubleWritable(1.1)), transform
                .map(Arrays.asList((Writable) new DoubleWritable(1.1), new Text("one"), new IntWritable(1))));

        assertEquals(Arrays.asList((Writable) new IntWritable(2), new Text("two"), new DoubleWritable(200.2)), transform
                .map(Arrays.asList((Writable) new DoubleWritable(200.2), new Text("two"), new IntWritable(2))));
    }

    @Test
    public void testConditionalReplaceValueTransform() {
        Schema schema = getSchema(ColumnType.Integer);

        Condition condition = new IntegerColumnCondition("column", ConditionOp.LessThan, 0);
        condition.setInputSchema(schema);

        Transform transform = new ConditionalReplaceValueTransform("column", new IntWritable(0), condition);
        transform.setInputSchema(schema);

        assertEquals(Collections.singletonList((Writable) new IntWritable(10)),
                transform.map(Collections.singletonList((Writable) new IntWritable(10))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(1)),
                transform.map(Collections.singletonList((Writable) new IntWritable(1))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(0))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(-1))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(-10))));
    }

    @Test
    public void testConditionalReplaceValueTransformWithDefault() {
        Schema schema = getSchema(ColumnType.Integer);

        Condition condition = new IntegerColumnCondition("column", ConditionOp.LessThan, 0);
        condition.setInputSchema(schema);

        Transform transform = new ConditionalReplaceValueTransformWithDefault("column", new IntWritable(0), new IntWritable(1), condition);
        transform.setInputSchema(schema);

        assertEquals(Collections.singletonList((Writable) new IntWritable(1)),
                transform.map(Collections.singletonList((Writable) new IntWritable(10))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(1)),
                transform.map(Collections.singletonList((Writable) new IntWritable(1))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(1)),
                transform.map(Collections.singletonList((Writable) new IntWritable(0))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(-1))));
        assertEquals(Collections.singletonList((Writable) new IntWritable(0)),
                transform.map(Collections.singletonList((Writable) new IntWritable(-10))));
    }

    @Test
    public void testConditionalCopyValueTransform() {
        Schema schema = new Schema.Builder().addColumnsString("first", "second", "third").build();

        Condition condition = new StringColumnCondition("third", ConditionOp.Equal, "");
        Transform transform = new ConditionalCopyValueTransform("third", "second", condition);
        transform.setInputSchema(schema);

        List<Writable> list = Arrays.asList((Writable) new Text("first"), new Text("second"), new Text("third"));
        assertEquals(list, transform.map(list));

        list = Arrays.asList((Writable) new Text("first"), new Text("second"), new Text(""));
        List<Writable> exp = Arrays.asList((Writable) new Text("first"), new Text("second"), new Text("second"));
        assertEquals(exp, transform.map(list));
    }

    @Test
    public void testSequenceDifferenceTransform() {
        Schema schema = new SequenceSchema.Builder().addColumnString("firstCol").addColumnInteger("secondCol")
                .addColumnDouble("thirdCol").build();

        List<List<Writable>> sequence = new ArrayList<>();
        sequence.add(Arrays.<Writable>asList(new Text("val0"), new IntWritable(10), new DoubleWritable(10)));
        sequence.add(Arrays.<Writable>asList(new Text("val1"), new IntWritable(15), new DoubleWritable(15)));
        sequence.add(Arrays.<Writable>asList(new Text("val2"), new IntWritable(25), new DoubleWritable(25)));
        sequence.add(Arrays.<Writable>asList(new Text("val3"), new IntWritable(40), new DoubleWritable(40)));

        Transform t = new SequenceDifferenceTransform("secondCol");
        t.setInputSchema(schema);

        List<List<Writable>> out = t.mapSequence(sequence);

        List<List<Writable>> expected = new ArrayList<>();
        expected.add(Arrays.<Writable>asList(new Text("val0"), new IntWritable(0), new DoubleWritable(10)));
        expected.add(Arrays.<Writable>asList(new Text("val1"), new IntWritable(15 - 10), new DoubleWritable(15)));
        expected.add(Arrays.<Writable>asList(new Text("val2"), new IntWritable(25 - 15), new DoubleWritable(25)));
        expected.add(Arrays.<Writable>asList(new Text("val3"), new IntWritable(40 - 25), new DoubleWritable(40)));

        assertEquals(expected, out);



        t = new SequenceDifferenceTransform("thirdCol", "newThirdColName", 2,
                SequenceDifferenceTransform.FirstStepMode.SpecifiedValue, NullWritable.INSTANCE);
        Schema outputSchema = t.transform(schema);
        assertTrue(outputSchema instanceof SequenceSchema);
        assertEquals(outputSchema.getColumnNames(), Arrays.asList("firstCol", "secondCol", "newThirdColName"));

        expected = new ArrayList<>();
        expected.add(Arrays.<Writable>asList(new Text("val0"), new IntWritable(10), NullWritable.INSTANCE));
        expected.add(Arrays.<Writable>asList(new Text("val1"), new IntWritable(15), NullWritable.INSTANCE));
        expected.add(Arrays.<Writable>asList(new Text("val2"), new IntWritable(25), new DoubleWritable(25 - 10)));
        expected.add(Arrays.<Writable>asList(new Text("val3"), new IntWritable(40), new DoubleWritable(40 - 15)));
    }


    @Test
    public void testAddConstantColumnTransform() {
        Schema schema = new Schema.Builder().addColumnString("first").addColumnDouble("second").build();

        Transform transform = new AddConstantColumnTransform("newCol", ColumnType.Integer, new IntWritable(10));
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(3, out.numColumns());
        assertEquals(Arrays.asList("first", "second", "newCol"), out.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.String, ColumnType.Double, ColumnType.Integer), out.getColumnTypes());


        assertEquals(Arrays.asList((Writable) new Text("something"), new DoubleWritable(1.0), new IntWritable(10)),
                transform.map(Arrays.asList((Writable) new Text("something"), new DoubleWritable(1.0))));
        assertEquals(Arrays.asList((Writable) new Text("something2"), new DoubleWritable(100.0), new IntWritable(10)),
                transform.map(Arrays.asList((Writable) new Text("something2"), new DoubleWritable(100.0))));
    }

    @Test
    public void testReplaceStringTransform() {
        Schema schema = getSchema(ColumnType.String);

        // Linked
        Map<String, String> map = new LinkedHashMap<>();
        map.put("mid", "C2");
        map.put("\\d", "one");
        Transform transform = new ReplaceStringTransform("column", map);
        transform.setInputSchema(schema);
        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new Text("BoneConeTone")),
                transform.map(Collections.singletonList((Writable) new Text("B1midT3"))));

        // No link
        map = new HashMap<>();
        map.put("^\\s+|\\s+$", "");
        transform = new ReplaceStringTransform("column", map);
        transform.setInputSchema(schema);
        out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        TestCase.assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

        assertEquals(Collections.singletonList((Writable) new Text("4.25")),
                transform.map(Collections.singletonList((Writable) new Text("  4.25 "))));
    }

    @Test
    public void testReduceSequenceTransform(){

        Schema schema = new SequenceSchema.Builder()
                .addColumnsDouble("col%d",0,2)
                .build();

        IAssociativeReducer reducer = new Reducer.Builder(ReduceOp.Mean)
                .countColumns("col1")
                .maxColumn("col2")
                .build();

        ReduceSequenceTransform t = new ReduceSequenceTransform(reducer);
        t.setInputSchema(schema);

        List<List<Writable>> seq = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(5)),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(8)));

        List<List<Writable>> exp = Collections.singletonList(
                Arrays.<Writable>asList(new DoubleWritable(3), new LongWritable(3L), new DoubleWritable(8)));
        List<List<Writable>> act = t.mapSequence(seq);
        assertEquals(exp, act);

        Schema expOutSchema = new SequenceSchema.Builder()
                .addColumnDouble("mean(col0)")
                .addColumn(new LongMetaData("count(col1)",0L,null))
                .addColumnDouble("max(col2)")
                .build();

        assertEquals(expOutSchema, t.transform(schema));
    }

    @Test
    public void testSequenceMovingWindowReduceTransform(){
        List<List<Writable>> seq = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(5)),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), new DoubleWritable(11)));

        List<List<Writable>> exp1 = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(2), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(5), new DoubleWritable((2+5)/2.0)),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(8), new DoubleWritable((2+5+8)/3.0)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), new DoubleWritable(11), new DoubleWritable((5+8+11)/3.0)));

        List<List<Writable>> exp2 = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(2), NullWritable.INSTANCE),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(5), NullWritable.INSTANCE),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(8), new DoubleWritable((2+5+8)/3.0)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), new DoubleWritable(11), new DoubleWritable((5+8+11)/3.0)));

        Schema schema = new SequenceSchema.Builder().addColumnsDouble("col%d",0,2).build();
        Schema expOutSchema1 = new SequenceSchema.Builder().addColumnsDouble("col%d",0,2).addColumnDouble("mean(3,col2)").build();
        Schema expOutSchema2 = new SequenceSchema.Builder().addColumnsDouble("col%d",0,2).addColumnDouble("newCol").build();

        SequenceMovingWindowReduceTransform t1 = new SequenceMovingWindowReduceTransform("col2",3,ReduceOp.Mean);
        SequenceMovingWindowReduceTransform t2 = new SequenceMovingWindowReduceTransform("col2","newCol",
                3,ReduceOp.Mean, SequenceMovingWindowReduceTransform.EdgeCaseHandling.SpecifiedValue, NullWritable.INSTANCE);

        t1.setInputSchema(schema);
        assertEquals(expOutSchema1, t1.transform(schema));

        t2.setInputSchema(schema);
        assertEquals(expOutSchema2, t2.transform(schema));

        List<List<Writable>> act1 = t1.mapSequence(seq);
        List<List<Writable>> act2 = t2.mapSequence(seq);

        assertEquals(exp1, act1);
        assertEquals(exp2, act2);
    }

    @Test
    public void testTrimSequenceTransform(){
        List<List<Writable>> seq = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(5)),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), new DoubleWritable(11)));

        List<List<Writable>> expTrimFirst = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), new DoubleWritable(11)));

        List<List<Writable>> expTrimLast = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(5)));

        SequenceTrimTransform tFirst = new SequenceTrimTransform(2, true);
        SequenceTrimTransform tLast = new SequenceTrimTransform(2, false);

        Schema schema = new SequenceSchema.Builder().addColumnsDouble("col%d",0,2).build();
        tFirst.setInputSchema(schema);
        tLast.setInputSchema(schema);

        assertEquals(expTrimFirst, tFirst.mapSequence(seq));
        assertEquals(expTrimLast, tLast.mapSequence(seq));
    }


    @Test
    public void testSequenceOffsetTransform(){

        List<List<Writable>> seq = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(5)),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), new DoubleWritable(11)));

        Schema schema = new SequenceSchema.Builder().addColumnsDouble("col%d",0,2).build();

        //First: test InPlace
        List<List<Writable>> exp1 = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(1), new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(4), new DoubleWritable(11)));

        List<List<Writable>> exp2 = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(7), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(10), new DoubleWritable(5)));

        //In-place + trim
        SequenceOffsetTransform t_inplace_trim_p2 = new SequenceOffsetTransform(Collections.singletonList("col1"),
                2, SequenceOffsetTransform.OperationType.InPlace, SequenceOffsetTransform.EdgeHandling.TrimSequence, null);
        SequenceOffsetTransform t_inplace_trim_m2 = new SequenceOffsetTransform(Collections.singletonList("col1"),
                -2, SequenceOffsetTransform.OperationType.InPlace, SequenceOffsetTransform.EdgeHandling.TrimSequence, null);
        t_inplace_trim_p2.setInputSchema(schema);
        t_inplace_trim_m2.setInputSchema(schema);

        assertEquals(exp1, t_inplace_trim_p2.mapSequence(seq));
        assertEquals(exp2, t_inplace_trim_m2.mapSequence(seq));


        //In-place + specified
        SequenceOffsetTransform t_inplace_specified_p2 = new SequenceOffsetTransform(Collections.singletonList("col1"),
                2, SequenceOffsetTransform.OperationType.InPlace, SequenceOffsetTransform.EdgeHandling.SpecifiedValue, NullWritable.INSTANCE);
        SequenceOffsetTransform t_inplace_specified_m2 = new SequenceOffsetTransform(Collections.singletonList("col1"),
                -2, SequenceOffsetTransform.OperationType.InPlace, SequenceOffsetTransform.EdgeHandling.SpecifiedValue, NullWritable.INSTANCE);
        t_inplace_specified_p2.setInputSchema(schema);
        t_inplace_specified_m2.setInputSchema(schema);

        List<List<Writable>> exp3 = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), NullWritable.INSTANCE, new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), NullWritable.INSTANCE, new DoubleWritable(5)),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(1), new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(4), new DoubleWritable(11)));
        List<List<Writable>> exp4 = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(7), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(10), new DoubleWritable(5)),
                Arrays.<Writable>asList(new DoubleWritable(6), NullWritable.INSTANCE, new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), NullWritable.INSTANCE, new DoubleWritable(11)));

        assertEquals(exp3, t_inplace_specified_p2.mapSequence(seq));
        assertEquals(exp4, t_inplace_specified_m2.mapSequence(seq));




        //Second: test NewColumn
        List<List<Writable>> exp1a = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(1), new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), new DoubleWritable(4), new DoubleWritable(11)));

        List<List<Writable>> exp2a = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(7), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(10), new DoubleWritable(5)));
        SequenceOffsetTransform t_newcol_trim_p2 = new SequenceOffsetTransform(Collections.singletonList("col1"),
                2, SequenceOffsetTransform.OperationType.NewColumn, SequenceOffsetTransform.EdgeHandling.TrimSequence, null);
        SequenceOffsetTransform t_newcol_trim_m2 = new SequenceOffsetTransform(Collections.singletonList("col1"),
                -2, SequenceOffsetTransform.OperationType.NewColumn, SequenceOffsetTransform.EdgeHandling.TrimSequence, null);
        t_newcol_trim_p2.setInputSchema(schema);
        t_newcol_trim_m2.setInputSchema(schema);

        assertEquals(exp1a, t_newcol_trim_p2.mapSequence(seq));
        assertEquals(exp2a, t_newcol_trim_m2.mapSequence(seq));

        List<List<Writable>> exp3a = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), NullWritable.INSTANCE, new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), NullWritable.INSTANCE, new DoubleWritable(5)),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), new DoubleWritable(1), new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), new DoubleWritable(4), new DoubleWritable(11)));
        List<List<Writable>> exp4a = Arrays.asList(
                Arrays.<Writable>asList(new DoubleWritable(0), new DoubleWritable(1), new DoubleWritable(7), new DoubleWritable(2)),
                Arrays.<Writable>asList(new DoubleWritable(3), new DoubleWritable(4), new DoubleWritable(10), new DoubleWritable(5)),
                Arrays.<Writable>asList(new DoubleWritable(6), new DoubleWritable(7), NullWritable.INSTANCE, new DoubleWritable(8)),
                Arrays.<Writable>asList(new DoubleWritable(9), new DoubleWritable(10), NullWritable.INSTANCE, new DoubleWritable(11)));

        SequenceOffsetTransform t_newcol_specified_p2 = new SequenceOffsetTransform(Collections.singletonList("col1"),
                2, SequenceOffsetTransform.OperationType.NewColumn, SequenceOffsetTransform.EdgeHandling.SpecifiedValue, NullWritable.INSTANCE);
        SequenceOffsetTransform t_newcol_specified_m2 = new SequenceOffsetTransform(Collections.singletonList("col1"),
                -2, SequenceOffsetTransform.OperationType.NewColumn, SequenceOffsetTransform.EdgeHandling.SpecifiedValue, NullWritable.INSTANCE);
        t_newcol_specified_p2.setInputSchema(schema);
        t_newcol_specified_m2.setInputSchema(schema);

        assertEquals(exp3a, t_newcol_specified_p2.mapSequence(seq));
        assertEquals(exp4a, t_newcol_specified_m2.mapSequence(seq));


        //Finally: check edge case
        assertEquals(Collections.emptyList(), t_inplace_trim_p2.mapSequence(exp1));
        assertEquals(Collections.emptyList(), t_inplace_trim_m2.mapSequence(exp1));
        assertEquals(Collections.emptyList(), t_newcol_trim_p2.mapSequence(exp1));
        assertEquals(Collections.emptyList(), t_newcol_trim_m2.mapSequence(exp1));
    }

    @Test
    public void testStringListToCountsNDArrayTransform() throws Exception {

        StringListToCountsNDArrayTransform t = new StringListToCountsNDArrayTransform(
                "inCol", "outCol", Arrays.asList("cat","dog","horse"), ",", false, true);

        Schema s = new Schema.Builder().addColumnString("inCol").build();
        t.setInputSchema(s);

        List<Writable> l = Collections.<Writable>singletonList(new Text("cat,cat,dog,dog,dog,unknown"));

        List<Writable> out = t.map(l);

        assertEquals(Collections.singletonList(new NDArrayWritable(Nd4j.create(new double[]{2,3,0}))), out);

        String json = JsonMappers.getMapper().writeValueAsString(t);
        Transform transform2 = JsonMappers.getMapper().readValue(json, StringListToCountsNDArrayTransform.class);
        Assert.assertEquals(t, transform2);
    }


    @Test
    public void testStringListToIndicesNDArrayTransform() throws Exception {

        StringListToIndicesNDArrayTransform t = new StringListToIndicesNDArrayTransform(
                "inCol", "outCol", Arrays.asList("apple", "cat","dog","horse"), ",", false, true);

        Schema s = new Schema.Builder().addColumnString("inCol").build();
        t.setInputSchema(s);

        List<Writable> l = Collections.<Writable>singletonList(new Text("cat,dog,dog,dog,unknown"));

        List<Writable> out = t.map(l);

        assertEquals(Collections.singletonList(new NDArrayWritable(Nd4j.create(new double[]{1,2,2,2}))), out);

        String json = JsonMappers.getMapper().writeValueAsString(t);
        Transform transform2 = JsonMappers.getMapper().readValue(json, StringListToIndicesNDArrayTransform.class);
        Assert.assertEquals(t, transform2);
    }


    @Test
    public void testTextToCharacterIndexTransform(){

        Schema s = new Schema.Builder().addColumnString("col").addColumnDouble("d").build();

        List<List<Writable>> inSeq = Arrays.asList(
                Arrays.<Writable>asList(new Text("text"), new DoubleWritable(1.0)),
                Arrays.<Writable>asList(new Text("ab"), new DoubleWritable(2.0)));

        Map<Character,Integer> map = new HashMap<>();
        map.put('a', 0);
        map.put('b', 1);
        map.put('e', 2);
        map.put('t', 3);
        map.put('x', 4);

        List<List<Writable>> exp = Arrays.asList(
                Arrays.<Writable>asList(new IntWritable(3), new DoubleWritable(1.0)),
                Arrays.<Writable>asList(new IntWritable(2), new DoubleWritable(1.0)),
                Arrays.<Writable>asList(new IntWritable(4), new DoubleWritable(1.0)),
                Arrays.<Writable>asList(new IntWritable(3), new DoubleWritable(1.0)),
                Arrays.<Writable>asList(new IntWritable(0), new DoubleWritable(2.0)),
                Arrays.<Writable>asList(new IntWritable(1), new DoubleWritable(2.0)));

        Transform t = new TextToCharacterIndexTransform("col", "newName", map, false);
        t.setInputSchema(s);

        Schema outputSchema = t.transform(s);
        assertEquals(2, outputSchema.getColumnNames().size());
        assertEquals(ColumnType.Integer, outputSchema.getType(0));
        assertEquals(ColumnType.Double, outputSchema.getType(1));

        IntegerMetaData intMetadata = (IntegerMetaData)outputSchema.getMetaData(0);
        assertEquals(0, (int)intMetadata.getMinAllowedValue());
        assertEquals(4, (int)intMetadata.getMaxAllowedValue());

        List<List<Writable>> out = t.mapSequence(inSeq);
        assertEquals(exp, out);
    }

    @Test
    public void testTextToTermIndexSequenceTransform(){

        Schema schema = new Schema.Builder()
                .addColumnString("ID")
                .addColumnString("TEXT")
                .addColumnDouble("FEATURE")
                .build();
        List<String> vocab = Arrays.asList("zero", "one", "two", "three");
        List<List<Writable>> inSeq = Arrays.asList(
                Arrays.<Writable>asList(new Text("a"), new Text("zero four two"), new DoubleWritable(4.2)),
                Arrays.<Writable>asList(new Text("b"), new Text("six one two four three five"), new DoubleWritable(87.9)));

        Schema expSchema = new Schema.Builder()
                .addColumnString("ID")
                .addColumnInteger("INDEXSEQ", 0, 3)
                .addColumnDouble("FEATURE")
                .build();
        List<List<Writable>> exp = Arrays.asList(
                Arrays.<Writable>asList(new Text("a"), new IntWritable(0), new DoubleWritable(4.2)),
                Arrays.<Writable>asList(new Text("a"), new IntWritable(2), new DoubleWritable(4.2)),
                Arrays.<Writable>asList(new Text("b"), new IntWritable(1), new DoubleWritable(87.9)),
                Arrays.<Writable>asList(new Text("b"), new IntWritable(2), new DoubleWritable(87.9)),
                Arrays.<Writable>asList(new Text("b"), new IntWritable(3), new DoubleWritable(87.9)));

        Transform t = new TextToTermIndexSequenceTransform("TEXT", "INDEXSEQ", vocab, " ", false);
        t.setInputSchema(schema);

        Schema outputSchema = t.transform(schema);
        assertEquals(expSchema.getColumnNames(), outputSchema.getColumnNames());
        assertEquals(expSchema.getColumnTypes(), outputSchema.getColumnTypes());
        assertEquals(expSchema, outputSchema);

        assertEquals(3, outputSchema.getColumnNames().size());
        assertEquals(ColumnType.String, outputSchema.getType(0));
        assertEquals(ColumnType.Integer, outputSchema.getType(1));
        assertEquals(ColumnType.Double, outputSchema.getType(2));

        IntegerMetaData intMetadata = (IntegerMetaData)outputSchema.getMetaData(1);
        assertEquals(0, (int)intMetadata.getMinAllowedValue());
        assertEquals(3, (int)intMetadata.getMaxAllowedValue());

        List<List<Writable>> out = t.mapSequence(inSeq);
        assertEquals(exp, out);

        TransformProcess tp = new TransformProcess.Builder(schema).transform(t).build();
        String json = tp.toJson();
        TransformProcess tp2 = TransformProcess.fromJson(json);
        assertEquals(tp, tp2);
    }
}
