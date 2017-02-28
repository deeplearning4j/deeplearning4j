/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.filter;

import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.IntegerColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 21/03/2016.
 */
public class TestFilters {


    @Test
    public void testFilterNumColumns() {
        List<List<Writable>> list = new ArrayList<>();
        list.add(Collections.singletonList((Writable) new IntWritable(-1)));
        list.add(Collections.singletonList((Writable) new IntWritable(0)));
        list.add(Collections.singletonList((Writable) new IntWritable(2)));

        Schema schema = new Schema.Builder().addColumnInteger("intCol", 0, 10) //Only values in the range 0 to 10 are ok
                        .addColumnDouble("doubleCol", -100.0, 100.0) //-100 to 100 only; no NaN or infinite
                        .build();
        Filter numColumns = new InvalidNumColumns(schema);
        for (int i = 0; i < list.size(); i++)
            assertTrue(numColumns.removeExample(list.get(i)));

        List<Writable> correct = Arrays.<Writable>asList(new IntWritable(0), new DoubleWritable(2));
        assertFalse(numColumns.removeExample(correct));

    }

    @Test
    public void testFilterInvalidValues() {

        List<List<Writable>> list = new ArrayList<>();
        list.add(Collections.singletonList((Writable) new IntWritable(-1)));
        list.add(Collections.singletonList((Writable) new IntWritable(0)));
        list.add(Collections.singletonList((Writable) new IntWritable(2)));

        Schema schema = new Schema.Builder().addColumnInteger("intCol", 0, 10) //Only values in the range 0 to 10 are ok
                        .addColumnDouble("doubleCol", -100.0, 100.0) //-100 to 100 only; no NaN or infinite
                        .build();

        Filter filter = new FilterInvalidValues("intCol", "doubleCol");
        filter.setInputSchema(schema);

        //Test valid examples:
        assertFalse(filter.removeExample(asList((Writable) new IntWritable(0), new DoubleWritable(0))));
        assertFalse(filter.removeExample(asList((Writable) new IntWritable(10), new DoubleWritable(0))));
        assertFalse(filter.removeExample(asList((Writable) new IntWritable(0), new DoubleWritable(-100))));
        assertFalse(filter.removeExample(asList((Writable) new IntWritable(0), new DoubleWritable(100))));

        //Test invalid:
        assertTrue(filter.removeExample(asList((Writable) new IntWritable(-1), new DoubleWritable(0))));
        assertTrue(filter.removeExample(asList((Writable) new IntWritable(11), new DoubleWritable(0))));
        assertTrue(filter.removeExample(asList((Writable) new IntWritable(0), new DoubleWritable(-101))));
        assertTrue(filter.removeExample(asList((Writable) new IntWritable(0), new DoubleWritable(101))));
    }

    @Test
    public void testConditionFilter() {
        Schema schema = new Schema.Builder().addColumnInteger("column").build();

        Condition condition = new IntegerColumnCondition("column", ConditionOp.LessThan, 0);
        condition.setInputSchema(schema);

        Filter filter = new ConditionFilter(condition);

        assertFalse(filter.removeExample(Collections.singletonList((Writable) new IntWritable(10))));
        assertFalse(filter.removeExample(Collections.singletonList((Writable) new IntWritable(1))));
        assertFalse(filter.removeExample(Collections.singletonList((Writable) new IntWritable(0))));
        assertTrue(filter.removeExample(Collections.singletonList((Writable) new IntWritable(-1))));
        assertTrue(filter.removeExample(Collections.singletonList((Writable) new IntWritable(-10))));
    }

}
