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

package org.datavec.api.transform.stringreduce;

import org.datavec.api.transform.StringReduceOp;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 21/03/2016.
 */
public class TestReduce {

    @Test
    public void testReducerDouble() {

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new Text("1"), new Text("2")));
        inputs.add(Arrays.asList((Writable) new Text("1"), new Text("2")));
        inputs.add(Arrays.asList((Writable) new Text("1"), new Text("2")));

        Map<StringReduceOp, String> exp = new LinkedHashMap<>();
        exp.put(StringReduceOp.MERGE, "12");
        exp.put(StringReduceOp.APPEND, "12");
        exp.put(StringReduceOp.PREPEND, "21");
        exp.put(StringReduceOp.REPLACE, "2");

        for (StringReduceOp op : exp.keySet()) {

            Schema schema = new Schema.Builder().addColumnString("key").addColumnString("column").build();

            StringReducer reducer = new StringReducer.Builder(op).build();

            reducer.setInputSchema(schema);

            List<Writable> out = reducer.reduce(inputs);

            assertEquals(3, out.size());
            assertEquals(exp.get(op), out.get(0).toString());
        }
    }


}
