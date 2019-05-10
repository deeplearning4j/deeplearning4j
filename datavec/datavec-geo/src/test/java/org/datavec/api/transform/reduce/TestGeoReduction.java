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

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.ops.IAggregableReduceOp;
import org.datavec.api.transform.reduce.geo.CoordinatesReduction;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author saudet
 */
public class TestGeoReduction {

    @Test
    public void testCustomReductions() {

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("1#5")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("2#6")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("3#7")));
        inputs.add(Arrays.asList((Writable) new Text("someKey"), new Text("4#8")));

        List<Writable> expected = Arrays.asList((Writable) new Text("someKey"), new Text("10.0#26.0"));

        Schema schema = new Schema.Builder().addColumnString("key").addColumnString("coord").build();

        Reducer reducer = new Reducer.Builder(ReduceOp.Count).keyColumns("key")
                        .customReduction("coord", new CoordinatesReduction("coordSum", ReduceOp.Sum, "#")).build();

        reducer.setInputSchema(schema);

        IAggregableReduceOp<List<Writable>, List<Writable>> aggregableReduceOp = reducer.aggregableReducer();
        for (List<Writable> l : inputs)
            aggregableReduceOp.accept(l);
        List<Writable> out = aggregableReduceOp.get();

        assertEquals(2, out.size());
        assertEquals(expected, out);

        //Check schema:
        String[] expNames = new String[] {"key", "coordSum"};
        ColumnType[] expTypes = new ColumnType[] {ColumnType.String, ColumnType.String};
        Schema outSchema = reducer.transform(schema);

        assertEquals(2, outSchema.numColumns());
        for (int i = 0; i < 2; i++) {
            assertEquals(expNames[i], outSchema.getName(i));
            assertEquals(expTypes[i], outSchema.getType(i));
        }
    }
}
