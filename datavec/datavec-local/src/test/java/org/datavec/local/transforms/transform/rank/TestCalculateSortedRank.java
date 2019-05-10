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

package org.datavec.local.transforms.transform.rank;


import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.comparator.DoubleWritableComparator;


import org.datavec.local.transforms.LocalTransformExecutor;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 1/06/2016.
 */
public class TestCalculateSortedRank  {

    @Test
    public void testCalculateSortedRank() {

        List<List<Writable>> data = new ArrayList<>();
        data.add(Arrays.asList((Writable) new Text("0"), new DoubleWritable(0.0)));
        data.add(Arrays.asList((Writable) new Text("3"), new DoubleWritable(0.3)));
        data.add(Arrays.asList((Writable) new Text("2"), new DoubleWritable(0.2)));
        data.add(Arrays.asList((Writable) new Text("1"), new DoubleWritable(0.1)));

        List<List<Writable>> rdd = (data);

        Schema schema = new Schema.Builder().addColumnsString("TextCol").addColumnDouble("DoubleCol").build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                        .calculateSortedRank("rank", "DoubleCol", new DoubleWritableComparator()).build();

        Schema outSchema = tp.getFinalSchema();
        assertEquals(3, outSchema.numColumns());
        assertEquals(Arrays.asList("TextCol", "DoubleCol", "rank"), outSchema.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.String, ColumnType.Double, ColumnType.Long), outSchema.getColumnTypes());

        List<List<Writable>> out = LocalTransformExecutor.execute(rdd, tp);

        List<List<Writable>> collected = out;
        assertEquals(4, collected.size());
        for (int i = 0; i < 4; i++)
            assertEquals(3, collected.get(i).size());

        for (List<Writable> example : collected) {
            int exampleNum = example.get(0).toInt();
            int rank = example.get(2).toInt();
            assertEquals(exampleNum, rank);
        }
    }

}
