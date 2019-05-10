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

package org.datavec.spark.transform.sparkfunction;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Converts a row to a record
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
public class ToRecord implements Function<Row, List<Writable>> {
    private Schema schema;

    @Override
    public List<Writable> call(Row v1) throws Exception {
        List<Writable> ret = new ArrayList<>();
        if (v1.size() != schema.numColumns())
            throw new IllegalArgumentException("Invalid number of columns for row " + v1.size()
                            + " should have matched schema columns " + schema.numColumns());
        for (int i = 0; i < v1.size(); i++) {
            if (v1.get(i) == null)
                throw new IllegalStateException("Row item " + i + " is null");
            switch (schema.getType(i)) {
                case Double:
                    ret.add(new DoubleWritable(v1.getDouble(i)));
                    break;
                case Float:
                    ret.add(new FloatWritable(v1.getFloat(i)));
                    break;
                case Integer:
                    ret.add(new IntWritable(v1.getInt(i)));
                    break;
                case Long:
                    ret.add(new LongWritable(v1.getLong(i)));
                    break;
                default:
                    throw new IllegalStateException("Illegal type");
            }

        }
        return ret;
    }
}
