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

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.types.StructType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.DataFrames;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * Convert a record to a row
 * @author Adam Gibson
 */
public class ToRow implements Function<List<Writable>, Row> {
    private Schema schema;
    private StructType structType;

    public ToRow(Schema schema) {
        this.schema = schema;
        structType = DataFrames.fromSchema(schema);
    }

    @Override
    public Row call(List<Writable> v1) throws Exception {
        if (v1.size() == 1 && v1.get(0) instanceof NDArrayWritable) {
            NDArrayWritable writable = (NDArrayWritable) v1.get(0);
            INDArray arr = writable.get();
            if (arr.columns() != schema.numColumns())
                throw new IllegalStateException(
                                "Illegal record of size " + v1 + ". Should have been " + schema.numColumns());
            Object[] values = new Object[arr.columns()];
            for (int i = 0; i < values.length; i++) {
                switch (schema.getColumnTypes().get(i)) {
                    case Double:
                        values[i] = arr.getDouble(i);
                        break;
                    case Integer:
                        values[i] = (int) arr.getDouble(i);
                        break;
                    case Long:
                        values[i] = (long) arr.getDouble(i);
                        break;
                    case Float:
                        values[i] = (float) arr.getDouble(i);
                        break;
                    default:
                        throw new IllegalStateException(
                                        "This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
                }
            }

            Row row = new GenericRowWithSchema(values, structType);
            return row;
        } else {
            if (v1.size() != schema.numColumns())
                throw new IllegalStateException(
                                "Illegal record of size " + v1 + ". Should have been " + schema.numColumns());
            Object[] values = new Object[v1.size()];
            for (int i = 0; i < values.length; i++) {
                switch (schema.getColumnTypes().get(i)) {
                    case Double:
                        values[i] = v1.get(i).toDouble();
                        break;
                    case Integer:
                        values[i] = v1.get(i).toInt();
                        break;
                    case Long:
                        values[i] = v1.get(i).toLong();
                        break;
                    case Float:
                        values[i] = v1.get(i).toFloat();
                        break;
                    default:
                        throw new IllegalStateException(
                                        "This api should not be used with strings , binary data or ndarrays. This is only for columnar data");
                }
            }

            Row row = new GenericRowWithSchema(values, structType);
            return row;
        }

    }
}
