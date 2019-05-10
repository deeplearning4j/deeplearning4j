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

package org.datavec.api.transform.transform.parse;

import lombok.Data;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 *  Convert string writables to doubles
 *
 *  @author Adam GIbson
 */
@Data
public class ParseDoubleTransform extends BaseTransform {

    @Override
    public String toString() {
        return getClass().getName();
    }

    /**
     * Get the output schema for this transformation, given an input schema
     *
     * @param inputSchema
     */
    @Override
    public Schema transform(Schema inputSchema) {
        Schema.Builder newSchema = new Schema.Builder();
        for (int i = 0; i < inputSchema.numColumns(); i++) {
            if (inputSchema.getType(i) == ColumnType.String) {
                newSchema.addColumnDouble(inputSchema.getMetaData(i).getName());
            } else
                newSchema.addColumn(inputSchema.getMetaData(i));

        }
        return newSchema.build();
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        List<Writable> transform = new ArrayList<>();
        for (Writable w : writables) {
            if (w instanceof Text) {
                transform.add(new DoubleWritable(w.toDouble()));
            } else {
                transform.add(w);
            }
        }
        return transform;
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        return Double.parseDouble(input.toString());
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<?> list = (List<?>) sequence;
        List<Object> ret = new ArrayList<>();
        for (Object o : list)
            ret.add(map(o));
        return ret;
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return getInputSchema().getColumnNames().get(0);
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return inputSchema.getColumnNames().toArray(new String[inputSchema.numColumns()]);
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return inputSchema.getColumnNames().toArray(new String[inputSchema.numColumns()]);
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return outputColumnName();
    }
}
