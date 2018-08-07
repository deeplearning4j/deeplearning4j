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

package org.datavec.api.transform.filter;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Remove invalid records of a certain size.
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
public class InvalidNumColumns implements Filter {
    private Schema inputSchema;

    /**
     * @param writables Example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeExample(Object writables) {
        return false;
    }

    /**
     * @param sequence sequence example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeSequence(Object sequence) {
        return false;
    }

    /**
     * @param writables Example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeExample(List<Writable> writables) {
        return writables.size() != inputSchema.numColumns();
    }

    /**
     * @param sequence sequence example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        for (List<Writable> record : sequence)
            if (record.size() != inputSchema.numColumns())
                return true;
        return false;
    }

    /**
     * Get the output schema for this transformation, given an input schema
     *
     * @param inputSchema
     */
    @Override
    public Schema transform(Schema inputSchema) {
        return inputSchema;
    }

    @Override
    public void setInputSchema(Schema schema) {
        this.inputSchema = schema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return inputSchema.getColumnNames().get(0);
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
        return columnNames()[0];
    }
}
