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

package org.datavec.api.transform.transform.column;

import lombok.Data;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Add a new column, where the values in that column for all records are identical (according to the specified value)
 *
 * @author Alex Black
 */
@Data
public class AddConstantColumnTransform implements Transform {

    private final String newColumnName;
    private final ColumnType newColumnType;
    private final Writable fixedValue;

    private Schema inputSchema;


    public AddConstantColumnTransform(@JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("newColumnType") ColumnType newColumnType,
                    @JsonProperty("fixedValue") Writable fixedValue) {
        this.newColumnName = newColumnName;
        this.newColumnType = newColumnType;
        this.fixedValue = fixedValue;
    }


    @Override
    public Schema transform(Schema inputSchema) {
        List<ColumnMetaData> outMeta = new ArrayList<>();
        outMeta.addAll(inputSchema.getColumnMetaData());

        ColumnMetaData newColMeta = newColumnType.newColumnMetaData(newColumnName);
        outMeta.add(newColMeta);
        return inputSchema.newSchema(outMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public String outputColumnName() {
        return newColumnName;
    }

    @Override
    public String[] outputColumnNames() {
        return new String[] {outputColumnName()};
    }

    @Override
    public String[] columnNames() {
        return new String[0];
    }

    @Override
    public String columnName() {
        return newColumnName;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        List<Writable> out = new ArrayList<>(writables.size() + 1);
        out.addAll(writables);
        out.add(fixedValue);
        return out;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> outSeq = new ArrayList<>(sequence.size());
        for (List<Writable> l : sequence) {
            outSeq.add(map(l));
        }
        return outSeq;
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException();
    }
}
