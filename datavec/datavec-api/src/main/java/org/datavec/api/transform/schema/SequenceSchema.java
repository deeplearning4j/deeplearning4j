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

package org.datavec.api.transform.schema;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.writable.*;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * A SequenceSchema is a {@link Schema} for sequential data.
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class SequenceSchema extends Schema {
    private final Integer minSequenceLength;
    private final Integer maxSequenceLength;

    public SequenceSchema(List<ColumnMetaData> columnMetaData) {
        this(columnMetaData, null, null);
    }

    public SequenceSchema(@JsonProperty("columns") List<ColumnMetaData> columnMetaData,
                    @JsonProperty("minSequenceLength") Integer minSequenceLength,
                    @JsonProperty("maxSequenceLength") Integer maxSequenceLength) {
        super(columnMetaData);
        this.minSequenceLength = minSequenceLength;
        this.maxSequenceLength = maxSequenceLength;
    }

    private SequenceSchema(Builder builder) {
        super(builder);
        this.minSequenceLength = builder.minSequenceLength;
        this.maxSequenceLength = builder.maxSequenceLength;
    }

    @Override
    public SequenceSchema newSchema(List<ColumnMetaData> columnMetaData) {
        return new SequenceSchema(columnMetaData, minSequenceLength, maxSequenceLength);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        int nCol = numColumns();

        int maxNameLength = 0;
        for (String s : getColumnNames()) {
            maxNameLength = Math.max(maxNameLength, s.length());
        }

        //Header:
        sb.append("SequenceSchema(");

        if (minSequenceLength != null)
            sb.append("minSequenceLength=").append(minSequenceLength);
        if (maxSequenceLength != null) {
            if (minSequenceLength != null)
                sb.append(",");
            sb.append("maxSequenceLength=").append(maxSequenceLength);
        }

        sb.append(")\n");
        sb.append(String.format("%-6s", "idx")).append(String.format("%-" + (maxNameLength + 8) + "s", "name"))
                        .append(String.format("%-15s", "type")).append("meta data").append("\n");

        for (int i = 0; i < nCol; i++) {
            String colName = getName(i);
            ColumnType type = getType(i);
            ColumnMetaData meta = getMetaData(i);
            String paddedName = String.format("%-" + (maxNameLength + 8) + "s", "\"" + colName + "\"");
            sb.append(String.format("%-6d", i)).append(paddedName).append(String.format("%-15s", type)).append(meta)
                            .append("\n");
        }

        return sb.toString();
    }

    public static class Builder extends Schema.Builder {

        private Integer minSequenceLength;
        private Integer maxSequenceLength;

        public Builder minSequenceLength(int minSequenceLength) {
            this.minSequenceLength = minSequenceLength;
            return this;
        }

        public Builder maxSequenceLength(int maxSequenceLength) {
            this.maxSequenceLength = maxSequenceLength;
            return this;
        }


        @Override
        public SequenceSchema build() {
            return new SequenceSchema(this);
        }


    }


    /**
     * Infers a sequence schema based
     * on the record
     * @param record the record to infer the schema based on
     * @return the inferred sequence schema
     *
     */
    public static SequenceSchema inferSequenceMulti(List<List<List<Writable>>> record) {
        SequenceSchema.Builder builder = new SequenceSchema.Builder();
        int minSequenceLength = record.get(0).size();
        int maxSequenceLength = record.get(0).size();
        for (int i = 0; i < record.size(); i++) {
            if (record.get(i) instanceof DoubleWritable)
                builder.addColumnDouble(String.valueOf(i));
            else if (record.get(i) instanceof IntWritable)
                builder.addColumnInteger(String.valueOf(i));
            else if (record.get(i) instanceof LongWritable)
                builder.addColumnLong(String.valueOf(i));
            else if (record.get(i) instanceof FloatWritable)
                builder.addColumnFloat(String.valueOf(i));

            else
                throw new IllegalStateException("Illegal writable for inferring schema of type "
                                + record.get(i).getClass().toString() + " with record " + record.get(0));
            builder.minSequenceLength(Math.min(record.get(i).size(), minSequenceLength));
            builder.maxSequenceLength(Math.max(record.get(i).size(), maxSequenceLength));
        }


        return builder.build();
    }

    /**
     * Infers a sequence schema based
     * on the record
     * @param record the record to infer the schema based on
     * @return the inferred sequence schema
     *
     */
    public static SequenceSchema inferSequence(List<List<Writable>> record) {
        SequenceSchema.Builder builder = new SequenceSchema.Builder();
        for (int i = 0; i < record.size(); i++) {
            if (record.get(i) instanceof DoubleWritable)
                builder.addColumnDouble(String.valueOf(i));
            else if (record.get(i) instanceof IntWritable)
                builder.addColumnInteger(String.valueOf(i));
            else if (record.get(i) instanceof LongWritable)
                builder.addColumnLong(String.valueOf(i));
            else if (record.get(i) instanceof FloatWritable)
                builder.addColumnFloat(String.valueOf(i));

            else
                throw new IllegalStateException(
                                "Illegal writable for infering schema of type " + record.get(i).getClass().toString());
        }

        builder.minSequenceLength(record.size());
        builder.maxSequenceLength(record.size());
        return builder.build();
    }
}
