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

package org.datavec.api.transform.transform.categorical;

import lombok.Data;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.List;

/**
 * Convert a String column
 * to a categorical column
 */
@JsonIgnoreProperties({"inputSchema", "columnNumber"})
@Data
public class StringToCategoricalTransform extends BaseColumnTransform {

    private final List<String> stateNames;

    public StringToCategoricalTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("stateNames") List<String> stateNames) {
        super(columnName);
        if(stateNames == null || stateNames.isEmpty()) {
            throw new IllegalArgumentException("State names must not be null or empty");
        }

        this.stateNames = stateNames;
    }

    public StringToCategoricalTransform(String columnName, String... stateNames) {
        this(columnName, Arrays.asList(stateNames));
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        return new CategoricalMetaData(newColumnName, stateNames);
    }

    @Override
    public Writable map(Writable columnWritable) {
        return columnWritable;
    }

    @Override
    public String toString() {
        return "StringToCategoricalTransform(stateNames=" + stateNames + ")";
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
        return input;
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        return sequence;
    }
}
