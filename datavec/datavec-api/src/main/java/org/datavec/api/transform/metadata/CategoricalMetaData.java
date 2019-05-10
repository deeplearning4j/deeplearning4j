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

package org.datavec.api.transform.metadata;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Metadata for categorical columns.
 */
@JsonIgnoreProperties({"stateNamesSet"})
@EqualsAndHashCode
@Data
public class CategoricalMetaData extends BaseColumnMetaData {

    private final List<String> stateNames;
    private final Set<String> stateNamesSet; //For fast lookup

    public CategoricalMetaData(String name, String... stateNames) {
        this(name, Arrays.asList(stateNames));
    }

    public CategoricalMetaData(@JsonProperty("name") String name, @JsonProperty("stateNames") List<String> stateNames) {
        super(name);
        this.stateNames = stateNames;
        stateNamesSet = new HashSet<>(stateNames);
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Categorical;
    }

    @Override
    public boolean isValid(Writable writable) {
        return stateNamesSet.contains(writable.toString());
    }

    /**
     * Is the given object valid for this column,
     * given the column type and any
     * restrictions given by the
     * ColumnMetaData object?
     *
     * @param input object to check
     * @return true if value, false if invalid
     */
    @Override
    public boolean isValid(Object input) {
        return stateNamesSet.contains(input.toString());
    }

    @Override
    public CategoricalMetaData clone() {
        return new CategoricalMetaData(name, stateNames);
    }

    public List<String> getStateNames() {
        return stateNames;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("CategoricalMetaData(name=\"").append(name).append("\",stateNames=[");
        boolean first = true;
        for (String s : stateNamesSet) {
            if (!first)
                sb.append(",");
            sb.append("\"").append(s).append("\"");
            first = false;
        }
        sb.append("])");
        return sb.toString();
    }
}
