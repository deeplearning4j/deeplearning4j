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
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Metadata for an String column
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class StringMetaData extends BaseColumnMetaData {

    //regex + min/max length are nullable: null -> no restrictions on these
    private final String regex;
    private final Integer minLength;
    private final Integer maxLength;

    public StringMetaData() {
        super(null);
        regex = null;
        minLength = null;
        maxLength = null;
    }

    /**
     * Default constructor with no restrictions on allowable strings
     */
    public StringMetaData(String name) {
        this(name, null, null, null);
    }

    /**
     * @param mustMatchRegex Nullable. If not null: this is a regex that each string must match in order for the entry
     *                       to be considered valid.
     * @param minLength      Min allowable String length. If null: no restriction on min String length
     * @param maxLength      Max allowable String length. If null: no restriction on max String length
     */
    public StringMetaData(@JsonProperty("name") String name, @JsonProperty("regex") String mustMatchRegex,
                    @JsonProperty("minLength") Integer minLength, @JsonProperty("maxLength") Integer maxLength) {
        super(name);
        this.regex = mustMatchRegex;
        this.minLength = minLength;
        this.maxLength = maxLength;
    }


    @Override
    public ColumnType getColumnType() {
        return ColumnType.String;
    }

    @Override
    public boolean isValid(Writable writable) {
        String str = writable.toString();
        int len = str.length();
        if (minLength != null && len < minLength)
            return false;
        if (maxLength != null && len > maxLength)
            return false;

        return regex == null || str.matches(regex);
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
        String str = input.toString();
        int len = str.length();
        if (minLength != null && len < minLength)
            return false;
        if (maxLength != null && len > maxLength)
            return false;

        return regex == null || str.matches(regex);
    }

    @Override
    public StringMetaData clone() {
        return new StringMetaData(name, regex, minLength, maxLength);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("StringMetaData(name=\"").append(name).append("\",");
        if (minLength != null)
            sb.append("minLengthAllowed=").append(minLength);
        if (maxLength != null) {
            if (minLength != null)
                sb.append(",");
            sb.append("maxLengthAllowed=").append(maxLength);
        }
        if (regex != null) {
            if (minLength != null || maxLength != null)
                sb.append(",");
            sb.append("regex=").append(regex);
        }
        sb.append(")");
        return sb.toString();
    }

}
