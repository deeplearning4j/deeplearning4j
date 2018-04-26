/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.transform.string;

import lombok.Data;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Append a String to the
 * values in a single column
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema", "columnNumber"})
@Data
public class AppendStringColumnTransform extends BaseColumnTransform {

    private String toAppend;

    public AppendStringColumnTransform(@JsonProperty("columnName") String columnName, @JsonProperty("toAppend") String toAppend) {
        super(columnName);
        this.toAppend = toAppend;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new StringMetaData(newName); //Output after transform: String (Text)
    }

    @Override
    public Writable map(Writable columnWritable) {
        return new Text(columnWritable + toAppend);
    }

    @Override
    public String toString() {
        return "AppendStringColumnTransform(append=\"" + toAppend + "\")";
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
        return input.toString() + toAppend;
    }
}
