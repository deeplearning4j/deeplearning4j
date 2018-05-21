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

package org.datavec.api.transform.serde.testClasses;

import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Alex on 11/01/2017.
 */
public class CustomTransform extends BaseColumnTransform {

    private final double someArg;

    public CustomTransform(@JsonProperty("columnName") String columnName, @JsonProperty("someArg") double someArg) {
        super(columnName);
        this.someArg = someArg;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new DoubleMetaData(newName);
    }

    @Override
    public Writable map(Writable columnWritable) {
        return columnWritable;
    }

    @Override
    public String toString() {
        return "CustomTransform()";
    }

    @Override
    public Object map(Object input) {
        return input;
    }
}
