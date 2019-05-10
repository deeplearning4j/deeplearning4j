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

package org.datavec.api.transform;

import org.datavec.api.transform.metadata.*;
import org.datavec.api.writable.WritableType;

/**
 * The type of column.
 */
public enum ColumnType {
    String, Integer, Long, Double, Float, Categorical, Time, Bytes, //Arbitrary byte[] data
    Boolean, NDArray;

    public ColumnMetaData newColumnMetaData(String columnName) {
        switch (this) {
            case String:
                return new StringMetaData(columnName);
            case Integer:
                return new IntegerMetaData(columnName);
            case Long:
                return new LongMetaData(columnName);
            case Double:
                return new DoubleMetaData(columnName);
            case Float:
                return new FloatMetaData(columnName);
            case Time:
                return new TimeMetaData(columnName);
            case Boolean:
                return new CategoricalMetaData(columnName, "true", "false");
            case Categorical:
                throw new UnsupportedOperationException(
                                "Cannot create new categorical column using this method: categorical state names would be unknown");
            case NDArray:
                throw new UnsupportedOperationException(
                                "Cannot create new NDArray column using this method: shape information would be unknown");
            default: //And Bytes
                throw new UnsupportedOperationException("Unknown or not supported column type: " + this);
        }
    }

    /**
     * @return The type of writable for this column
     */
    public WritableType getWritableType(){
        switch (this){
            case String:
                return WritableType.Text;
            case Integer:
                return WritableType.Int;
            case Long:
                return WritableType.Long;
            case Double:
                return WritableType.Double;
            case Float:
                return WritableType.Float;
            case Categorical:
                return WritableType.Text;
            case Time:
                return WritableType.Long;
            case Bytes:
                return WritableType.Byte;
            case Boolean:
                return WritableType.Boolean;
            case NDArray:
                return WritableType.Image;
            default:
                throw new IllegalStateException("Unknown writable type for column type: " + this);
        }
    }

}
