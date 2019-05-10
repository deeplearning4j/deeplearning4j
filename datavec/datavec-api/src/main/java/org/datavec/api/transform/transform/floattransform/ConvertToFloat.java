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

package org.datavec.api.transform.transform.floattransform;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.FloatMetaData;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;

/**
 * Convert any value to a Float
 *
 * @author Fariz Rahman (farizrahman4u)
 */
@NoArgsConstructor
@Data
public class ConvertToFloat extends BaseFloatTransform {

    /**
     * @param column Name of the column to convert to a Float column
     */
    public ConvertToFloat(String column) {
        super(column);
    }

    @Override
    public FloatWritable map(Writable writable) {
        if(writable.getType() == WritableType.Double){
            return (FloatWritable)writable;
        }
        return new FloatWritable(writable.toFloat());
    }


    @Override
    public Object map(Object input) {
        if(input instanceof Number){
            return ((Number) input).floatValue();
        }
        return Float.parseFloat(input.toString());
    }


    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        return new FloatMetaData(newColumnName);
    }

    @Override
    public String toString() {
        return "ConvertToFloat(columnName=" + columnName + ")";
    }
}
