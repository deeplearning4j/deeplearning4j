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

package org.datavec.api.transform.transform.doubletransform;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;

/**
 * Convert any value to an Double
 *
 * @author Justin Long (crockpotveggies)
 */
@NoArgsConstructor
@Data
public class ConvertToDouble extends BaseDoubleTransform {

    /**
     * @param column Name of the column to convert to a Double column
     */
    public ConvertToDouble(String column) {
        super(column);
    }

    @Override
    public DoubleWritable map(Writable writable) {
        if(writable.getType() == WritableType.Double){
            return (DoubleWritable)writable;
        }
        return new DoubleWritable(writable.toDouble());
    }


    @Override
    public Object map(Object input) {
        if(input instanceof Number){
            return ((Number) input).doubleValue();
        }
        return Double.parseDouble(input.toString());
    }


    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        return new DoubleMetaData(newColumnName);
    }

    @Override
    public String toString() {
        return "ConvertToDouble(columnName=" + columnName + ")";
    }
}
