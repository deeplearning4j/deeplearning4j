/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.imports.descriptors.properties.adapters;

import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.tensorflow.framework.DataType;

import java.lang.reflect.Field;

public class DataTypeAdapter implements AttributeAdapter {

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        on.setValueFor(fieldFor,dtypeConv((DataType) inputAttributeValue));
    }

    public static org.nd4j.linalg.api.buffer.DataType dtypeConv(DataType dataType) {
        val x = dataType.getNumber();

        return dtypeConv(x);
    };


    public static org.nd4j.linalg.api.buffer.DataType dtypeConv(int dataType) {
        switch (dataType) {
            case DataType.DT_FLOAT_VALUE: return org.nd4j.linalg.api.buffer.DataType.FLOAT;
            case DataType.DT_DOUBLE_VALUE: return org.nd4j.linalg.api.buffer.DataType.DOUBLE;
            case DataType.DT_INT32_VALUE: return org.nd4j.linalg.api.buffer.DataType.INT;
            case DataType.DT_UINT8_VALUE: return org.nd4j.linalg.api.buffer.DataType.UBYTE;
            case DataType.DT_INT16_VALUE: return org.nd4j.linalg.api.buffer.DataType.SHORT;
            case DataType.DT_INT8_VALUE: return org.nd4j.linalg.api.buffer.DataType.BYTE;
            case DataType.DT_STRING_VALUE: return org.nd4j.linalg.api.buffer.DataType.UTF8;
            case DataType.DT_INT64_VALUE: return org.nd4j.linalg.api.buffer.DataType.LONG;
            case DataType.DT_BOOL_VALUE: return org.nd4j.linalg.api.buffer.DataType.BOOL;
            case DataType.DT_UINT16_VALUE: return org.nd4j.linalg.api.buffer.DataType.UINT16;
            case DataType.DT_HALF_VALUE: return org.nd4j.linalg.api.buffer.DataType.HALF;
            case DataType.DT_UINT32_VALUE: return org.nd4j.linalg.api.buffer.DataType.UINT32;
            case DataType.DT_UINT64_VALUE: return org.nd4j.linalg.api.buffer.DataType.UINT64;
            default: throw new UnsupportedOperationException("DataType isn't supported: " + dataType + " - " + DataType.forNumber(dataType));
        }
    };
}
