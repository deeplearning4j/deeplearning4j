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

package org.nd4j.imports.descriptors.properties.adapters;

import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.tensorflow.framework.DataType;

import java.lang.reflect.Field;

public class DataTypeAdapter implements AttributeAdapter {

    @Override
    public void mapAttributeFor(Object inputAttributeValue, Field fieldFor, DifferentialFunction on) {
        on.setValueFor(fieldFor,dtypeConv((DataType) inputAttributeValue));
    }

    protected org.nd4j.linalg.api.buffer.DataType dtypeConv(DataType dataType) {
        val x = dataType.getNumber();

        switch (x) {
            case 1: return org.nd4j.linalg.api.buffer.DataType.FLOAT;
            case 2: return org.nd4j.linalg.api.buffer.DataType.DOUBLE;
            case 3: return org.nd4j.linalg.api.buffer.DataType.INT;
            case 9: return org.nd4j.linalg.api.buffer.DataType.LONG;
            case 19: return org.nd4j.linalg.api.buffer.DataType.HALF;
            default: throw new UnsupportedOperationException("DataType isn't supported: " + dataType.name());
        }
    };
}
