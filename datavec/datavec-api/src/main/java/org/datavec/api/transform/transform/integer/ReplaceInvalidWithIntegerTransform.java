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

package org.datavec.api.transform.transform.integer;

import lombok.Data;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Replace an invalid (non-integer) value in a column with a specified integer
 */
@Data
public class ReplaceInvalidWithIntegerTransform extends BaseIntegerTransform {

    private final int value;

    public ReplaceInvalidWithIntegerTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("value") int value) {
        super(columnName);
        this.value = value;
    }

    @Override
    public Writable map(Writable writable) {
        if (inputSchema.getMetaData(columnNumber).isValid(writable)) {
            return writable;
        } else {
            return new IntWritable(value);
        }
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
        Number n = (Number) input;
        if (inputSchema.getMetaData(columnNumber).isValid(new IntWritable(n.intValue()))) {
            return input;
        } else {
            return value;
        }
    }
}
