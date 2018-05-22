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

package org.datavec.api.transform.transform.integer;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Replace an empty/missing integer with a certain value.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class ReplaceEmptyIntegerWithValueTransform extends BaseIntegerTransform {

    private final int value;

    public ReplaceEmptyIntegerWithValueTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("value") int value) {
        super(columnName);
        this.value = value;
    }

    @Override
    public Writable map(Writable writable) {
        String s = writable.toString();
        if (s == null || s.isEmpty())
            return new IntWritable(value);
        return writable;
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
        return value;
    }
}
