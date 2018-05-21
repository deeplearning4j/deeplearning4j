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
import lombok.EqualsAndHashCode;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * String transform that removes all whitespace charaters
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class RemoveWhiteSpaceTransform extends BaseStringTransform {


    public RemoveWhiteSpaceTransform(@JsonProperty("columnName") String columnName) {
        super(columnName);
    }

    @Override
    public Text map(Writable writable) {
        String value = writable.toString().replaceAll("\\s", "");
        return new Text(value);
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
        String value = input.toString().replaceAll("\\s", "");
        return value;
    }
}
