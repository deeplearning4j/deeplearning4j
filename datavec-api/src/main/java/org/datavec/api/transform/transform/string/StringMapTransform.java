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

import java.util.Map;

/**
 * A simple String -> String map function.
 *
 * Keys in the map are the original values; the Values in the map are their replacements.
 * If a String appears in the data but does not appear in the provided map (as a key), that String values will
 * not be modified.
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class StringMapTransform extends BaseStringTransform {

    private final Map<String, String> map;

    /**
     *
     * @param columnName Name of the column
     * @param map Key: From. Value: To
     */
    public StringMapTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("map") Map<String, String> map) {
        super(columnName);
        this.map = map;
    }

    @Override
    public Text map(Writable writable) {
        String orig = writable.toString();
        if (map.containsKey(orig)) {
            return new Text(map.get(orig));
        }

        if (writable instanceof Text)
            return (Text) writable;
        else
            return new Text(writable.toString());
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
        String orig = input.toString();
        if (map.containsKey(orig)) {
            return map.get(orig);
        }

        if (input instanceof String)
            return input;
        else
            return orig;
    }
}
