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

package org.datavec.api.transform.transform.string;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * Replaces String values that match regular expressions.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class ReplaceStringTransform extends BaseStringTransform {

    private final Map<String, String> map;

    /**
     * Constructs a new ReplaceStringTransform using the specified
     * @param columnName Name of the column
     * @param map Key: regular expression; Value: replacement value
     */
    public ReplaceStringTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("map") Map<String, String> map) {
        super(columnName);
        this.map = map;
    }

    @Override
    public Text map(final Writable writable) {
        String value = writable.toString();
        value = replaceAll(value);
        return new Text(value);
    }

    @Override
    public Object map(final Object o) {
        String value = o.toString();
        value = replaceAll(value);
        return value;
    }

    private String replaceAll(String value) {
        if (map != null && !map.isEmpty()) {
            for (Map.Entry<String, String> entry : map.entrySet()) {
                value = value.replaceAll(entry.getKey(), entry.getValue());
            }
        }
        return value;
    }

}
