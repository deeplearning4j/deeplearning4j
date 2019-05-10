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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * This method maps all String values, except those is the specified list, to a single String  value
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode
public class MapAllStringsExceptListTransform extends BaseStringTransform {

    private final Set<String> exceptions;
    private final String newValue;

    public MapAllStringsExceptListTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("newValue") String newValue, @JsonProperty("exceptions") List<String> exceptions) {
        super(columnName);
        this.newValue = newValue;
        this.exceptions = new HashSet<>(exceptions);
    }

    @Override
    public Text map(Writable writable) {
        String str = writable.toString();
        if (exceptions.contains(str)) {
            return new Text(str);
        } else {
            return new Text(newValue);
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
        String str = input.toString();
        if (exceptions.contains(str)) {
            return str;
        } else {
            return newValue;
        }
    }
}
