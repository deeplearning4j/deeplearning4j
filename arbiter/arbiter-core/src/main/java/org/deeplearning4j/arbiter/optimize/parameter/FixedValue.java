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

package org.deeplearning4j.arbiter.optimize.parameter;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.serde.jackson.GenericDeserializer;
import org.deeplearning4j.arbiter.optimize.serde.jackson.GenericSerializer;
import org.deeplearning4j.arbiter.util.ObjectUtils;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * FixedValue is a ParameterSpace that defines only a single fixed value
 *
 * @param <T> Type of (fixed) value
 */
@EqualsAndHashCode
public class FixedValue<T> implements ParameterSpace<T> {
    @JsonSerialize(using = GenericSerializer.class)
    @JsonDeserialize(using = GenericDeserializer.class)
    private Object value;
    private int index;

    @JsonCreator
    public FixedValue(@JsonProperty("value") T value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return "FixedValue(" + ObjectUtils.valueToString(value) + ")";
    }

    @Override
    public T getValue(double[] input) {
        return (T) value;
    }

    @Override
    public int numParameters() {
        return 0;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.emptyList();
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        return Collections.emptyMap();
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public void setIndices(int... indices) {
        if (indices != null && indices.length != 0)
            throw new IllegalArgumentException(
                            "Invalid call: FixedValue ParameterSpace " + "should not be given an index (0 params)");
    }
}
