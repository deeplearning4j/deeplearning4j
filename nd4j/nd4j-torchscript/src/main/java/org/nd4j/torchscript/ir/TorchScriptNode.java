/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.torchscript.ir;

import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a single operation/node in a TorchScript computation graph.
 */
@Data
@Builder
public class TorchScriptNode {

    /**
     * Unique name/identifier for this node
     */
    private String name;

    /**
     * Operation type (e.g., "aten::conv2d", "aten::batch_norm", "aten::relu")
     */
    private String opType;

    /**
     * Names of input values
     */
    @Builder.Default
    private List<String> inputNames = new ArrayList<>();

    /**
     * Names of output values
     */
    @Builder.Default
    private List<String> outputNames = new ArrayList<>();

    /**
     * Operation attributes (kernel size, stride, padding, etc.)
     */
    @Builder.Default
    private Map<String, Object> attributes = new HashMap<>();

    /**
     * Scope/namespace (e.g., "layer1.0.conv1")
     */
    private String scope;

    /**
     * Get the schema name (e.g., "aten::conv2d" -> "conv2d").
     *
     * @return the base operation name without namespace
     */
    public String getSchemaName() {
        if (opType == null) {
            return null;
        }
        int colonIndex = opType.lastIndexOf("::");
        if (colonIndex >= 0) {
            return opType.substring(colonIndex + 2);
        }
        return opType;
    }

    /**
     * Get the namespace (e.g., "aten::conv2d" -> "aten").
     *
     * @return the namespace or null
     */
    public String getNamespace() {
        if (opType == null) {
            return null;
        }
        int colonIndex = opType.indexOf("::");
        if (colonIndex >= 0) {
            return opType.substring(0, colonIndex);
        }
        return null;
    }

    /**
     * Check if this is an ATen operation.
     *
     * @return true if this is an aten:: operation
     */
    public boolean isAtenOp() {
        return opType != null && opType.startsWith("aten::");
    }

    /**
     * Check if this is a prim:: operation (primitive operations).
     *
     * @return true if this is a prim:: operation
     */
    public boolean isPrimOp() {
        return opType != null && opType.startsWith("prim::");
    }

    /**
     * Get an attribute value.
     *
     * @param name attribute name
     * @return attribute value or null
     */
    public Object getAttribute(String name) {
        return attributes.get(name);
    }

    /**
     * Get an attribute as a specific type.
     *
     * @param name attribute name
     * @param type expected type
     * @param <T>  type parameter
     * @return attribute value or null
     */
    @SuppressWarnings("unchecked")
    public <T> T getAttribute(String name, Class<T> type) {
        Object value = attributes.get(name);
        if (value != null && type.isInstance(value)) {
            return (T) value;
        }
        return null;
    }

    /**
     * Get an integer attribute.
     *
     * @param name         attribute name
     * @param defaultValue default if not found
     * @return attribute value or default
     */
    public int getIntAttribute(String name, int defaultValue) {
        Object value = attributes.get(name);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return defaultValue;
    }

    /**
     * Get a long attribute.
     *
     * @param name         attribute name
     * @param defaultValue default if not found
     * @return attribute value or default
     */
    public long getLongAttribute(String name, long defaultValue) {
        Object value = attributes.get(name);
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        return defaultValue;
    }

    /**
     * Get a float attribute.
     *
     * @param name         attribute name
     * @param defaultValue default if not found
     * @return attribute value or default
     */
    public float getFloatAttribute(String name, float defaultValue) {
        Object value = attributes.get(name);
        if (value instanceof Number) {
            return ((Number) value).floatValue();
        }
        return defaultValue;
    }

    /**
     * Get a boolean attribute.
     *
     * @param name         attribute name
     * @param defaultValue default if not found
     * @return attribute value or default
     */
    public boolean getBoolAttribute(String name, boolean defaultValue) {
        Object value = attributes.get(name);
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        return defaultValue;
    }

    /**
     * Get an integer list attribute (e.g., kernel_size, stride).
     *
     * @param name attribute name
     * @return list of integers or empty list
     */
    @SuppressWarnings("unchecked")
    public List<Integer> getIntListAttribute(String name) {
        Object value = attributes.get(name);
        if (value instanceof List) {
            List<?> list = (List<?>) value;
            List<Integer> result = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Number) {
                    result.add(((Number) item).intValue());
                }
            }
            return result;
        }
        return new ArrayList<>();
    }

    /**
     * Check if an attribute exists.
     *
     * @param name attribute name
     * @return true if attribute exists
     */
    public boolean hasAttribute(String name) {
        return attributes.containsKey(name);
    }

    /**
     * Get the number of inputs.
     *
     * @return number of inputs
     */
    public int getNumInputs() {
        return inputNames.size();
    }

    /**
     * Get the number of outputs.
     *
     * @return number of outputs
     */
    public int getNumOutputs() {
        return outputNames.size();
    }

    /**
     * Get the first output name (most common case).
     *
     * @return first output name or null
     */
    public String getOutputName() {
        return outputNames.isEmpty() ? null : outputNames.get(0);
    }
}
