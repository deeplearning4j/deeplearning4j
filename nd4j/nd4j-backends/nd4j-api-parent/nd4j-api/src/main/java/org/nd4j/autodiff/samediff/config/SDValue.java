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

package org.nd4j.autodiff.samediff.config;

import lombok.Getter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.*;
import org.nd4j.autodiff.samediff.internal.IDependeeGroup;

/**
 * An SDValue represents a value that can be passed in
 * and returned from a {@link org.nd4j.autodiff.samediff.SameDiff}
 * graph for execution.
 *
 * @author Adam Gibson
 */
@Getter
public class SDValue implements IDependeeGroup<INDArray> {

    private SDValueType sdValueType;
    private INDArray tensorValue;
    private Map<String, INDArray> dictValue;
    private List<INDArray> listValue;

    private SDValue() {
    }

    public void setCloseable(boolean closeable) {
        if(tensorValue != null) {
            tensorValue.setCloseable(closeable);
        }

        if(listValue != null) {
            listValue.stream()
                    .filter(arr -> arr != null)
                    .forEach(arr -> arr.setCloseable(closeable));
        }

        if(dictValue != null) {
            dictValue.values().forEach(arr -> arr.setCloseable(closeable));
        }
    }

    /**
     * Returns an ID based on the underlying array identity, NOT a unique SDValue ID.
     * This is critical for DependencyMap to correctly track dependencies when the same
     * array is wrapped in different SDValue instances. Two SDValues wrapping the same
     * array(s) will return the same ID.
     */
    public long getId() {
        // Return ID based on array identity to match equals/hashCode semantics
        if (tensorValue != null) {
            return tensorValue.getId();
        } else if (listValue != null && !listValue.isEmpty()) {
            // For lists, combine array IDs - use first non-null array's ID as base
            for (INDArray arr : listValue) {
                if (arr != null) {
                    return arr.getId();
                }
            }
        } else if (dictValue != null && !dictValue.isEmpty()) {
            // For dicts, use first non-null array's ID as base
            for (INDArray arr : dictValue.values()) {
                if (arr != null) {
                    return arr.getId();
                }
            }
        }
        // Fallback for empty values - use hashCode which is also based on content
        return hashCode();
    }

    public Collection<INDArray> getCollection() {
        if (tensorValue != null)
            return Arrays.asList(tensorValue);
        if (listValue != null)
            return listValue;
        if (dictValue != null)
            return dictValue.values();
        return Collections.emptyList();
    }

    /**
     * Create an empty value for the given
     * {@link DataType}
     *
     * @param valueType the value type to create {@link SDValue} for
     * @param dataType  the data type of the empty value
     * @return an empty ({@link Nd4j#empty(DataType)} for {@link SDValueType#TENSOR}
     *         or an empty list or map for the other associated types
     */
    public static SDValue empty(SDValueType valueType, DataType dataType) {
        switch (valueType) {
            case LIST:
                return SDValue.create(Arrays.asList());
            case DICT:
                return SDValue.create(Collections.emptyMap());
            case TENSOR:
                return SDValue.create(Nd4j.zeros(1).castTo(dataType));
            default:
                throw new IllegalArgumentException("Unable to create empty value, unknown value type " + valueType);
        }
    }

    /**
     * Return an {@link INDArray}
     * if the value type is {@link SDValueType#LIST}
     * and the number of elements is 1 otherwise
     * return the {@link #tensorValue}
     *
     * @return
     */
    public INDArray getTensorValue() {
        if (listValue != null && listValue.size() == 1)
            return listValue.get(0);
        return tensorValue;
    }

    /**
     * Return an {@link INDArray[]}
     * if the value type is {@link SDValueType#TENSOR}
     * else return the list type
     *
     * @return
     */
    public List<INDArray> getListValue() {
        if (tensorValue != null)
            return Arrays.asList(tensorValue);
        return listValue;
    }

    /**
     * Wrap an {@link INDArray} in a tensor
     * with an {@link SDValueType#TENSOR} type
     *
     * @param inputValue the input value for the {@link SDValue}
     * @return the created value
     */
    public static SDValue create(INDArray inputValue) {
        SDValue sdValue = new SDValue();
        sdValue.tensorValue = inputValue;
        sdValue.sdValueType = SDValueType.TENSOR;
        return sdValue;
    }

    /**
     * Wrap an {@link INDArray[]} in a value
     * with an {@link SDValueType#LIST} type
     *
     * @param inputValue the input value
     * @return the created value
     */
    public static SDValue create(Collection<INDArray> inputValue) {
        SDValue sdValue = new SDValue();
        sdValue.listValue = (List<INDArray>) inputValue;
        sdValue.sdValueType = SDValueType.LIST;
        return sdValue;
    }

    /**
     * Wrap an {@link INDArray[]} in a value
     * with an {@link SDValueType#LIST} type
     *
     * @param inputValue the input value
     * @return the created value
     */
    public static SDValue create(List<INDArray> inputValue) {
        SDValue sdValue = new SDValue();
        sdValue.listValue = inputValue;
        sdValue.sdValueType = SDValueType.LIST;
        return sdValue;
    }

    /**
     * Wrap an {@link Map<String<INDArray>} in a value
     * with an {@link SDValueType#DICT} type
     *
     * @param inputValue the input value
     * @return the created value
     */
    public static SDValue create(Map<String, INDArray> inputValue) {
        SDValue sdValue = new SDValue();
        sdValue.dictValue = inputValue;
        sdValue.sdValueType = SDValueType.DICT;
        return sdValue;
    }

    /**
     * Equality is based on the underlying array identity, NOT the SDValue wrapper's ID.
     * This is critical for the dependency tracker to correctly identify when the same
     * array is wrapped in different SDValue instances. Without this, constants and
     * variables can be prematurely deallocated causing use-after-free bugs.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SDValue other = (SDValue) o;

        if (this.sdValueType != other.sdValueType) return false;

        switch (this.sdValueType) {
            case TENSOR:
                // Compare by array identity (getId()), not array content
                if (this.tensorValue == null && other.tensorValue == null) return true;
                if (this.tensorValue == null || other.tensorValue == null) return false;
                return this.tensorValue.getId() == other.tensorValue.getId();
            case LIST:
                if (this.listValue == null && other.listValue == null) return true;
                if (this.listValue == null || other.listValue == null) return false;
                if (this.listValue.size() != other.listValue.size()) return false;
                for (int i = 0; i < this.listValue.size(); i++) {
                    INDArray thisArr = this.listValue.get(i);
                    INDArray otherArr = other.listValue.get(i);
                    if (thisArr == null && otherArr == null) continue;
                    if (thisArr == null || otherArr == null) return false;
                    if (thisArr.getId() != otherArr.getId()) return false;
                }
                return true;
            case DICT:
                if (this.dictValue == null && other.dictValue == null) return true;
                if (this.dictValue == null || other.dictValue == null) return false;
                if (!this.dictValue.keySet().equals(other.dictValue.keySet())) return false;
                for (String key : this.dictValue.keySet()) {
                    INDArray thisArr = this.dictValue.get(key);
                    INDArray otherArr = other.dictValue.get(key);
                    if (thisArr == null && otherArr == null) continue;
                    if (thisArr == null || otherArr == null) return false;
                    if (thisArr.getId() != otherArr.getId()) return false;
                }
                return true;
            default:
                return false;
        }
    }

    /**
     * Hash code is based on the underlying array identity to match equals().
     * Uses array IDs to ensure consistent hashing with equals().
     */
    @Override
    public int hashCode() {
        int result = sdValueType != null ? sdValueType.hashCode() : 0;

        switch (this.sdValueType) {
            case TENSOR:
                if (tensorValue != null) {
                    result = 31 * result + Long.hashCode(tensorValue.getId());
                }
                break;
            case LIST:
                if (listValue != null) {
                    for (INDArray arr : listValue) {
                        result = 31 * result + (arr != null ? Long.hashCode(arr.getId()) : 0);
                    }
                }
                break;
            case DICT:
                if (dictValue != null) {
                    for (Map.Entry<String, INDArray> entry : dictValue.entrySet()) {
                        result = 31 * result + entry.getKey().hashCode();
                        result = 31 * result + (entry.getValue() != null ? Long.hashCode(entry.getValue().getId()) : 0);
                    }
                }
                break;
        }

        return result;
    }

    @Override
    public String toString() {
        INDArray h = this.getTensorValue();
        StringBuilder st = new StringBuilder();
        if (h != null) {
            st.append("--sdValueId-");
            st.append(this.getId() + "--key--" + this.getSdValueType() + " --Array " + h.getId());
        } else {

            List<INDArray> listx = this.getListValue();
            if (listx != null && listx.size() > 0) {
                st.append("--sdValueId-");
                st.append(this.getId() + "--key--" + this.getSdValueType() + " -- List Size " + listx.size());
                for (INDArray gh : this.getListValue()) {
                    if (gh == null) {
                        st.append(" --Array NULL ");
                    } else {
                        st.append(" --Array " + gh.getId() + " --\t ");
                    }

                }
            }
        }
        return st.toString();
    }
}
