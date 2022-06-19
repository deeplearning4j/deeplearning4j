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

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.concurrent.atomic.AtomicLong;
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
@EqualsAndHashCode
public class SDValue implements IDependeeGroup<INDArray> {

    private SDValueType sdValueType;
    private INDArray tensorValue;
    private Map<String, INDArray> dictValue;
    private List<INDArray> listValue;
    private static final AtomicLong counter = new AtomicLong(0);
    protected transient long id = counter.getAndIncrement();

    private SDValue() {
    }

    public long getId() {
        return id;
    }

    public Collection<INDArray> getCollection() {
        return getListValue();
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

    @Override
    public boolean equals(Object o) {
        SDValue sd = (SDValue) o;
        return sd.getId() == this.getId();
    }

    @Override
    public int hashCode() {
        return Long.hashCode(this.getId());
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
