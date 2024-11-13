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
package org.nd4j.nativeblas;

import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;

public class OpaqueNDArrayArr extends PointerPointer<OpaqueNDArray> {


    public OpaqueNDArrayArr(OpaqueNDArray... array) { super(array); }


    /**
     * @see {@link #createFrom(INDArray...)}
     * @param array
     * @return
     */
    public static OpaqueNDArrayArr createFrom(List<INDArray> array) {
        OpaqueNDArray[] inputs = array.stream()
                .map(OpaqueNDArray::fromINDArray).toArray(OpaqueNDArray[]::new);
        OpaqueNDArrayArr inputsOpaque = (OpaqueNDArrayArr) new OpaqueNDArrayArr().capacity(inputs.length);
        inputsOpaque.put(inputs);
        return inputsOpaque;
    }


    /**
     * Simple creation method
     * that handles ensuring a proper 
     * instantiation of the array with capacity
     * and putting the result pointers in the array.
     * @param array the array to create the OpaqueNDArrayArr from
     * @return
     */
    public static OpaqueNDArrayArr createFrom(INDArray... array) {
        OpaqueNDArray[] inputs = Arrays.stream(array)
                .map(OpaqueNDArray::fromINDArray).toArray(OpaqueNDArray[]::new);
        OpaqueNDArrayArr inputsOpaque = (OpaqueNDArrayArr) new OpaqueNDArrayArr().capacity(inputs.length);
        inputsOpaque.put(inputs);
        return inputsOpaque;
    }

}
