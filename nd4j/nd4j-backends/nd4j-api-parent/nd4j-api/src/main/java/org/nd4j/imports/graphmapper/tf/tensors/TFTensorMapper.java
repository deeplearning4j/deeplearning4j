/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.imports.graphmapper.tf.tensors;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.Buffer;
import java.nio.ByteBuffer;

/**
 * @param <J> Java array type
 * @param <B> Java buffer type
 */
public interface TFTensorMapper<J,B extends Buffer> {

    enum ValueSource {EMPTY, VALUE_COUNT, BINARY};

    DataType dataType();

    long[] shape();

    boolean isEmpty();

    ValueSource valueSource();

    int valueCount();

    J newArray(int length);

    B getBuffer(ByteBuffer bb);

    INDArray toNDArray();

    void getValue(J jArr, int i);

    void getValue(J jArr, B buffer, int i);

    INDArray arrayFor(long[] shape, J jArr);


}
