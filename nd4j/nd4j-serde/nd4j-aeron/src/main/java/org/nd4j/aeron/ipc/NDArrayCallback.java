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

package org.nd4j.aeron.ipc;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An ndarray listener
 * @author Adam Gibson
 */
public interface NDArrayCallback {


    /**
     * A listener for ndarray message
     * @param message the message for the callback
     */
    void onNDArrayMessage(NDArrayMessage message);

    /**
     * Used for partial updates using tensor along
     * dimension
     * @param arr the array to count as an update
     * @param idx the index for the tensor along dimension
     * @param dimensions the dimensions to act on for the tensor along dimension
     */
    void onNDArrayPartial(INDArray arr, long idx, int... dimensions);

    /**
     * Setup an ndarray
     * @param arr
     */
    void onNDArray(INDArray arr);

}
