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

import java.io.Serializable;

/**
 * A simple interface for retrieving an
 * ndarray
 *
 * @author Adam Gibson
 */
public interface NDArrayHolder extends Serializable {


    /**
     * Set the ndarray
     * @param arr the ndarray for this holder
     *            to use
     */
    void setArray(INDArray arr);


    /**
     * The number of updates
     * that have been sent to this older.
     * @return
     */
    int totalUpdates();

    /**
     * Retrieve an ndarray
     * @return
     */
    INDArray get();

    /**
     * Retrieve a partial view of the ndarray.
     * This method uses tensor along dimension internally
     * Note this will call dup()
     * @param idx the index of the tad to get
     * @param dimensions the dimensions to use
     * @return the tensor along dimension based on the index and dimensions
     * from the master array.
     */
    INDArray getTad(int idx, int... dimensions);
}
