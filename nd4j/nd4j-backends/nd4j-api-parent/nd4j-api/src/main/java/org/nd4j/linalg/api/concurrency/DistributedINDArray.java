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

package org.nd4j.linalg.api.concurrency;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

/**
 * This interface describe holder for INDArray which persists in this or that way on multiple computational devices, or on the same device but with different values
 *
 * @author raver119@gmail.com
 */
public interface DistributedINDArray {

    /**
     * This method returns ArrayType for this instance
     * @return
     */
    ArrayType getINDArrayType();

    /**
     * This method returns INDArray for specific entry (i.e. for specific device, if you put entries that way)
     *
     * @param entry
     * @return
     */
    INDArray entry(int entry);

    /**
     * This method returns INDArray for the current device
     *
     * PLEASE NOTE: if you use more than one thread per device you'd better not use this method unless you're 100% sure
     * @return
     */
    INDArray entry();

    /**
     * This method propagates given INDArray to all entries as is
     *
     * @param array
     */
    void propagate(INDArray array);

    /**
     * This method returns total number of entries within this DistributedINDArray instance
     * @return
     */
    int numEntries();

    /**
     * This method returns number of activated entries
     * @return
     */
    int numActiveEntries();

    /**
     * This method allocates INDArray for specified entry
     *
     * @param entry
     * @param shapeDescriptor
     */
    void allocate(int entry, LongShapeDescriptor shapeDescriptor);

    /**
     * This method allocates INDArray for specified entry
     *
     * @param entry
     */
    void allocate(int entry, DataType dataType, long... shape);
}
