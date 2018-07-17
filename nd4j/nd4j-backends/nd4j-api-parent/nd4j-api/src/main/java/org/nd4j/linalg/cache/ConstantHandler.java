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

package org.nd4j.linalg.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * This interface describes
 * memory reuse strategy
 * for java-originated arrays.
 *
 * @author raver119@gmail.com
 */
public interface ConstantHandler {

    /**
     * If specific hardware supports dedicated constant memory,
     * this method forces DataBuffer passed in to be moved
     * to that constant memory.
     *
     * PLEASE NOTE: This method implementation is hardware-dependant.
     *
     * @param dataBuffer
     * @return
     */
    long moveToConstantSpace(DataBuffer dataBuffer);

    /**
     *
     * PLEASE NOTE: This method implementation is hardware-dependant.
     * PLEASE NOTE: This method does NOT allow concurrent use of any array
     *
     * @param dataBuffer
     * @return
     */
    DataBuffer relocateConstantSpace(DataBuffer dataBuffer);

    /**
     * This method returns DataBuffer with
     * constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that
     * you'll never ever change values
     * within result DataBuffer
     *
     * @param array
     * @return
     */
    DataBuffer getConstantBuffer(int[] array);

    /**
     * This method returns DataBuffer with
     * constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that
     * you'll never ever change values
     * within result DataBuffer
     *
     * @param array
     * @return
     */
    DataBuffer getConstantBuffer(long[] array);

    /**
     * This method returns DataBuffer
     * with constant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll
     * never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    DataBuffer getConstantBuffer(float[] array);

    /**
     * This method returns DataBuffer with contant equal to input array.
     *
     * PLEASE NOTE: This method assumes that you'll never ever change values within result DataBuffer
     *
     * @param array
     * @return
     */
    DataBuffer getConstantBuffer(double[] array);

    /**
     * This method removes all cached constants
     */
    void purgeConstants();

    /**
     * This method returns memory used for cache, in bytes
     *
     * @return
     */
    long getCachedBytes();
}
