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

package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
public interface ShapeInfoProvider {
    /**
     * This method creates long shapeInformation buffer, based on shape being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(long[] shape, DataBuffer.Type dataType);

    /**
     * This method creates long shapeInformation buffer, based on shape & order being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(long[] shape, char order, DataBuffer.Type dataType);

    /**
     * This method creates long shapeInformation buffer, based on detailed shape info being passed in
     * @param shape
     * @return
     */
    Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long elementWiseStride, char order, DataBuffer.Type dataType);



    /**
     * This method forces cache purge, if cache is available for specific implementation
     */
    void purgeCache();

    /**
     * This method returns memory used for cache, in bytes
     * @return
     */
    long getCachedBytes();
}
