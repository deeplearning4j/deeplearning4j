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

package org.nd4j.linalg.jcublas;

import lombok.val;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.jita.constant.ProtectedCudaShapeInfoProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.ndarray.ShapeInfoProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author raver119@gmail.com
 */
public class CachedShapeInfoProvider extends BaseShapeInfoProvider {
    private static Logger logger = LoggerFactory.getLogger(CachedShapeInfoProvider.class);

    protected ShapeInfoProvider provider = ProtectedCudaShapeInfoProvider.getInstance();

    public CachedShapeInfoProvider() {

    }

    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long elementWiseStride, char order, DataType type) {
        return provider.createShapeInformation(shape, stride, elementWiseStride, order, type);
    }


    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long elementWiseStride, char order, long extras) {
        return provider.createShapeInformation(shape, stride, elementWiseStride, order, extras);
    }

    /**
     * This method forces cache purge, if cache is available for specific implementation
     */
    @Override
    public void purgeCache() {
        provider.purgeCache();
    }
}
