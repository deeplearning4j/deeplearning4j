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

package org.nd4j.linalg.jcublas;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.common.primitives.Pair;
import org.nd4j.jita.constant.ProtectedCachedShapeInfoProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.ndarray.ShapeInfoProvider;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author raver119@gmail.com
 */
public class CachedShapeInfoProvider extends BaseShapeInfoProvider {
    private static Logger logger = LoggerFactory.getLogger(CachedShapeInfoProvider.class);

    protected ShapeInfoProvider provider = ProtectedCachedShapeInfoProvider.getInstance();

    public CachedShapeInfoProvider() {

    }

    /**
     * Override to strip ARRAY_EMPTY for rank-0 scalars before delegating to the provider.
     * BaseShapeInfoProvider.createShapeInformation(long[] shapeInfo) wraps the raw bytes without
     * stripping the EMPTY flag, which causes rank-0 scalars with valid data to be treated as empty.
     * ProtectedCachedShapeInfoProvider correctly strips the flag for rank-0, so we replicate that
     * logic here and then route through the same provider that handles the cached path.
     */
    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shapeInfo) {
        long[] shape = Shape.shape(shapeInfo);
        long[] stride = Shape.stride(shapeInfo);
        long ews = Shape.elementWiseStride(shapeInfo);
        char order = Shape.order(shapeInfo);
        long extras = Shape.extras(shapeInfo);
        if (shape.length == 0) {
            // Rank-0 scalar: strip ARRAY_EMPTY — a scalar always has exactly 1 element
            extras = extras & ~ArrayOptionsHelper.ATYPE_EMPTY_BIT;
        }
        return provider.createShapeInformation(shape, stride, ews, order, extras);
    }

    @Override
    public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride, long elementWiseStride, char order, DataType type, boolean empty) {
        return provider.createShapeInformation(shape, stride, elementWiseStride, order, type, empty);
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
