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

package org.nd4j.linalg.indexing;

import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * A point index is used for pulling something like a specific row from
 * an array. A view will be created based on the point at the given dimension.
 *
 * Negative indices can also be specified allowing for dynamic
 * resolution of dimensions/coordinates at runtime.
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
@Slf4j
public class  PointIndex implements INDArrayIndex {
    private long point;
    private boolean initialized = false;


    private PointIndex(){}

    /**
     *
     * @param point
     */
    public PointIndex(long point) {
        this.point = point;
        initialized = point > 0;
    }

    @Override
    public long end() {
        return point;
    }

    @Override
    public long offset() {
        return point;
    }

    @Override
    public long length() {
        return 1;
    }

    @Override
    public long stride() {
        return 1;
    }

    @Override
    public void reverse() {

    }

    @Override
    public boolean isInterval() {
        return false;
    }

    @Override
    public void init(INDArray arr, long begin, int dimension) {
        if(begin < 0) {
            begin += arr.size(dimension);
            point = begin;
        } else {
            point = begin;
        }
    }

    @Override
    public void init(INDArray arr, int dimension) {
        point = arr.size(dimension);
    }

    @Override
    public void init(long begin, long end, long max) {
        if(begin < 0) {
            initialized = false;
            log.debug("Not initializing due to missing positive dimensions. Initialization will be attempted again during runtime.");
            return;
        }

        point = begin;
        initialized = true;
    }

    @Override
    public void init(long begin, long end) {
        if(begin < 0) {
            initialized = false;
            log.debug("Not initializing due to missing positive dimensions. Initialization will be attempted again during runtime.");
            return;
        }

        point = begin;
        initialized = true;
    }

    @Override
    public boolean initialized() {
        return initialized && point >= 0;
    }

    @Override
    public INDArrayIndex dup() {
        PointIndex pointIndex = new PointIndex();
        pointIndex.initialized = initialized;
        pointIndex.point = point;
        return pointIndex;
    }

    @Override
    public String toString(){
        return "Point(" + point + ")";
    }
}
