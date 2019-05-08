/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.indexing;

import com.google.common.primitives.Longs;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class PointIndex implements INDArrayIndex {
    private long point;

    /**
     *
     * @param point
     */
    public PointIndex(long point) {
        this.point = point;
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

    }

    @Override
    public void init(INDArray arr, int dimension) {

    }

    @Override
    public void init(long begin, long end, long max) {

    }

    @Override
    public void init(long begin, long end) {

    }

    @Override
    public String toString(){
        return "Point(" + point + ")";
    }
}
