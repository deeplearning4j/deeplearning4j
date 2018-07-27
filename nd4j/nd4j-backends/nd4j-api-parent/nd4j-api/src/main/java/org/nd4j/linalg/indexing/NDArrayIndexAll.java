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

package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Mainly meant for internal use:
 * represents all of the elements of a dimension
 *
 * @author Adam Gibson
 */
public class NDArrayIndexAll extends IntervalIndex {

    /**
     * @param inclusive whether to include the last number
     */
    public NDArrayIndexAll(boolean inclusive) {
        super(inclusive, 1);
    }


    @Override
    public void init(INDArray arr, long begin, int dimension) {
        this.begin = 0;
        this.end = arr.size(dimension);
        this.length = (Math.abs(end - begin));
    }



}
