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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Mainly meant for internal use:
 * represents all the elements of a dimension
 *
 * @author Adam Gibson
 */
@Slf4j
public class NDArrayIndexAll extends IntervalIndex {

    public NDArrayIndexAll() {
        super(true, 1);
    }


    @Override
    public void init(INDArray arr, long begin, int dimension) {
       //this may happen in cases where init is called too early
        //and we have something like new axis dimensions specified
        if(dimension >= arr.rank() || dimension < 0)
            return;
        initialized = true;
        inclusive = false;
        this.begin = 0;

        this.end = arr.size(dimension);
        this.length = (end - begin) / stride + 1;
    }

    @Override
    public INDArrayIndex dup() {
        NDArrayIndexAll all = new NDArrayIndexAll();
        all.inclusive = this.inclusive;
        all.begin = this.begin;
        all.end = this.begin;
        all.initialized = this.initialized;
        all.index = this.index;
        all.length = this.length;
        all.stride = this.stride;
        return all;
    }

    @Override
    public String toString(){
        return "all()";
    }

}
