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
 * @author Adam Gibson
 */ //static opType checking used for checking if an index should be represented as all
public class NDArrayIndexEmpty implements INDArrayIndex {
    @Override
    public long end() {
        return 0;
    }

    @Override
    public long offset() {
        return 0;
    }

    @Override
    public long length() {
        return 0;
    }

    @Override
    public long stride() {
        return 1;
    }

    @Override
    public long current() {
        return 0;
    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public long next() {
        return 0;
    }


    @Override
    public void reverse() {

    }

    @Override
    public boolean isInterval() {
        return false;
    }

    @Override
    public void setInterval(boolean isInterval) {

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
    public void reset() {

    }
}
