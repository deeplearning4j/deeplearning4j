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

package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Iterator;

/**
 * @author Christian Weilbach
 */
public class FirstAxisIterator implements Iterator<Object> {
    private INDArray iterateOver;
    private int i = 0;


    /**
     *
     * @param iterateOver
     */
    public FirstAxisIterator(INDArray iterateOver) {
        this.iterateOver = iterateOver;
    }

    @Override
    public boolean hasNext() {
        return i < iterateOver.slices();
    }

    @Override
    public void remove() {

    }

    @Override
    public Object next() {
        INDArray s = iterateOver.slice(i++);
        if (s.isScalar()) {
            return s.getDouble(0);
        } else {
            return s;
        }
    }

}
