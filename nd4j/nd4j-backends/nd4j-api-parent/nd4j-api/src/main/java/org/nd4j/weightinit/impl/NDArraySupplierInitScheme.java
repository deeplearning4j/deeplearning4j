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

package org.nd4j.weightinit.impl;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.weightinit.WeightInit;
import org.nd4j.weightinit.WeightInitScheme;

/**
 *
 */
@AllArgsConstructor
public class NDArraySupplierInitScheme implements WeightInitScheme {

    private NDArraySupplier supplier;

    public NDArraySupplierInitScheme(final INDArray arr){
        this(new NDArraySupplierInitScheme.NDArraySupplier() {
            @Override
            public INDArray getArr() {
                return arr;
            }
        });
    }

    /**
     * A simple {@link INDArray facade}
     */
    public  interface NDArraySupplier {
        /**
         * An array proxy method.
          * @return
         */
        INDArray getArr();
    }

    @Override
    public INDArray create(long[] shape, INDArray paramsView) {
        return supplier.getArr();
    }

    @Override
    public INDArray create(DataType dataType, long[] shape) {
        return supplier.getArr();
    }

    @Override
    public char order() {
        return 'f';
    }

    @Override
    public WeightInit type() {
        return WeightInit.SUPPLIED;
    }
}
