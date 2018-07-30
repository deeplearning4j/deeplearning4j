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

package org.nd4j.weightinit;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Abstract class for {@link WeightInitScheme}
 * This handles boilerplate like delegating to the parameters view.
 *
 *
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public abstract class BaseWeightInitScheme implements WeightInitScheme {
    private char order;

    /**
     * Initialize with c weight ordering by default
     */
    public BaseWeightInitScheme() {
        this('c');
    }

    public BaseWeightInitScheme(char order) {
        this.order = order;
    }

    public abstract INDArray doCreate(long[] shape, INDArray paramsView);

    @Override
    public INDArray create(long[] shape, INDArray paramsView) {
        return handleParamsView(doCreate(shape,paramsView),paramsView);
    }

    @Override
    public INDArray create(long... shape) {
        INDArray ret = doCreate(shape,null);
        return ret;
    }

    @Override
    public char order() {
        return order;
    }

    protected INDArray handleParamsView(INDArray outputArray, INDArray paramView) {
        //minor optimization when the views are the same, just return
        if(paramView == null || paramView == outputArray)
            return outputArray;
        INDArray flat = Nd4j.toFlattened(order(), outputArray);
        if (flat.length() != paramView.length())
            throw new RuntimeException("ParamView length does not match initialized weights length (view length: "
                    + paramView.length() + ", view shape: " + Arrays.toString(paramView.shape())
                    + "; flattened length: " + flat.length());

        paramView.assign(flat);

        return paramView.reshape(order(), outputArray.shape());
    }


}
