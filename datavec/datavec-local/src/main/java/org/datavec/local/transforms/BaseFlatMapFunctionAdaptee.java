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

package org.datavec.local.transforms;


import org.datavec.local.transforms.functions.FlatMapFunctionAdapter;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.List;

/**
 *
 * This class should be used instead of direct referral to FlatMapFunction
 *
 */
public class BaseFlatMapFunctionAdaptee<K, V>  {

    protected final FlatMapFunctionAdapter<K, V> adapter;

    public BaseFlatMapFunctionAdaptee(FlatMapFunctionAdapter<K, V> adapter) {
        this.adapter = adapter;
    }

    public List<V> call(K k)  {
        try {
            return adapter.call(k);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

    }
}
