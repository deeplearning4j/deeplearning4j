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

package org.deeplearning4j.nn.conf.constraint;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * Constrain the weights to be non-negative
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class NonNegativeConstraint extends BaseConstraint {

    public NonNegativeConstraint(){ }

    @Override
    public void apply(INDArray param) {
        BooleanIndexing.replaceWhere(param, 0.0, Conditions.lessThan(0.0));
    }

    @Override
    public NonNegativeConstraint clone() { return new NonNegativeConstraint();}

}
