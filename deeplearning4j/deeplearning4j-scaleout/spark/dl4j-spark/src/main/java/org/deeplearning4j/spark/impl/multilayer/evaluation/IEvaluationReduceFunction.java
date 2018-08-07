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

package org.deeplearning4j.spark.impl.multilayer.evaluation;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.eval.IEvaluation;

/**
 *
 * Reduction function for use with {@link IEvaluateFlatMapFunction} for distributed evaluation
 *
 * @author Alex Black
 */
@Slf4j
public class IEvaluationReduceFunction<T extends IEvaluation> implements Function2<T[], T[], T[]> {
    public IEvaluationReduceFunction() {}

    @Override
    public T[] call(T[] eval1, T[] eval2) throws Exception {
        //Shouldn't *usually* happen...
        if(eval1 == null){
            return eval2;
        } else if(eval2 == null){
            return eval1;
        }


        for (int i = 0; i < eval1.length; i++) {
            if(eval1[i] == null){
                eval1[i] = eval2[i];
            } else if(eval2[i] != null){
                eval1[i].merge(eval2[i]);
            }
        }
        return eval1;
    }
}
