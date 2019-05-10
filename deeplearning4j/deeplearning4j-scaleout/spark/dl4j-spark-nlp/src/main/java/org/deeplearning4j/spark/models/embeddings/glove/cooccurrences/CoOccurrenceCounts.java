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

package org.deeplearning4j.spark.models.embeddings.glove.cooccurrences;

import org.apache.spark.api.java.function.Function2;
import org.nd4j.linalg.primitives.CounterMap;


/**
 * Co occurrence count reduction
 * @author Adam Gibson
 */
public class CoOccurrenceCounts implements
                Function2<CounterMap<String, String>, CounterMap<String, String>, CounterMap<String, String>> {


    @Override
    public CounterMap<String, String> call(CounterMap<String, String> v1, CounterMap<String, String> v2)
                    throws Exception {
        v1.incrementAll(v2);
        return v1;
    }
}
