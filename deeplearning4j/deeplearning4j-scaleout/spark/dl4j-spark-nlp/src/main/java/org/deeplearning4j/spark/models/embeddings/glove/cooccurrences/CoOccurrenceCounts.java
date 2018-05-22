/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

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
