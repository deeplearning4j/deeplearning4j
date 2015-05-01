/*
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

package org.deeplearning4j.spark.models.embeddings.common;

import com.hazelcast.core.IFunction;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 3/30/15.
 */
public class GetVector extends BaseWord2VecFunction implements IFunction<InMemoryLookupTable,INDArray> {

    public GetVector(int index,String from) {
        this(index,0,from,null);
    }

    public GetVector(int fromIndex, int toIndex, String from, String to) {
        super(fromIndex, toIndex, from, to);
    }

    @Override
    public INDArray apply(InMemoryLookupTable input) {
        return getFrom(input);
    }
}
