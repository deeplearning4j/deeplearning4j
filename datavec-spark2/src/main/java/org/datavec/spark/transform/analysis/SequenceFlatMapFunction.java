/*
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.spark.transform.analysis;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.writable.Writable;

import java.util.Iterator;
import java.util.List;

/**
 * SequenceFlatMapFunction: very simple function used to flatten a sequence
 * Typically used only internally for certain analysis operations
 *
 * @author Alex Black
 */
public class SequenceFlatMapFunction implements FlatMapFunction<List<List<Writable>>, List<Writable>> {

    private final SequenceFlatMapFunctionAdapter adapter = new SequenceFlatMapFunctionAdapter();

    @Override
    public Iterator<List<Writable>> call(List<List<Writable>> collections) throws Exception {
        return adapter.call(collections).iterator();
    }

}
