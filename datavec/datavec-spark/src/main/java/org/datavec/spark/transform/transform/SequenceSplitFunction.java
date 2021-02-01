/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.spark.transform.transform;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.api.writable.Writable;

import java.util.Iterator;
import java.util.List;

/**
 * Created by Alex on 17/03/2016.
 */
public class SequenceSplitFunction implements FlatMapFunction<List<List<Writable>>, List<List<Writable>>> {

    private final SequenceSplit split;

    public SequenceSplitFunction(SequenceSplit split) {
        this.split = split;
    }

    @Override
    public Iterator<List<List<Writable>>> call(List<List<Writable>> collections) throws Exception {
        return split.split(collections).iterator();
    }

}
