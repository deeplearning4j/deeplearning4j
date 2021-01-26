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

package org.deeplearning4j.spark.impl.common.repartition;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * This is a simple function used to convert a {@code JavaRDD<Tuple2<T,U>>} to a {@code JavaPairRDD<T,U>} via a
 * {JavaRDD.mappartitionsToPair()} call.
 *
 * @author Alex Black
 */
public class MapTupleToPairFlatMap<T, U> implements PairFlatMapFunction<Iterator<Tuple2<T, U>>, T, U> {

    @Override
    public Iterator<Tuple2<T, U>> call(Iterator<Tuple2<T, U>> tuple2Iterator) throws Exception {
        List<Tuple2<T, U>> list = new ArrayList<>();
        while (tuple2Iterator.hasNext()) {
            list.add(tuple2Iterator.next());
        }
        return list.iterator();
    }
}
