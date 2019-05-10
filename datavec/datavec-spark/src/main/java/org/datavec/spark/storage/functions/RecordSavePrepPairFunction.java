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

package org.datavec.spark.storage.functions;

import org.apache.hadoop.io.LongWritable;
import org.apache.spark.api.java.function.PairFunction;
import org.datavec.api.writable.Writable;
import org.datavec.hadoop.records.reader.mapfile.record.RecordWritable;
import scala.Tuple2;

import java.util.List;

/**
 * A simple function to prepare data for saving via {@link org.datavec.spark.storage.SparkStorageUtils}
 *
 * @author Alex Black
 */
public class RecordSavePrepPairFunction
                implements PairFunction<Tuple2<List<Writable>, Long>, LongWritable, RecordWritable> {
    @Override
    public Tuple2<LongWritable, RecordWritable> call(Tuple2<List<Writable>, Long> t2) throws Exception {
        return new Tuple2<>(new LongWritable(t2._2()), new RecordWritable(t2._1()));
    }
}
