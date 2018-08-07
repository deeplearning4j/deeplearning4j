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

package org.datavec.spark.functions.pairdata;

import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import org.datavec.spark.util.DataVecSparkUtil;
import scala.Tuple2;
import scala.Tuple3;

/** A function to read files (assuming exactly 2 per input) from a PortableDataStream and combine the contents into a BytesPairWritable
 * @see DataVecSparkUtil#combineFilesForSequenceFile(JavaSparkContext, String, String, PathToKeyConverter, PathToKeyConverter)
 */
public class MapToBytesPairWritableFunction implements
                PairFunction<Tuple2<String, Iterable<Tuple3<String, Integer, PortableDataStream>>>, Text, BytesPairWritable> {
    @Override
    public Tuple2<Text, BytesPairWritable> call(
                    Tuple2<String, Iterable<Tuple3<String, Integer, PortableDataStream>>> in) throws Exception {
        byte[] first = null;
        byte[] second = null;
        String firstOrigPath = null;
        String secondOrigPath = null;
        Iterable<Tuple3<String, Integer, PortableDataStream>> iterable = in._2();
        for (Tuple3<String, Integer, PortableDataStream> tuple : iterable) {
            if (tuple._2() == 0) {
                first = tuple._3().toArray();
                firstOrigPath = tuple._1();
            } else if (tuple._2() == 1) {
                second = tuple._3().toArray();
                secondOrigPath = tuple._1();
            }
        }
        return new Tuple2<>(new Text(in._1()), new BytesPairWritable(first, second, firstOrigPath, secondOrigPath));
    }
}
