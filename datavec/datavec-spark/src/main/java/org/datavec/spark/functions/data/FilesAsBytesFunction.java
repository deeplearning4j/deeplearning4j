/*-
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

package org.datavec.spark.functions.data;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import scala.Tuple2;

/**A PairFunction that simply loads bytes[] from a a PortableDataStream, and wraps it (and the String key)
 * in Text and BytesWritable respectively.
 * @author Alex Black
 */
public class FilesAsBytesFunction implements PairFunction<Tuple2<String, PortableDataStream>, Text, BytesWritable> {
    @Override
    public Tuple2<Text, BytesWritable> call(Tuple2<String, PortableDataStream> in) throws Exception {
        return new Tuple2<>(new Text(in._1()), new BytesWritable(in._2().toArray()));
    }
}
