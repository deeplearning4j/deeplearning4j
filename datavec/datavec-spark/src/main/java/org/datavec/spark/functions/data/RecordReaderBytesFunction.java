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

package org.datavec.spark.functions.data;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.Writable;
import scala.Tuple2;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.net.URI;
import java.util.List;

/**RecordReaderBytesFunction: Converts binary data (in the form of a BytesWritable) to DataVec format data
 * ({@code Collection<Writable>}) using a RecordReader
 * @author Alex Black
 */
public class RecordReaderBytesFunction implements Function<Tuple2<Text, BytesWritable>, List<Writable>> {

    private final RecordReader recordReader;

    public RecordReaderBytesFunction(RecordReader recordReader) {
        this.recordReader = recordReader;
    }

    @Override
    public List<Writable> call(Tuple2<Text, BytesWritable> v1) throws Exception {
        URI uri = new URI(v1._1().toString());
        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(v1._2().getBytes()));
        return recordReader.record(uri, dis);
    }


}
