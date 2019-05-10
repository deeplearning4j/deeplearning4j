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

package org.datavec.api.formats.input.impl;

import org.datavec.api.conf.Configuration;
import org.datavec.api.formats.input.InputFormat;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.WritableType;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Input format for the @link {ListStringRecordReader}
 * @author Adam Gibson
 */
public class ListStringInputFormat implements InputFormat {
    /**
     * Creates a reader from an input split
     *
     * @param split the split to read
     * @param conf
     * @return the reader from the given input split
     */
    @Override
    public RecordReader createReader(InputSplit split, Configuration conf) throws IOException, InterruptedException {
        RecordReader reader = new ListStringRecordReader();
        reader.initialize(conf, split);
        return reader;
    }

    /**
     * Creates a reader from an input split
     *
     * @param split the split to read
     * @return the reader from the given input split
     */
    @Override
    public RecordReader createReader(InputSplit split) throws IOException, InterruptedException {
        RecordReader reader = new ListStringRecordReader();
        reader.initialize(split);
        return reader;
    }

    /**
     * Serialize the fields of this object to <code>out</code>.
     *
     * @param out <code>DataOuput</code> to serialize this object into.
     * @throws IOException
     */
    @Override
    public void write(DataOutput out) throws IOException {

    }

    /**
     * Deserialize the fields of this object from <code>in</code>.
     * <p>
     * <p>For efficiency, implementations should attempt to re-use storage in the
     * existing object where possible.</p>
     *
     * @param in <code>DataInput</code> to deseriablize this object from.
     * @throws IOException
     */
    @Override
    public void readFields(DataInput in) throws IOException {

    }

    /**
     * Convert Writable to double. Whether this is supported depends on the specific writable.
     */
    @Override
    public double toDouble() {
        return 0;
    }

    /**
     * Convert Writable to float. Whether this is supported depends on the specific writable.
     */
    @Override
    public float toFloat() {
        return 0;
    }

    /**
     * Convert Writable to int. Whether this is supported depends on the specific writable.
     */
    @Override
    public int toInt() {
        return 0;
    }

    /**
     * Convert Writable to long. Whether this is supported depends on the specific writable.
     */
    @Override
    public long toLong() {
        return 0;
    }

    @Override
    public WritableType getType() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        throw new UnsupportedOperationException();
    }
}
