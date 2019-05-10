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

package org.datavec.api.formats.input;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.WritableType;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * @author Adam Gibson
 */
public abstract class BaseInputFormat implements InputFormat {

    @Override
    public RecordReader createReader(InputSplit split) throws IOException, InterruptedException {
        return createReader(split, null);
    }

    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    @Override
    public double toDouble() {
        throw new UnsupportedOperationException();
    }

    @Override
    public float toFloat() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int toInt() {
        throw new UnsupportedOperationException();
    }

    @Override
    public long toLong() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public WritableType getType(){
        throw new UnsupportedOperationException();
    }
}
