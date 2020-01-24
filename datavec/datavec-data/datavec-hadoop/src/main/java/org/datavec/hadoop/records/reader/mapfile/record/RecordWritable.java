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

package org.datavec.hadoop.records.reader.mapfile.record;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.hadoop.io.Writable;
import org.datavec.api.writable.WritableFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 29/05/2017.
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
public class RecordWritable implements Writable {
    private List<org.datavec.api.writable.Writable> record;

    @Override
    public void write(DataOutput out) throws IOException {
        WritableFactory wf = WritableFactory.getInstance();
        out.writeInt(record.size());
        for (org.datavec.api.writable.Writable w : record) {
            wf.writeWithType(w, out);
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        WritableFactory wf = WritableFactory.getInstance();
        int numRecords = in.readInt();

        record = new ArrayList<>(numRecords);
        for (int i = 0; i < numRecords; i++) {
            record.add(wf.readWithType(in));
        }
    }
}
