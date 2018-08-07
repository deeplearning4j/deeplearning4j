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

package org.datavec.api.records.writer.impl.misc;


import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.split.partition.PartitionMetaData;
import org.datavec.api.writable.Writable;

import java.io.IOException;
import java.util.List;

/**
 * Write matlab records
 *
 * @author Adam Gibson
 */
public class MatlabRecordWriter extends FileRecordWriter {
    public MatlabRecordWriter() {}


    @Override
    public PartitionMetaData write(List<Writable> record) throws IOException {
        StringBuilder result = new StringBuilder();

        int count = 0;
        for (Writable w : record) {
            // attributes
            if (count > 0) {
                boolean tabs = false;
                result.append((tabs ? "\t" : " "));
            }
            result.append(w.toString());
            count++;

        }

        out.write(result.toString().getBytes());
        out.write(NEW_LINE.getBytes());

        return PartitionMetaData.builder().numRecordsUpdated(1).build();

    }
}
