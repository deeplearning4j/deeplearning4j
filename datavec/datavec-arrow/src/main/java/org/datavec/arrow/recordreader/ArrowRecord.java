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

package org.datavec.arrow.recordreader;

import lombok.AllArgsConstructor;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.writable.Writable;

import java.net.URI;
import java.util.List;

/**
 * An {@link ArrowRecord} is a {@link Record}
 * wrapper around {@link ArrowWritableRecordBatch}
 * containing an index to the individual row.
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
public class ArrowRecord implements Record {
    private ArrowWritableRecordBatch arrowWritableRecordBatch;
    private  int index;
    private URI recordUri;

    @Override
    public List<Writable> getRecord() {
        return arrowWritableRecordBatch.get(index);
    }

    @Override
    public void setRecord(List<Writable> record) {
        arrowWritableRecordBatch.set(index,record);
    }

    @Override
    public RecordMetaData getMetaData() {
        RecordMetaData ret = new RecordMetaDataIndex(index,recordUri,ArrowRecordReader.class);
        return ret;
    }

    @Override
    public void setMetaData(RecordMetaData recordMetaData) {

    }
}
