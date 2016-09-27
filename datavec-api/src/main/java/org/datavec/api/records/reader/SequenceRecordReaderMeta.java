/*
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

package org.datavec.api.records.reader;

import org.datavec.api.berkeley.Pair;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.Writable;

import java.io.IOException;
import java.util.List;

/**
 * This is TEMPORARY interface to maintain forward compatibility until 0.7.0 release, at which point
 * the functionality here will be moved to SequenceRecordReader<br>
 * <p>
 * SequenceRecordReaderMeta adds methods that provide both the record AND metadata for the record; it also has methods to
 * load an arbitrary subset of the data (that would normally be returned by the record reader) for one or more specified
 * RecordMetaData instance(s).
 *
 * @author Alex
 */
public interface SequenceRecordReaderMeta extends SequenceRecordReader {

    /**
     * Similar to {@link #sequenceRecord()}, but returns a {@link Record} object, that may include metadata such as the source
     * of the data
     *
     * @return next sequence record
     */
    SequenceRecord nextSequence();

    /**
     * Load a single sequence record from the given {@link RecordMetaData} instance<br>
     * Note: that for data that isn't splittable (i.e., text data that needs to be scanned/split), it is more efficient to
     * load multiple records at once using {@link #loadSequenceFromMetaData(List)}
     *
     * @param recordMetaData Metadata for the sequence record that we want to load from
     * @return Single sequence record for the given RecordMetaData instance
     * @throws IOException If I/O error occurs during loading
     */
    SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException;

    /**
     * Load multiple sequence records from the given a list of {@link RecordMetaData} instances<br>
     *
     * @param recordMetaDatas Metadata for the records that we want to load from
     * @return Multiple sequence record for the given RecordMetaData instances
     * @throws IOException If I/O error occurs during loading
     */
    List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException;

}
