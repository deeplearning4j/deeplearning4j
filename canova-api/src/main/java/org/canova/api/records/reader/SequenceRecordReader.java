/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.api.records.reader;

import org.canova.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;

/**
 * A sequence of records.
 *
 * @author Adam Gibson
 */
public interface SequenceRecordReader extends RecordReader {
    /**
     * Returns a sequence record`
     * @return a sequence of records
     */
    Collection<Collection<Writable>> sequenceRecord();

    /**Load a sequence record from the given DataInputStream
     * Unlike {@link #next()} the internal state of the RecordReader is not modified
     * Implementations of this method should not close the DataInputStream
     * @throws IOException if error occurs during reading from the input stream
     */
    Collection<Collection<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException;
}
