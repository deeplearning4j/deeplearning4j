/*-
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.hadoop.records.reader.mapfile;

import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.util.List;

/**
 * An interface to handle Index to key conversion, for use in {@link MapFileReader}
 *
 * @author Alex Black
 */
public interface IndexToKey {

    /**
     * Initialise the instance, and return the first and last record indexes (inclusive) for each reader
     *
     * @param readers The underlying map file readers
     */
    List<Pair<Long, Long>> initialize(MapFile.Reader[] readers, Class<? extends Writable> valueClass)
                    throws IOException;

    /**
     * Get the key for the given index
     *
     * @param index 0 to getNumRecords(reader)
     * @return The key for the given index
     */
    WritableComparable getKeyForIndex(long index);

    /**
     * Getter infer the number of records in the given map file(s)
     *
     * @return Number of records in the map file(s)
     */
    long getNumRecords() throws IOException;

}
