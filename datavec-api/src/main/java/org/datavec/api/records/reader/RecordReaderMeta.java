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
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * This is TEMPORARY interface to maintain forward compatibility until 0.7.0 release, at which point
 * the functionality here will be moved to RecordReader<br>
 *
 * RecordReaderMeta adds methods that provide both the record AND metadata for the record
 *
 * @author Alex
 */
public interface RecordReaderMeta extends RecordReader {

    Pair<List<Writable>,RecordMetaData> nextMeta();

}
