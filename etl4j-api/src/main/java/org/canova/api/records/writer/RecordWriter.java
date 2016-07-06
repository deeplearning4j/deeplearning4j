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

package org.canova.api.records.writer;



import org.canova.api.conf.Configurable;
import org.canova.api.writable.Writable;

import java.io.Closeable;
import java.io.IOException;
import java.util.Collection;

/**
 *  Record writer
 *  @author Adam Gibson
 */
public interface RecordWriter extends Closeable,Configurable {
    String APPEND = "org.canova.api.record.writer.append";

    /**
     * Write a record
     * @param record the record to write
     */
    void write(Collection<Writable> record) throws IOException;


    /**
     * Close the recod reader
     */
    void close();

}
