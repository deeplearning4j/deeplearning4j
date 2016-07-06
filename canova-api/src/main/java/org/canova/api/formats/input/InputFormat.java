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

package org.canova.api.formats.input;


import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;

import java.io.IOException;

/**
 * Create an input format
 *
 * @author Adam Gibson
 */
public interface InputFormat extends Writable {

    /**
     * Creates a reader from an input split
     * @param split the split to read
     * @return the reader from the given input split
     */
    RecordReader createReader(InputSplit split,Configuration conf) throws IOException, InterruptedException;

    /**
     * Creates a reader from an input split
     * @param split the split to read
     * @return the reader from the given input split
     */
    RecordReader createReader(InputSplit split) throws IOException, InterruptedException;

}
