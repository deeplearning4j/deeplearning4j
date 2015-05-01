/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.cli.api.flags;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;

import java.net.URI;

/**
 * Base Input/Output Flag class provides extra URI parsing utilities
 *
 * @author sonali
 */
public abstract class BaseIOFlag implements Flag {
    //URI parsing utilities

    protected RecordReader createReader(URI uri) {
        return null;
    }

    protected RecordWriter createWriter(URI uri) {
        return null;
    }

}
