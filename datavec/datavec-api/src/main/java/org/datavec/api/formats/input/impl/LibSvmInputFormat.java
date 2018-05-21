/*-
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

package org.datavec.api.formats.input.impl;

import org.datavec.api.conf.Configuration;
import org.datavec.api.formats.input.BaseInputFormat;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.misc.LibSvmRecordReader;
import org.datavec.api.split.InputSplit;

import java.io.IOException;

/**
 * Lib svm input format
 *
 * @author Adam Gibson
 */
public class LibSvmInputFormat extends BaseInputFormat {
    @Override
    public RecordReader createReader(InputSplit split, Configuration conf) throws IOException, InterruptedException {
        RecordReader reader = new LibSvmRecordReader();
        reader.initialize(conf, split);
        return reader;
    }
}
