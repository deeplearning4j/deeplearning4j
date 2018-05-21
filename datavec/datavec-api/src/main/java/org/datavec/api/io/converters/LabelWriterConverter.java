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

package org.datavec.api.io.converters;

import org.datavec.api.io.WritableConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Convert a label in to an index based on the
 *
 *
 * @author Adam Gibson
 */
public class LabelWriterConverter implements WritableConverter {
    private List<String> labels;

    public LabelWriterConverter(List<String> labels) {
        this.labels = labels;
    }

    @Override
    public Writable convert(Writable writable) throws WritableConverterException {
        String label = writable.toString();
        return new IntWritable(labels.indexOf(label));
    }
}
