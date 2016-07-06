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

package org.canova.api.io.converters;

import org.canova.api.io.WritableConverter;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.io.data.FloatWritable;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;

/**
 * Convert a writable to a
 * double
 * @author Adam Gibson
 */
public class DoubleWritableConverter implements WritableConverter {
    @Override
    public Writable convert(Writable writable) throws WritableConverterException{
        if(writable instanceof Text || writable instanceof FloatWritable || writable instanceof IntWritable || writable instanceof DoubleWritable) {
            return new DoubleWritable(Double.valueOf(writable.toString()));
        }

        throw new WritableConverterException("Unable to convert type " + writable.getClass());
    }
}
