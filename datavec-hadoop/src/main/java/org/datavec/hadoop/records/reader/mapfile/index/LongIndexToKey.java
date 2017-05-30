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

package org.datavec.hadoop.records.reader.mapfile.index;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.WritableComparable;
import org.datavec.hadoop.records.reader.mapfile.IndexToKey;

import java.io.IOException;

/**
 * Created by Alex on 29/05/2017.
 */
public class LongIndexToKey implements IndexToKey {

    @Override
    public void initialize(MapFile.Reader reader) throws IOException {
        //No-op
    }

    @Override
    public LongWritable getKeyForIndex(long index) {
        return new LongWritable(index);
    }

    @Override
    public long getNumRecords(MapFile.Reader reader) throws IOException {
        //Assumption, from MapFile javadoc:
        //"Map files are created by adding entries in-order."
        //Therefore last key -> tells us how many records there are

        LongWritable l = new LongWritable(-1);
        reader.finalKey(l);

        if(l.get() <= 0){
            throw new IllegalStateException("Invalid number of keys found: " + l.get());
        }

        return l.get() + 1; //Assume zero indexed
    }
}
