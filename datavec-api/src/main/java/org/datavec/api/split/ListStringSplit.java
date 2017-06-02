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

package org.datavec.api.split;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * An input split that already
 * has delimited data of some kind.
 */
public class ListStringSplit implements InputSplit {
    private List<List<String>> data;


    public ListStringSplit(List<List<String>> data) {
        this.data = data;
    }

    /**
     * Length of the split
     *
     * @return
     */
    @Override
    public long length() {
        return data.size();
    }

    /**
     * Locations of the splits
     *
     * @return
     */
    @Override
    public URI[] locations() {
        return new URI[0];
    }

    @Override
    public Iterator<URI> locationsIterator() {
        return Collections.emptyIterator();
    }

    @Override
    public Iterator<String> locationsPathIterator() {
        return Collections.emptyIterator();
    }

    @Override
    public void reset() {
        //No op
    }

    /**
     * Serialize the fields of this object to <code>out</code>.
     *
     * @param out <code>DataOuput</code> to serialize this object into.
     * @throws IOException
     */
    @Override
    public void write(DataOutput out) throws IOException {

    }

    /**
     * Deserialize the fields of this object from <code>in</code>.
     * <p>
     * <p>For efficiency, implementations should attempt to re-use storage in the
     * existing object where possible.</p>
     *
     * @param in <code>DataInput</code> to deseriablize this object from.
     * @throws IOException
     */
    @Override
    public void readFields(DataInput in) throws IOException {

    }

    /**
     * Convert Writable to double. Whether this is supported depends on the specific writable.
     */
    @Override
    public double toDouble() {
        throw new UnsupportedOperationException();
    }

    /**
     * Convert Writable to float. Whether this is supported depends on the specific writable.
     */
    @Override
    public float toFloat() {
        throw new UnsupportedOperationException();
    }

    /**
     * Convert Writable to int. Whether this is supported depends on the specific writable.
     */
    @Override
    public int toInt() {
        throw new UnsupportedOperationException();
    }

    /**
     * Convert Writable to long. Whether this is supported depends on the specific writable.
     */
    @Override
    public long toLong() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        throw new UnsupportedOperationException();
    }

    public List<List<String>> getData() {
        return data;
    }
}
