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


import org.datavec.api.writable.Writable;

import java.io.InputStream;
import java.net.URI;
import java.util.Iterator;

/**
 * An input split.
 * Basically, a list of loadable locations
 * exposed as an iterator.
 *
 *
 * @author Adam Gibson
 */
public interface InputSplit extends Writable {


    /**
     * Open an {@link InputStream}
     * for the given location.
     * Note that the user is responsible for closing
     * the associated input stream.
     * @param location the location to open the input stream for
     * @return the opened input stream
     */
    InputStream openInputStreamFor(String location) throws Exception;

    /**
     *  Length of the split
     * @return
     */
    long length();

    /**
     * Locations of the splits
     * @return
     */
    URI[] locations();

    /**
     *
     * @return
     */
    Iterator<URI> locationsIterator();

    /**
     *
     * @return
     */
    Iterator<String> locationsPathIterator();

    /**
     * Reset the InputSplit without reinitializing it from scratch.
     * In many cases, this is a no-op.
     * For InputSplits that have randomization: reset should shuffle the order.
     */
    void reset();

    /**
     * @return True if the reset() method is supported (or is a no-op), false otherwise. If false is returned, reset()
     *         may throw an exception
     */
    boolean resetSupported();
}
