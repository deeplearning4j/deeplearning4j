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

import java.io.*;
import java.net.URI;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;

/**
 * A simple InputSplit based on a collection of URIs
 *
 * @author Alex Black
 */
public class CollectionInputSplit extends BaseInputSplit {

    public CollectionInputSplit(URI[] array){
        this(Arrays.asList(array));
    }

    public CollectionInputSplit(Collection<URI> list) {
        uriStrings = new LinkedList<>();
        for (URI uri : list) {
            uriStrings.add(uri.toString());
        }
    }

    @Override
    public void updateSplitLocations(boolean reset) {

    }

    @Override
    public boolean needsBootstrapForWrite() {
        return false;
    }

    @Override
    public void bootStrapForWrite() {

    }

    @Override
    public OutputStream openOutputStreamFor(String location) throws Exception {
        return null;
    }

    @Override
    public InputStream openInputStreamFor(String location) throws Exception {
        return null;
    }

    @Override
    public long length() {
        return uriStrings.size();
    }

    @Override
    public void reset() {
        //No op
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

 }
