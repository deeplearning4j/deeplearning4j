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

import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.Collections;
import java.util.Iterator;

/**
 * String split used for single line inputs
 * @author Adam Gibson
 */
public class StringSplit implements InputSplit {
    private String data;

    public StringSplit(String data) {
        this.data = data;
    }

    @Override
    public boolean canWriteToLocation(URI location) {
        return true;
    }

    @Override
    public String addNewLocation() {
        return null;
    }

    @Override
    public String addNewLocation(String location) {
        return null;
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
        return data.length();
    }

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

    @Override
    public boolean resetSupported() {
        return true;
    }



    public String getData() {
        return data;
    }

}
