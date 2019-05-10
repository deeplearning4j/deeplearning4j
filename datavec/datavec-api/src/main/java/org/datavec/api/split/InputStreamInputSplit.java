/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.split;

import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

/**
 *
 * Input stream input split.
 * The normal pattern is reading the whole
 * input stream and turning that in to a record.
 * This is meant for streaming raw data
 * rather than normal mini batch pre processing.
 * @author Adam Gibson
 */
public class InputStreamInputSplit implements InputSplit {
    private InputStream is;
    private URI[] location;

    /**
     * Instantiate with the given
     * file as a uri
     * @param is the input stream to use
     * @param path the path to use
     */
    public InputStreamInputSplit(InputStream is, String path) {
        this(is, URI.create(path));
    }

    /**
     * Instantiate with the given
     * file as a uri
     * @param is the input stream to use
     * @param path the path to use
     */
    public InputStreamInputSplit(InputStream is, File path) {
        this(is, path.toURI());
    }

    /**
     * Instantiate with the given
     * file as a uri
     * @param is the input stream to use
     * @param path the path to use
     */
    public InputStreamInputSplit(InputStream is, URI path) {
        this.is = is;
        this.location = new URI[] {path};
    }


    public InputStreamInputSplit(InputStream is) {
        this.is = is;
        this.location = new URI[0];
    }

    @Override
    public boolean canWriteToLocation(URI location) {
        return false;
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
        throw new UnsupportedOperationException();
    }

    @Override
    public void bootStrapForWrite() {
        throw new UnsupportedOperationException();

    }

    @Override
    public OutputStream openOutputStreamFor(String location) throws Exception {
        throw new UnsupportedOperationException();
    }

    @Override
    public InputStream openInputStreamFor(String location) throws Exception {
        return is;
    }

    @Override
    public long length() {
        throw new UnsupportedOperationException();
    }

    @Override
    public URI[] locations() {
        return location;
    }

    @Override
    public Iterator<URI> locationsIterator() {
        return Collections.singletonList(location[0]).iterator();
    }

    @Override
    public Iterator<String> locationsPathIterator() {
        if(location.length >= 1)
            return Collections.singletonList(location[0].getPath()).iterator();
        return Arrays.asList("").iterator();
    }

    @Override
    public void reset() {
        if(!resetSupported()) {
            throw new UnsupportedOperationException("Reset not supported from streams");
        }
        try {
            is = openInputStreamFor(location[0].getPath());
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean resetSupported() {
        return location != null && location.length > 0;
    }


    public InputStream getIs() {
        return is;
    }

    public void setIs(InputStream is) {
        this.is = is;
    }

}
