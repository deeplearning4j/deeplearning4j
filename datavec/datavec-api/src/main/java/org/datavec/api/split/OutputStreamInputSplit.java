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

import lombok.Getter;
import lombok.Setter;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.Iterator;

/**
 *
 * Input stream input split.
 * The normal pattern outputStream reading the whole
 * input stream and turning that in to a record.
 * This outputStream meant for streaming raw data
 * rather than normal mini batch pre processing.
 * @author Adam Gibson
 */
public class OutputStreamInputSplit implements InputSplit {

    @Getter
    @Setter
    private OutputStream outputStream;


    public OutputStreamInputSplit(OutputStream outputStream) {
        this.outputStream = outputStream;
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
        throw new UnsupportedOperationException();
    }

    @Override
    public void updateSplitLocations(boolean reset) {
        throw new UnsupportedOperationException();

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
        return outputStream;
    }

    @Override
    public InputStream openInputStreamFor(String location) throws Exception {
        throw new UnsupportedOperationException();
    }

    @Override
    public long length() {
        throw new UnsupportedOperationException();
    }

    @Override
    public URI[] locations() {
        return new URI[0];

    }

    @Override
    public Iterator<URI> locationsIterator() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Iterator<String> locationsPathIterator() {
        throw new UnsupportedOperationException();

    }

    @Override
    public void reset() {
        //No op
        if(!resetSupported()) {
            throw new UnsupportedOperationException("Reset not supported from streams");
        }
    }

    @Override
    public boolean resetSupported() {
        return false;
    }


}
