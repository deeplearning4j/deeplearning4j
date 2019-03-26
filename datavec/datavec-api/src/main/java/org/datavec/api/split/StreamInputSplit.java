/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

import lombok.NonNull;
import org.nd4j.linalg.function.Function;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class StreamInputSplit implements InputSplit {

    protected List<URI> uris;
    protected Function<String,InputStream> streamCreatorFn;

    public StreamInputSplit(@NonNull List<URI> uris, @NonNull Function<String,InputStream> streamCreatorFn){
        this.uris = uris;
        this.streamCreatorFn = streamCreatorFn;
    }

    @Override
    public boolean canWriteToLocation(URI location) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String addNewLocation() {
        throw new UnsupportedOperationException();
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
        throw new UnsupportedOperationException();
    }

    @Override
    public long length() {
        return uris.size();
    }

    @Override
    public URI[] locations() {
        return uris.toArray(new URI[uris.size()]);
    }

    @Override
    public Iterator<URI> locationsIterator() {
        return uris.iterator();
    }

    @Override
    public Iterator<String> locationsPathIterator() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void reset() {
        //No op
    }

    @Override
    public boolean resetSupported() {
        return false;
    }
}
