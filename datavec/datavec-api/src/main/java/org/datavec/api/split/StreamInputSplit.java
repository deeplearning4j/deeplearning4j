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

import lombok.Data;
import lombok.NonNull;
import org.datavec.api.util.files.ShuffledListIterator;
import org.nd4j.common.function.Function;
import org.nd4j.common.util.MathUtils;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.*;

/**
 * StreamInputSplit is a way of specifying input as a bunch of URIs, as well as the way those URIs should be opened.
 * For example, if data was stored remotely (HDFS, S3, etc) you could use StreamInputSplit to load them, doing two things:
 * (a) providing the URIs of the remote data to the constructor<br>
 * (b) providing a {@code Function<URI,InputStream>} that opens an InputStream for the given URI.<br>
 * <br>
 * Note: supports optional randomization (shuffling of order in which streams are read) via {@link #StreamInputSplit(List, Function, Random)}
 * by providing a {@link Random} instance. If no Random instance is provided, the order will always be according to the
 * order in the provided list of URIs.
 *
 * @author Alex Black
 */
@Data
public class StreamInputSplit implements InputSplit {

    protected List<URI> uris;
    protected Function<URI,InputStream> streamCreatorFn;
    protected Random rng;
    protected int[] order;

    /**
     * Create a StreamInputSplit with no randomization
     *
     * @param uris            The list of URIs to load
     * @param streamCreatorFn The function to be used to create InputStream objects for a given URI.
     */
    public StreamInputSplit(@NonNull List<URI> uris, @NonNull Function<URI,InputStream> streamCreatorFn) {
        this(uris, streamCreatorFn, null);
    }

    /**
     * Create a StreamInputSplit with optional randomization
     *
     * @param uris            The list of URIs to load
     * @param streamCreatorFn The function to be used to create InputStream objects for a given URI
     * @param rng             Random number generator instance. If non-null: streams will be iterated over in a random
     *                        order. If null: no randomization (iteration order is according to the URIs list)
     */
    public StreamInputSplit(@NonNull List<URI> uris, @NonNull Function<URI,InputStream> streamCreatorFn, Random rng){
        this.uris = uris;
        this.streamCreatorFn = streamCreatorFn;
        this.rng = rng;
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
        if(rng == null){
            return uris.iterator();
        } else {
            if(order == null){
                order = new int[uris.size()];
                for( int i=0; i<order.length; i++ ){
                    order[i] = i;
                }
            }
            MathUtils.shuffleArray(order, rng);
            return new ShuffledListIterator<>(uris, order);
        }
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
        return true;
    }
}
