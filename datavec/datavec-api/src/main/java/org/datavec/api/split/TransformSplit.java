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

import lombok.NonNull;
import org.nd4j.linalg.collection.CompactHeapStringList;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Iterator;

/**
 * InputSplit implementation that maps the URIs of a given BaseInputSplit to new URIs. Useful when features and labels
 * are in different files sharing a common naming scheme, and the name of the output file can be determined given the
 * name of the input file.
 *
 * @author Ede Meijer
 */
public class TransformSplit extends BaseInputSplit {
    private final BaseInputSplit sourceSplit;
    private final URITransform transform;

    /**
     * Apply a given transformation to the raw URI objects
     *
     * @param sourceSplit the split with URIs to transform
     * @param transform transform operation that returns a new URI based on an input URI
     * @throws URISyntaxException thrown if the transformed URI is malformed
     */
    public TransformSplit(@NonNull BaseInputSplit sourceSplit, @NonNull URITransform transform)
            throws URISyntaxException {
        this.sourceSplit = sourceSplit;
        this.transform = transform;
        initialize();
    }

    /**
     * Static factory method, replace the string version of the URI with a simple search-replace pair
     *
     * @param sourceSplit the split with URIs to transform
     * @param search the string to search
     * @param replace the string to replace with
     * @throws URISyntaxException thrown if the transformed URI is malformed
     */
    public static TransformSplit ofSearchReplace(@NonNull BaseInputSplit sourceSplit, @NonNull final String search,
                                                 @NonNull final String replace) throws URISyntaxException {
        return new TransformSplit(sourceSplit, new URITransform() {
            @Override
            public URI apply(URI uri) throws URISyntaxException {
                return new URI(uri.toString().replace(search, replace));
            }
        });
    }

    private void initialize() throws URISyntaxException {
        length = sourceSplit.length();
        uriStrings = new CompactHeapStringList();
        Iterator<URI> iter = sourceSplit.locationsIterator();
        while (iter.hasNext()) {
            URI uri = iter.next();
            uri = transform.apply(uri);
            uriStrings.add(uri.toString());
        }
    }


    @Override
    public void updateSplitLocations(boolean reset) {
        sourceSplit.updateSplitLocations(reset);
    }

    @Override
    public boolean needsBootstrapForWrite() {
        return sourceSplit.needsBootstrapForWrite();
    }

    @Override
    public void bootStrapForWrite() {
        sourceSplit.bootStrapForWrite();
    }

    @Override
    public OutputStream openOutputStreamFor(String location) throws Exception {
        return sourceSplit.openOutputStreamFor(location);
    }

    @Override
    public InputStream openInputStreamFor(String location) throws Exception {
        return sourceSplit.openInputStreamFor(location);
    }

    @Override
    public void reset() {
        //No op: BaseInputSplit doesn't support randomization directly, and TransformSplit doesn't either
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    public interface URITransform {
        URI apply(URI uri) throws URISyntaxException;
    }
}
