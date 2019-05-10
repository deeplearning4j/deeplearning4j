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

import java.io.InputStream;
import java.io.OutputStream;
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

    @Override
    public boolean resetSupported() {
        return true;
    }



    public List<List<String>> getData() {
        return data;
    }
}
