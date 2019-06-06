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

import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.util.files.ShuffledListIterator;
import org.datavec.api.util.files.UriFromPathIterator;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Base input split
 *
 * @author Adam Gibson
 */
public abstract class BaseInputSplit implements InputSplit {

    protected List<String> uriStrings; //URIs, as a String, via toString() method (which includes file:/ etc)
    protected int[] iterationOrder;
    protected long length = 0;

    @Override
    public boolean canWriteToLocation(URI location) {
        return location.isAbsolute();
    }

    @Override
    public String addNewLocation() {
        throw new UnsupportedOperationException("Unable to add new location.");
    }

    @Override
    public String addNewLocation(String location) {
        throw new UnsupportedOperationException("Unable to add new location.");
    }

    @Override
    public URI[] locations() {
        if(uriStrings == null) {
            uriStrings = new ArrayList<>();
        }

        URI[] uris = new URI[uriStrings.size()];
        int i = 0;
        for (String s : uriStrings) {
            try {
                uris[i++] = new URI(s);
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            }
        }
        return uris;
    }

    @Override
    public Iterator<URI> locationsIterator() {
        return new UriFromPathIterator(locationsPathIterator());
    }

    @Override
    public Iterator<String> locationsPathIterator() {
        if (iterationOrder == null) {
            return uriStrings.iterator();
        }
        return new ShuffledListIterator<>(uriStrings, iterationOrder);
    }

    @Override
    public long length() {
        return 0;
    }



    /**
     * Samples the locations based on the PathFilter and splits the result into
     * an array of InputSplit objects, with sizes proportional to the weights.
     *
     * @param pathFilter to modify the locations in some way (null == as is)
     * @param weights    to split the locations into multiple InputSplit
     * @return           the sampled locations
     */
    // TODO: Specialize in InputStreamInputSplit and others for CSVRecordReader, etc
    public InputSplit[] sample(PathFilter pathFilter, double... weights) {
        URI[] paths = pathFilter != null ? pathFilter.filter(locations()) : locations();

        if (weights != null && weights.length > 0 && weights[0] != 1.0) {
            InputSplit[] splits = new InputSplit[weights.length];
            double totalWeight = 0;
            for (int i = 0; i < weights.length; i++) {
                totalWeight += weights[i];
            }

            double cumulWeight = 0;
            int[] partitions = new int[weights.length + 1];
            for (int i = 0; i < weights.length; i++) {
                partitions[i] = (int) Math.round(cumulWeight * paths.length / totalWeight);
                cumulWeight += weights[i];
            }
            partitions[weights.length] = paths.length;

            for (int i = 0; i < weights.length; i++) {
                List<URI> uris = new ArrayList<>();
                for (int j = partitions[i]; j < partitions[i + 1]; j++) {
                    uris.add(paths[j]);
                }
                splits[i] = new CollectionInputSplit(uris);
            }
            return splits;
        } else {
            return new InputSplit[] {new CollectionInputSplit(Arrays.asList(paths))};
        }
    }

}
