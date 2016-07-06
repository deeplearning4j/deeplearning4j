/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.api.split;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.canova.api.io.filters.PathFilter;

/**
 * Base input split
 *
 * @author Adam Gibson
 */
public abstract class BaseInputSplit implements InputSplit {

    protected URI[] locations;
    protected long length = 0;

    @Override
    public URI[] locations() {
        return locations;
    }

    @Override
    public long length() {
        return 0;
    }

    @Override
    public double toDouble(){
        throw new UnsupportedOperationException();
    }

    @Override
    public float toFloat(){
        throw new UnsupportedOperationException();
    }

    @Override
    public int toInt(){
        throw new UnsupportedOperationException();
    }

    @Override
    public long toLong(){
        throw new UnsupportedOperationException();
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

        if (weights != null && weights.length > 0) {
            InputSplit[] splits = new InputSplit[weights.length];
            double totalWeight = 0;
            for (int i = 0; i < weights.length; i++) {
                totalWeight += weights[i];
            }

            double cumulWeight = 0;
            int[] partitions = new int[weights.length + 1];
            for (int i = 0; i < weights.length; i++) {
                partitions[i] = (int)Math.round(cumulWeight * paths.length / totalWeight);
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
            return new InputSplit[] { new CollectionInputSplit(Arrays.asList(paths)) };
        }
    }
}
