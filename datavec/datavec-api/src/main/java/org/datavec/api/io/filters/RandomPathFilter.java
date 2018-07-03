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
package org.datavec.api.io.filters;

import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Randomizes the order of paths in an array.
 *
 * @author saudet
 */
public class RandomPathFilter implements PathFilter {

    protected Random random;
    protected String[] extensions;
    protected long maxPaths = 0;

    /** Calls {@code this(random, extensions, 0)}. */
    public RandomPathFilter(Random random, String... extensions) {
        this(random, extensions, 0);
    }

    /**
     * Constructs an instance of the PathFilter.
     *
     * @param random     object to use
     * @param extensions of files to keep
     * @param maxPaths   max number of paths to return (0 == unlimited)
     */
    public RandomPathFilter(Random random, String[] extensions, long maxPaths) {
        this.random = random;
        this.extensions = extensions;
        this.maxPaths = maxPaths;
    }

    protected boolean accept(String name) {
        if (extensions == null || extensions.length == 0) {
            return true;
        }
        for (String extension : extensions) {
            if (name.endsWith("." + extension)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public URI[] filter(URI[] paths) {
        ArrayList<URI> newpaths = new ArrayList<URI>();
        for (URI path : paths) {
            if (accept(path.toString())) {
                newpaths.add(path);
            }
            if (maxPaths > 0 && newpaths.size() >= maxPaths) {
                break;
            }
        }
        Collections.shuffle(newpaths, random);
        return newpaths.toArray(new URI[newpaths.size()]);
    }
}
