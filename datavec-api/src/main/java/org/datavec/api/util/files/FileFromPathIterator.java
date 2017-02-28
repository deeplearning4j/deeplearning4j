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

package org.datavec.api.util.files;

import lombok.AllArgsConstructor;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * A simple utility method to convert a {@code Iterator<String>} to an {@code Iterator<File>}, where each
 * String in the original iterator is created via URI.toString()
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class FileFromPathIterator implements Iterator<File> {

    private final Iterator<String> paths;

    @Override
    public boolean hasNext() {
        return paths.hasNext();
    }

    @Override
    public File next() {
        if (!hasNext()) {
            throw new NoSuchElementException("No next element");
        }
        try {
            return new File(new URI(paths.next()));
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
