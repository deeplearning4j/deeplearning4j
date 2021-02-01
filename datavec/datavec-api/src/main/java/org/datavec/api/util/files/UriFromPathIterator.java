/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.util.files;

import lombok.AllArgsConstructor;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.regex.Pattern;

@AllArgsConstructor
public class UriFromPathIterator implements Iterator<URI> {
    final Pattern schemaPattern = Pattern.compile("^.*?:/.*");

    private final Iterator<String> paths;

    @Override
    public boolean hasNext() {
        return paths.hasNext();
    }

    @Override
    public URI next() {

        if (!hasNext()) {
            throw new NoSuchElementException("No next element");
        }
        try {
            String s = paths.next();
            if(schemaPattern.matcher(s).matches()){
                return new URI(s);
            } else {
                //No scheme - assume file for backward compatibility
                return new File(s).toURI();
            }

        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
