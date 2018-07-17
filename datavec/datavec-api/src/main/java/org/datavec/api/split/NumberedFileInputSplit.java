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

import org.datavec.api.util.files.UriFromPathIterator;
import org.datavec.api.writable.WritableType;

import java.io.*;
import java.net.URI;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**InputSplit for sequences of numbered files.
 * Example usages:<br>
 * Suppose files are sequenced according to "myFile_100.txt", "myFile_101.txt", ..., "myFile_200.txt"
 * then use new NumberedFileInputSplit("myFile_%d.txt",100,200)
 * NumberedFileInputSplit utilizes String.format(), hence the requirement for "%d" to represent
 * the integer index.
 */
public class NumberedFileInputSplit implements InputSplit {
    private final String baseString;
    private final int minIdx;
    private final int maxIdx;

    private static final Pattern p = Pattern.compile("\\%(0\\d)?d");

    /**
     * @param baseString String that defines file format. Must contain "%d", which will be replaced with
     *                   the index of the file, possibly zero-padded to x digits if the pattern is in the form %0xd.
     * @param minIdxInclusive Minimum index/number (starting number in sequence of files, inclusive)
     * @param maxIdxInclusive Maximum index/number (last number in sequence of files, inclusive)
     *                        @see {NumberedFileInputSplitTest}
     */
    public NumberedFileInputSplit(String baseString, int minIdxInclusive, int maxIdxInclusive) {
        Matcher m = p.matcher(baseString);
        if (baseString == null || !m.find()) {
            throw new IllegalArgumentException("Base String must match this regular expression: " + p.toString());
        }
        this.baseString = baseString;
        this.minIdx = minIdxInclusive;
        this.maxIdx = maxIdxInclusive;
    }

    @Override
    public boolean canWriteToLocation(URI location) {
        return location.isAbsolute();
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
        //no-op (locations() is dynamic)
    }

    @Override
    public boolean needsBootstrapForWrite() {
        return locations() == null ||
                locations().length < 1
                || locations().length == 1 && !locations()[0].isAbsolute();
    }

    @Override
    public void bootStrapForWrite() {
        if(locations().length == 1 && !locations()[0].isAbsolute()) {
            File parentDir = new File(locations()[0]);
            File writeFile = new File(parentDir,"write-file");
            try {
                writeFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }


        }
    }

    @Override
    public OutputStream openOutputStreamFor(String location) throws Exception {
        FileOutputStream ret = location.startsWith("file:") ? new FileOutputStream(new File(URI.create(location))):
                new FileOutputStream(new File(location));
        return ret;
    }

    @Override
    public InputStream openInputStreamFor(String location) throws Exception {
        FileInputStream fileInputStream = new FileInputStream(location);
        return fileInputStream;
    }

    @Override
    public long length() {
        return maxIdx - minIdx + 1;
    }

    @Override
    public URI[] locations() {
        URI[] uris = new URI[(int) length()];
        int x = 0;
        for (int i = minIdx; i <= maxIdx; i++) {
            uris[x++] = Paths.get(String.format(baseString, i)).toUri();
        }
        return uris;
    }

    @Override
    public Iterator<URI> locationsIterator() {
        return new UriFromPathIterator(locationsPathIterator());
    }

    @Override
    public Iterator<String> locationsPathIterator() {
        return new NumberedFileIterator();
    }

    @Override
    public void reset() {
        //No op
    }

    @Override
    public boolean resetSupported() {
        return true;
    }


    private class NumberedFileIterator implements Iterator<String> {

        private int currIdx;

        private NumberedFileIterator() {
            currIdx = minIdx;
        }

        @Override
        public boolean hasNext() {
            return currIdx <= maxIdx;
        }

        @Override
        public String next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return String.format(baseString, currIdx++);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }
}