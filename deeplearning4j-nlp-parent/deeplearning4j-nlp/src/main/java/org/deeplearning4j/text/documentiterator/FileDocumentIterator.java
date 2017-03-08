/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.text.documentiterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Iterator;

/**
 * Iterate over files
 * @author Adam Gibson
 *
 */
public class FileDocumentIterator implements DocumentIterator {

    private Iterator<File> iter;
    private LineIterator lineIterator;
    private File rootDir;
    private static final Logger log = LoggerFactory.getLogger(FileDocumentIterator.class);

    public FileDocumentIterator(String path) {
        this(new File(path));
    }


    public FileDocumentIterator(File path) {
        if (path.isFile()) {
            iter = Arrays.asList(path).iterator();
            try {
                lineIterator = FileUtils.lineIterator(path);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            this.rootDir = path;
        } else {
            iter = FileUtils.iterateFiles(path, null, true);
            try {
                lineIterator = FileUtils.lineIterator(iter.next());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            this.rootDir = path;
        }


    }

    @Override
    public synchronized InputStream nextDocument() {
        try {
            if (lineIterator != null && !lineIterator.hasNext() && iter.hasNext()) {
                File next = iter.next();
                lineIterator.close();
                lineIterator = FileUtils.lineIterator(next);
                while (!lineIterator.hasNext()) {
                    lineIterator.close();
                    lineIterator = FileUtils.lineIterator(next);
                }


            }

            if (lineIterator != null && lineIterator.hasNext()) {
                return new BufferedInputStream(IOUtils.toInputStream(lineIterator.nextLine()));
            }
        } catch (Exception e) {
            log.warn("Error reading input stream...this is just a warning..Going to return", e);
            return null;
        }

        return null;
    }

    @Override
    public synchronized boolean hasNext() {
        return iter.hasNext() || lineIterator != null && lineIterator.hasNext();
    }

    @Override
    public void reset() {
        if (rootDir.isDirectory())
            iter = FileUtils.iterateFiles(rootDir, null, true);
        else
            iter = Arrays.asList(rootDir).iterator();

    }

}
