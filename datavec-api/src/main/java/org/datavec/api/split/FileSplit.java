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

package org.datavec.api.split;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.*;
import org.datavec.api.util.RandomUtils;
import org.datavec.api.util.files.URIUtil;
import org.nd4j.linalg.collection.CompactHeapStringList;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * File input split. Splits up a root directory in to files.
 *
 * @author Adam Gibson
 */
public class FileSplit extends BaseInputSplit {

    protected File rootDir;
    // Use for Collections, pass in list of file type strings
    protected String[] allowFormat = null;
    protected boolean recursive = true;
    protected Random random;
    protected boolean randomize = false;

    protected FileSplit(File rootDir, String[] allowFormat, boolean recursive, Random random, boolean runMain) {
        this.allowFormat = allowFormat;
        this.recursive = recursive;
        this.rootDir = rootDir;
        if (random != null) {
            this.random = random;
            this.randomize = true;
        }
        if (runMain)
            this.initialize();
    }

    public FileSplit(File rootDir) {
        this(rootDir, null, true, null, true);
    }

    public FileSplit(File rootDir, Random rng) {
        this(rootDir, null, true, rng, true);
    }

    public FileSplit(File rootDir, String[] allowFormat) {
        this(rootDir, allowFormat, true, null, true);
    }

    public FileSplit(File rootDir, String[] allowFormat, Random rng) {
        this(rootDir, allowFormat, true, rng, true);
    }

    public FileSplit(File rootDir, String[] allowFormat, boolean recursive) {
        this(rootDir, allowFormat, recursive, null, true);
    }


    protected void initialize() {
        Collection<File> subFiles;

        if (rootDir == null)
            throw new IllegalArgumentException("File path must not be null");
        else if (!rootDir.getAbsoluteFile().exists())
            // When implementing wild card characters in the rootDir, remove this if exists,
            // verify expanded paths exist and check for the edge case when expansion cannot be
            // translated to existed locations
            throw new IllegalArgumentException("No such file or directory: " + rootDir.getAbsolutePath());
        else if (rootDir.isDirectory()) {
            subFiles = new LinkedList<>();
            listFiles(subFiles, rootDir.toPath(), allowFormat, recursive);

            uriStrings = new CompactHeapStringList();

            if (randomize) {
                iterationOrder = new int[subFiles.size()];
                for (int i = 0; i < iterationOrder.length; i++) {
                    iterationOrder[i] = i;
                }
                RandomUtils.shuffleInPlace(iterationOrder, random);
            }
            for (File f : subFiles) {
                uriStrings.add(URIUtil.fileToURI(f).toString());
                ++length;
            }
        } else {
            // Lists one file
            String toString = rootDir.toURI().toString(); //URI.getPath(), getRawPath() etc don't have file:/ prefix necessary for conversion back to URI
            uriStrings = Collections.singletonList(toString);
            length += rootDir.length();
        }
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public void reset() {
        if (randomize) {
            //Shuffle the iteration order
            RandomUtils.shuffleInPlace(iterationOrder, random);
        }
    }


    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    public File getRootDir() {
        return rootDir;
    }

    private Collection<File> listFiles(Collection<File> fileNames, Path dir, String[] allowedFormats, boolean recursive) {
        IOFileFilter filter;
        if(allowedFormats==null) {
            filter = new RegexFileFilter(".*");
        } else {
            filter = new SuffixFileFilter(allowedFormats);
        }

        try(DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path path : stream) {
                if(Files.isDirectory(path) && recursive) {
                    listFiles(fileNames, path, allowedFormats, recursive);
                } else {
                    if(allowedFormats==null) {
                        fileNames.add(path.toFile());
                    } else {
                        if (filter.accept(path.toFile())) {
                            fileNames.add(path.toFile());
                        }
                    }
                }
            }
        } catch(IOException e) {
            e.printStackTrace();
        }
        return fileNames;
    }
}


