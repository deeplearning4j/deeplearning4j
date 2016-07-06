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

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;


import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * File input split. Splits up a root directory in to files.
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
        if (random != null){
            this.random = random;
            this.randomize = true;
        }
        if (runMain) this.initialize();
    }

    public FileSplit(File rootDir) {
        this(rootDir, null, true, null, true);
    }

    public FileSplit(File rootDir, Random rng) {
        this(rootDir, null, true, rng, true);
    }

    public FileSplit(File rootDir, String[] allowFormat) {
        this(rootDir, allowFormat, true,  null, true);
    }

    public FileSplit(File rootDir, String[] allowFormat, Random rng) {
        this(rootDir, allowFormat, true, rng, true);
    }

    public FileSplit(File rootDir, String[] allowFormat, boolean recursive) {
        this(rootDir, allowFormat, recursive, null, true);
    }


    protected void initialize() {
        Collection<File> subFiles;

        if(rootDir == null)
            throw new IllegalArgumentException("File must not be null");

        if(rootDir.isDirectory()) {
            subFiles = FileUtils.listFiles(rootDir, allowFormat, recursive);
            locations = new URI[subFiles.size()];

            if (randomize) Collections.shuffle((List<File>) subFiles, random);
            int count = 0;
            for(File f : subFiles) {
                if(f.getPath().startsWith("file:"))
                    locations[count++] = URI.create(f.getPath());
                else
                    locations[count++] = f.toURI();
                length += f.length();
            }
        }
        else {
            // Lists one file
            String path = rootDir.getPath();
            locations = new URI[1];
            if(path.startsWith("file:"))
                 locations[0] = URI.create(path);
            else
                locations[0] = rootDir.toURI();
            length += rootDir.length();
        }
    }

    @Override
    public long length() {
        return length;
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
}




