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

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.commons.io.filefilter.RegexFileFilter;
import org.apache.commons.io.filefilter.SuffixFileFilter;
import org.datavec.api.util.files.URIUtil;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.collection.CompactHeapStringList;
import org.nd4j.linalg.util.MathUtils;

import java.io.*;
import java.net.URI;
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
//        Collection<File> subFiles;

        if (rootDir == null)
            throw new IllegalArgumentException("File path must not be null");
        else if(rootDir.isAbsolute() && !rootDir.exists()) {
            try {
                if(!rootDir.createNewFile()) {
                    throw new IllegalArgumentException("Unable to create file " + rootDir.getAbsolutePath());
                }
                //ensure uri strings has the root file if it's not a directory
                else {
                    uriStrings = new ArrayList<>();
                    uriStrings.add(rootDir.toURI().toString());
                }
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }
        else if (!rootDir.getAbsoluteFile().exists())
            // When implementing wild card characters in the rootDir, remove this if exists,
            // verify expanded paths exist and check for the edge case when expansion cannot be
            // translated to existed locations
            throw new IllegalArgumentException("No such file or directory: " + rootDir.getAbsolutePath());
        else if (rootDir.isDirectory()) {
            List<File> list = listFiles(rootDir, allowFormat, recursive);

            uriStrings = new CompactHeapStringList();

            if (randomize) {
                iterationOrder = new int[list.size()];
                for (int i = 0; i < iterationOrder.length; i++) {
                    iterationOrder[i] = i;
                }

                MathUtils.shuffleArray(iterationOrder, random);
            }
            for (File f : list) {
                uriStrings.add(URIUtil.fileToURI(f).toString());
                ++length;
            }
        } else {
            // Lists one file
            String toString = URIUtil.fileToURI(rootDir).toString(); //URI.getPath(), getRawPath() etc don't have file:/ prefix necessary for conversion back to URI
            uriStrings = new ArrayList<>(1);
            uriStrings.add(toString);
            length += rootDir.length();
        }
    }

    @Override
    public String addNewLocation() {
        if(rootDir.isDirectory())
            return addNewLocation(new File(rootDir, UUID.randomUUID().toString()).toURI().toString());
        else {
            //add a file in the same directory as the file with the same extension as the original file
            return addNewLocation(new File(rootDir.getParent(), UUID.randomUUID().toString() + "." + FilenameUtils.getExtension(rootDir.getAbsolutePath())).toURI().toString());

        }
    }

    @Override
    public String addNewLocation(String location) {
        File f = new File(URI.create(location));
        try {
            f.createNewFile();
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }

        uriStrings.add(location);
        ++length;
        return location;
    }

    @Override
    public void updateSplitLocations(boolean reset) {
        if (reset) {
            initialize();
        }
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
                //since locations are dynamically generated, allow
                uriStrings.add(writeFile.toURI().toString());
            } catch (IOException e) {
                throw new IllegalStateException(e);
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
        FileInputStream ret = location.startsWith("file:") ? new FileInputStream(new File(URI.create(location))):
                new FileInputStream(new File(location));
        return ret;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public void reset() {
        if (randomize) {
            //Shuffle the iteration order
            MathUtils.shuffleArray(iterationOrder, random);
        }
    }

    @Override
    public boolean resetSupported() {
        return true;
    }


    public File getRootDir() {
        return rootDir;
    }

    private List<File> listFiles(File dir, String[] allowedFormats, boolean recursive) {
        Preconditions.checkState(dir.isDirectory(), "Argument is not a directory: %s", dir);
        IOFileFilter filter;
        if (allowedFormats == null) {
            filter = new RegexFileFilter(".*");
        } else {
            filter = new SuffixFileFilter(allowedFormats);
        }

        LinkedList<File> queue = new LinkedList<>();
        queue.add(dir);

        List<File> out = new ArrayList<>();
        while(!queue.isEmpty()){
            File[] listFiles = queue.remove().listFiles();
            if(listFiles != null){
                for(File f : listFiles){
                    boolean isDir = f.isDirectory();
                    if(isDir && recursive){
                        queue.add(f);
                    } else if(!isDir && filter.accept(f)){
                        out.add(f);
                    }
                }
            }
        }
        return out;
    }
}


