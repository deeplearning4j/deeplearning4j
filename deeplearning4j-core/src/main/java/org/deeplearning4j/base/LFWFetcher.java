/*
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

package org.deeplearning4j.base;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.*;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.util.ArchiveUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads LFW faces data transform. You can customize the size of the images as well
 * Reference: http://vis-www.cs.umass.edu/lfw/
 *
 * @author Adam Gibson
 * @author nyghtowl
 *
 */
public class LFWFetcher {
    protected static final Logger log = LoggerFactory.getLogger(LFWFetcher.class);

    protected static final File BASE_DIR = new File(System.getProperty("user.home"));
    protected String localDir = "LFW";
    protected File fileDir = new File(BASE_DIR, localDir);
    protected boolean isSubset = false;

    public static Map<String, String> lfwData = new HashMap<>();
    public static Map<String, String> lfwLabel = new HashMap<>();
    public static Map<String, String> lfwSubsetData = new HashMap<>();
    public static Map<String, String> lfwSubsetLabel = new HashMap<>();

    public LFWFetcher(String path, boolean isSubset){
        this.localDir = path;
        this.fileDir = new File(BASE_DIR, localDir);
        this.isSubset = isSubset;
        generateLfwMaps();
    }

    public LFWFetcher(boolean isSubset){
        new LFWFetcher(localDir, isSubset);
    }

    public LFWFetcher(String path){
        new LFWFetcher(path, isSubset);
    }

    public LFWFetcher(){}

    public void generateLfwMaps() {
        lfwData.put("filesFilename", "lfw.tgz");
        lfwData.put("filesURL", "http://vis-www.cs.umass.edu/lfw/lfw.tgz");
        lfwData.put("filesFilenameUnzipped", "lfw");

        lfwLabel.put("filesFilename", "lfw-names.txt");
        lfwLabel.put("filesURL", "http://vis-www.cs.umass.edu/lfw/lfw-names.txt");
        lfwLabel.put("filesFilenameUnzipped", "lfw-names.txt");

        // Subset of just faces with a name starting with A
        lfwSubsetData.put("filesFilename", "lfw-a.tgz");
        lfwSubsetData.put("filesURL", "http://vis-www.cs.umass.edu/lfw/lfw-a.tgz");
        lfwSubsetData.put("filesFilenameUnzipped", "lfw-a");
    }

    public void downloadAndUntar(Map urlMap) {
        try {
            File file = new File(fileDir, urlMap.get("filesFilename").toString());
            if (!file.isFile()) {
                FileUtils.copyURLToFile(new URL(urlMap.get("filesURL").toString()), file);
            }

            if (file.toString().endsWith(".tgz"))
                ArchiveUtils.unzipFileTo(file.getAbsolutePath(), fileDir.getAbsolutePath());
        } catch (IOException e) {
            throw new IllegalStateException("Unable to fetch images",e);
        }
    }

    public void fetch()  {
        if (!fileDir.exists()) {
            fileDir.mkdir();

            if (isSubset) {
                log.info("Downloading lfw subset...");
                downloadAndUntar(lfwSubsetData);
            }
            else {
                log.info("Downloading lfw...");
                downloadAndUntar(lfwData);
                downloadAndUntar(lfwLabel);
            }
        }
    }

    public boolean lfwExists(){
        //Check 4 files:
        if(isSubset){
            File f = new File(BASE_DIR, LFWFetcher.lfwSubsetData.get("filesFilenameUnzipped"));
            if (!f.exists()) return false;
        } else {
            File f = new File(BASE_DIR, LFWFetcher.lfwData.get("filesFilenameUnzipped"));
            if (!f.exists()) return false;
            f = new File(BASE_DIR, LFWFetcher.lfwLabel.get("filesFilenameUnzipped"));
            if (!f.exists()) return false;
        }
        return true;
    }


}
