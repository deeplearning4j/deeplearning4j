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

package org.datavec.image.mnist;

import org.apache.commons.io.FileUtils;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;

public class MnistFetcher {

    private File fileDir;
    private static Logger log = LoggerFactory.getLogger(MnistFetcher.class);
    private static final String trainingFilesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";

    private static final String trainingFilesFilename = "images-idx1-ubyte.gz";
    public static final String trainingFilesFilename_unzipped = "images-idx1-ubyte";

    private static final String trainingFileLabelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
    private static final String trainingFileLabelsFilename = "labels-idx1-ubyte.gz";
    public static final String trainingFileLabelsFilename_unzipped = "labels-idx1-ubyte";
    private static final String LOCAL_DIR_NAME = "MNIST";



    public File downloadAndUntar() throws IOException {
        if (fileDir != null) {
            return fileDir;
        }
        // mac gives unique tmp each run and we want to store this persist
        // this data across restarts
        File tmpDir = new File(System.getProperty("user.home"));

        File baseDir = new File(tmpDir, LOCAL_DIR_NAME);
        if (!(baseDir.isDirectory() || baseDir.mkdir())) {
            throw new IOException("Could not mkdir " + baseDir);
        }


        log.info("Downloading mnist...");
        // getFromOrigin training records
        File tarFile = new File(baseDir, trainingFilesFilename);

        if (!tarFile.isFile()) {
            FileUtils.copyURLToFile(new URL(trainingFilesURL), tarFile);
        }

        ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), baseDir.getAbsolutePath());



        // getFromOrigin training records
        File labels = new File(baseDir, trainingFileLabelsFilename);

        if (!labels.isFile()) {
            FileUtils.copyURLToFile(new URL(trainingFileLabelsURL), labels);
        }

        ArchiveUtils.unzipFileTo(labels.getAbsolutePath(), baseDir.getAbsolutePath());



        fileDir = baseDir;
        return fileDir;
    }

    public void untarFile(File baseDir, File tarFile) throws IOException {

        log.info("Untaring File: " + tarFile.toString());

        Process p = Runtime.getRuntime()
                        .exec(String.format("tar -C %s -xvf %s", baseDir.getAbsolutePath(), tarFile.getAbsolutePath()));
        BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        log.info("Here is the standard error of the command (if any):\n");
        String s;
        while ((s = stdError.readLine()) != null) {
            log.info(s);
        }
        stdError.close();


    }

    public static void gunzipFile(File baseDir, File gzFile) throws IOException {

        log.info("gunzip'ing File: " + gzFile.toString());

        Process p = Runtime.getRuntime().exec(String.format("gunzip %s", gzFile.getAbsolutePath()));
        BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        log.info("Here is the standard error of the command (if any):\n");
        String s;
        while ((s = stdError.readLine()) != null) {
            log.info(s);
        }
        stdError.close();


    }


}
