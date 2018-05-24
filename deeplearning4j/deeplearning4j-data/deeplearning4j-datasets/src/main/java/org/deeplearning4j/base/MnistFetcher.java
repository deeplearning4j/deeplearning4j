/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.nd4j.util.ArchiveUtils;

import java.io.*;
import java.net.URL;

@Data
@NoArgsConstructor
@Slf4j
public class MnistFetcher {

    protected static final String LOCAL_DIR_NAME = "MNIST";

    private File fileDir;
    private static final String TRAINING_FILES_URL_RELATIVE = "datasets/mnist/train-images-idx3-ubyte.gz";
    private static final String TRAINING_FILES_MD_5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873";
    private static final String TRAINING_FILES_FILENAME = "train-images-idx3-ubyte.gz";
    public static final String TRAINING_FILES_FILENAME_UNZIPPED = "train-images-idx3-ubyte";
    private static final String TRAINING_FILE_LABELS_URL_RELATIVE = "datasets/mnist/train-labels-idx1-ubyte.gz";
    private static final String TRAINING_FILE_LABELS_MD_5 = "d53e105ee54ea40749a09fcbcd1e9432";
    private static final String TRAINING_FILE_LABELS_FILENAME = "train-labels-idx1-ubyte.gz";
    public static final String TRAINING_FILE_LABELS_FILENAME_UNZIPPED = "train-labels-idx1-ubyte";

    //Test data:
    private static final String TEST_FILES_URL_RELATIVE = "datasets/mnist/t10k-images-idx3-ubyte.gz";
    private static final String TEST_FILES_MD_5 = "9fb629c4189551a2d022fa330f9573f3";
    private static final String TEST_FILES_FILENAME = "t10k-images-idx3-ubyte.gz";
    public static final String TEST_FILES_FILENAME_UNZIPPED = "t10k-images-idx3-ubyte";
    private static final String TEST_FILE_LABELS_URL_RELATIVE = "datasets/mnist/t10k-labels-idx1-ubyte.gz";
    private static final String TEST_FILE_LABELS_MD_5 = "ec29112dd5afa0611ce80d1b7f02629c";
    private static final String TEST_FILE_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz";
    public static final String TEST_FILE_LABELS_FILENAME_UNZIPPED = "t10k-labels-idx1-ubyte";


    public String getName() {
        return "MNIST";
    }

    public File getBaseDir() {
        return DL4JResources.getDirectory(ResourceType.DATASET, getName());
    }

    // --- Train files ---
    public String getTrainingFilesURL() {
        return DL4JResources.getURLString(TRAINING_FILES_URL_RELATIVE);
    }

    public String getTrainingFilesMD5() {
        return TRAINING_FILES_MD_5;
    }

    public String getTrainingFilesFilename() {
        return TRAINING_FILES_FILENAME;
    }

    public String getTrainingFilesFilename_unzipped() {
        return TRAINING_FILES_FILENAME_UNZIPPED;
    }

    public String getTrainingFileLabelsURL() {
        return DL4JResources.getURLString(TRAINING_FILE_LABELS_URL_RELATIVE);
    }

    public String getTrainingFileLabelsMD5() {
        return TRAINING_FILE_LABELS_MD_5;
    }

    public String getTrainingFileLabelsFilename() {
        return TRAINING_FILE_LABELS_FILENAME;
    }

    public String getTrainingFileLabelsFilename_unzipped() {
        return TRAINING_FILE_LABELS_FILENAME_UNZIPPED;
    }


    // --- Test files ---

    public String getTestFilesURL() {
        return DL4JResources.getURLString(TEST_FILES_URL_RELATIVE);
    }

    public String getTestFilesMD5() {
        return TEST_FILES_MD_5;
    }

    public String getTestFilesFilename() {
        return TEST_FILES_FILENAME;
    }

    public String getTestFilesFilename_unzipped() {
        return TEST_FILES_FILENAME_UNZIPPED;
    }

    public String getTestFileLabelsURL() {
        return DL4JResources.getURLString(TEST_FILE_LABELS_URL_RELATIVE);
    }

    public String getTestFileLabelsMD5() {
        return TEST_FILE_LABELS_MD_5;
    }

    public String getTestFileLabelsFilename() {
        return TEST_FILE_LABELS_FILENAME;
    }

    public String getTestFileLabelsFilename_unzipped() {
        return TEST_FILE_LABELS_FILENAME_UNZIPPED;
    }


    public File downloadAndUntar() throws IOException {
        if (fileDir != null) {
            return fileDir;
        }

        File baseDir = getBaseDir();
        if (!(baseDir.isDirectory() || baseDir.mkdir())) {
            throw new IOException("Could not mkdir " + baseDir);
        }

        log.info("Downloading {}...", getName());
        // get features
        File trainFeatures = new File(baseDir, getTrainingFilesFilename());
        File testFeatures = new File(baseDir, getTestFilesFilename());

        downloadAndExtract(new URL(getTrainingFilesURL()), trainFeatures, baseDir, getTrainingFilesMD5());
        downloadAndExtract(new URL(getTestFilesURL()), testFeatures, baseDir, getTestFilesMD5());

        // get labels
        File trainLabels = new File(baseDir, getTrainingFileLabelsFilename());
        File testLabels = new File(baseDir, getTestFileLabelsFilename());

        downloadAndExtract(new URL(getTrainingFileLabelsURL()), trainLabels, baseDir, getTrainingFileLabelsMD5());
        downloadAndExtract(new URL(getTestFileLabelsURL()), testLabels, baseDir, getTestFileLabelsMD5());

        fileDir = baseDir;
        return fileDir;
    }

    private void downloadAndExtract(URL url, File f, File extractToDir, String targetMD5) throws IOException {
        downloadAndExtract(0, url, f, extractToDir, targetMD5);
    }

    private void downloadAndExtract(int attempt, URL url, File f, File extractToDir, String targetMD5) throws IOException {
        int maxTries = 3;
        boolean isCorrectFile = f.exists() && f.isFile() && checkMD5OfFile(targetMD5, f);
        if (attempt < maxTries) {
            if(!isCorrectFile) {
                FileUtils.copyURLToFile(url, f);
                if (!checkMD5OfFile(targetMD5, f)) {
                    f.delete();
                    downloadAndExtract(attempt + 1, url, f, extractToDir, targetMD5);
                }
            }
            // try extracting
            try{
                ArchiveUtils.unzipFileTo(f.getAbsolutePath(), extractToDir.getAbsolutePath());
            } catch (Throwable t){
                log.warn("Error extracting MNIST files from file {} - retrying...", f.getAbsolutePath(), t);
                f.delete();
                downloadAndExtract(attempt + 1, url, f, extractToDir, targetMD5);
            }
        } else if (!isCorrectFile) {
            //Too many attempts
            throw new IOException("Could not download and extract " + url.getPath() + "\n properly despite trying " + maxTries
                            + " times, check your connection. File info:" + "\nTarget MD5: " + targetMD5
                            + "\nHash matches: " + checkMD5OfFile(targetMD5, f) + "\nIs valid file: " + f.isFile());
        }
    }

    private boolean checkMD5OfFile(String targetMD5, File file) throws IOException {
        InputStream in = FileUtils.openInputStream(file);
        String trueMd5 = DigestUtils.md5Hex(in);
        IOUtils.closeQuietly(in);
        return (targetMD5.equals(trueMd5));
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
