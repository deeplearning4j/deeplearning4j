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

package org.nd4j.common.resources;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.common.util.ArchiveUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

/**
 * Downloader utility methods
 *
 * @author Alex Black
 */
@Slf4j
public class Downloader {
    /**
     * Default connection timeout in milliseconds when using {@link FileUtils#copyURLToFile(URL, File, int, int)}
     */
    public static final int DEFAULT_CONNECTION_TIMEOUT = 60000;
    /**
     * Default read timeout in milliseconds when using {@link FileUtils#copyURLToFile(URL, File, int, int)}
     */
    public static final int DEFAULT_READ_TIMEOUT = 60000;

    private Downloader(){ }

    /**
     * As per {@link #download(String, URL, File, String, int, int, int)} with the connection and read timeouts
     * set to their default values - {@link #DEFAULT_CONNECTION_TIMEOUT} and {@link #DEFAULT_READ_TIMEOUT} respectively
     */
    public static void download(String name, URL url, File f, String targetMD5, int maxTries) throws IOException {
        download(name, url, f, targetMD5, maxTries, DEFAULT_CONNECTION_TIMEOUT, DEFAULT_READ_TIMEOUT);
    }

    /**
     * Download the specified URL to the specified file, and verify that the target MD5 matches
     *
     * @param name              Name (mainly for providing useful exceptions)
     * @param url               URL to download
     * @param f                 Destination file
     * @param targetMD5         Expected MD5 for file
     * @param maxTries          Maximum number of download attempts before failing and throwing an exception
     * @param connectionTimeout connection timeout in milliseconds, as used by {@link org.apache.commons.io.FileUtils#copyURLToFile(URL, File, int, int)}
     * @param readTimeout       read timeout in milliseconds, as used by {@link org.apache.commons.io.FileUtils#copyURLToFile(URL, File, int, int)}
     * @throws IOException If an error occurs during downloading
     */
    public static void download(String name, URL url, File f, String targetMD5, int maxTries, int connectionTimeout, int readTimeout) throws IOException {
        download(name, url, f, targetMD5, maxTries, 0, connectionTimeout, readTimeout);
    }

    private static void download(String name, URL url, File f, String targetMD5, int maxTries, int attempt, int connectionTimeout, int readTimeout) throws IOException {
        boolean isCorrectFile = f.exists() && f.isFile() && checkMD5OfFile(targetMD5, f);
        if (attempt < maxTries) {
            if(!isCorrectFile) {
                FileUtils.copyURLToFile(url, f, connectionTimeout, readTimeout);
                if (!checkMD5OfFile(targetMD5, f)) {
                    f.delete();
                    download(name, url, f, targetMD5, maxTries, attempt + 1, connectionTimeout, readTimeout);
                }
            }
        } else if (!isCorrectFile) {
            //Too many attempts
            throw new IOException("Could not download " + name + " from " + url + "\n properly despite trying " + maxTries
                    + " times, check your connection.");
        }
    }

    /**
     * As per {@link #downloadAndExtract(String, URL, File, File, String, int, int, int)} with the connection and read timeouts
     *      * set to their default values - {@link #DEFAULT_CONNECTION_TIMEOUT} and {@link #DEFAULT_READ_TIMEOUT} respectively
     */
    public static void downloadAndExtract(String name, URL url, File f, File extractToDir, String targetMD5, int maxTries) throws IOException {
        downloadAndExtract(name, url, f, extractToDir, targetMD5, maxTries, DEFAULT_CONNECTION_TIMEOUT, DEFAULT_READ_TIMEOUT);
    }

    /**
     * Download the specified URL to the specified file, verify that the MD5 matches, and then extract it to the specified directory.<br>
     * Note that the file must be an archive, with the correct file extension: .zip, .jar, .tar.gz, .tgz or .gz
     *
     * @param name         Name (mainly for providing useful exceptions)
     * @param url          URL to download
     * @param f            Destination file
     * @param extractToDir Destination directory to extract all files
     * @param targetMD5    Expected MD5 for file
     * @param maxTries     Maximum number of download attempts before failing and throwing an exception
     * @param connectionTimeout connection timeout in milliseconds, as used by {@link org.apache.commons.io.FileUtils#copyURLToFile(URL, File, int, int)}
     * @param readTimeout       read timeout in milliseconds, as used by {@link org.apache.commons.io.FileUtils#copyURLToFile(URL, File, int, int)}
     * @throws IOException If an error occurs during downloading
     */
    public static void downloadAndExtract(String name, URL url, File f, File extractToDir, String targetMD5, int maxTries,
                                          int connectionTimeout, int readTimeout) throws IOException {
        downloadAndExtract(0, maxTries, name, url, f, extractToDir, targetMD5, connectionTimeout, readTimeout);
    }

    private static void downloadAndExtract(int attempt, int maxTries, String name, URL url, File f, File extractToDir,
                                           String targetMD5, int connectionTimeout, int readTimeout) throws IOException {
        boolean isCorrectFile = f.exists() && f.isFile() && checkMD5OfFile(targetMD5, f);
        if (attempt < maxTries) {
            if(!isCorrectFile) {
                FileUtils.copyURLToFile(url, f, connectionTimeout, readTimeout);
                if (!checkMD5OfFile(targetMD5, f)) {
                    f.delete();
                    downloadAndExtract(attempt + 1, maxTries, name, url, f, extractToDir, targetMD5, connectionTimeout, readTimeout);
                }
            }
            // try extracting
            try{
                ArchiveUtils.unzipFileTo(f.getAbsolutePath(), extractToDir.getAbsolutePath(), false);
            } catch (Throwable t){
                log.warn("Error extracting {} files from file {} - retrying...", name, f.getAbsolutePath(), t);
                f.delete();
                downloadAndExtract(attempt + 1, maxTries, name, url, f, extractToDir, targetMD5, connectionTimeout, readTimeout);
            }
        } else if (!isCorrectFile) {
            //Too many attempts
            throw new IOException("Could not download and extract " + name + " from " + url.getPath() + "\n properly despite trying " + maxTries
                    + " times, check your connection. File info:" + "\nTarget MD5: " + targetMD5
                    + "\nHash matches: " + checkMD5OfFile(targetMD5, f) + "\nIs valid file: " + f.isFile());
        }
    }

    /**
     * Check the MD5 of the specified file
     * @param targetMD5 Expected MD5
     * @param file      File to check
     * @return          True if MD5 matches, false otherwise
     */
    public static boolean checkMD5OfFile(String targetMD5, File file) throws IOException {
        InputStream in = FileUtils.openInputStream(file);
        String trueMd5 = DigestUtils.md5Hex(in);
        IOUtils.closeQuietly(in);
        return (targetMD5.equals(trueMd5));
    }

}
