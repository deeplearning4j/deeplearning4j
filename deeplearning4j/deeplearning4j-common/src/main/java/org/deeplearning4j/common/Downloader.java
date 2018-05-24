package org.deeplearning4j.common;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.util.ArchiveUtils;

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

    private Downloader(){ }

    /**
     * Download the specified URL to the specified file, and verify that the target MD5 matches
     * @param name      Name (mainly for providing useful exceptions)
     * @param url       URL to download
     * @param f         Destination file
     * @param targetMD5 Expected MD5 for file
     * @param maxTries  Maximum number of download attempts before failing and throwing an exception
     * @throws IOException If an error occurs during downloading
     */
    public static void download(String name, URL url, File f, String targetMD5, int maxTries) throws IOException {
        download(name, url, f, targetMD5, maxTries, 0);
    }

    private static void download(String name, URL url, File f, String targetMD5, int maxTries, int attempt) throws IOException {
        boolean isCorrectFile = f.exists() && f.isFile() && checkMD5OfFile(targetMD5, f);
        if (attempt < maxTries) {
            if(!isCorrectFile) {
                FileUtils.copyURLToFile(url, f);
                if (!checkMD5OfFile(targetMD5, f)) {
                    f.delete();
                    download(name, url, f, targetMD5, maxTries, attempt + 1);
                }
            }
        } else if (!isCorrectFile) {
            //Too many attempts
            throw new IOException("Could not download " + name + " from " + url + "\n properly despite trying " + maxTries
                    + " times, check your connection.");
        }
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
     * @throws IOException If an error occurs during downloading
     */
    public static void downloadAndExtract(String name, URL url, File f, File extractToDir, String targetMD5, int maxTries) throws IOException {
        downloadAndExtract(0, maxTries, name, url, f, extractToDir, targetMD5);
    }

    private static void downloadAndExtract(int attempt, int maxTries, String name, URL url, File f, File extractToDir, String targetMD5) throws IOException {
        boolean isCorrectFile = f.exists() && f.isFile() && checkMD5OfFile(targetMD5, f);
        if (attempt < maxTries) {
            if(!isCorrectFile) {
                FileUtils.copyURLToFile(url, f);
                if (!checkMD5OfFile(targetMD5, f)) {
                    f.delete();
                    downloadAndExtract(attempt + 1, maxTries, name, url, f, extractToDir, targetMD5);
                }
            }
            // try extracting
            try{
                ArchiveUtils.unzipFileTo(f.getAbsolutePath(), extractToDir.getAbsolutePath());
            } catch (Throwable t){
                log.warn("Error extracting {} files from file {} - retrying...", name, f.getAbsolutePath(), t);
                f.delete();
                downloadAndExtract(attempt + 1, maxTries, name, url, f, extractToDir, targetMD5);
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
