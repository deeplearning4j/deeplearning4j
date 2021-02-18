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

package org.nd4j.common.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.common.base.Preconditions;

import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;

/**
 * @author Adam Gibson
 */
@Slf4j
public class ArchiveUtils {

    protected ArchiveUtils() {
    }

    /**
     * Extracts all files from the archive to the specified destination.<br>
     * Note: Logs the path of all extracted files by default. Use {@link #unzipFileTo(String, String, boolean)} if
     * logging is not desired.<br>
     * Can handle .zip, .jar, .tar.gz, .tgz, .tar, and .gz formats.
     * Format is interpreted from the filename
     *
     * @param file the file to extract the files from
     * @param dest the destination directory. Will be created if it does not exist
     * @throws IOException If an error occurs accessing the files or extracting
     */
    public static void unzipFileTo(String file, String dest) throws IOException {
        unzipFileTo(file, dest, true);
    }

    /**
     * Extracts all files from the archive to the specified destination, optionally logging the extracted file path.<br>
     * Can handle .zip, .jar, .tar.gz, .tgz, .tar, and .gz formats.
     * Format is interpreted from the filename
     *
     * @param file     the file to extract the files from
     * @param dest     the destination directory. Will be created if it does not exist
     * @param logFiles If true: log the path of every extracted file; if false do not log
     * @throws IOException If an error occurs accessing the files or extracting
     */
    public static void unzipFileTo(String file, String dest, boolean logFiles) throws IOException {
        File target = new File(file);
        if (!target.exists())
            throw new IllegalArgumentException("Archive doesnt exist");
        if (!new File(dest).exists())
            new File(dest).mkdirs();
        FileInputStream fin = new FileInputStream(target);
        int BUFFER = 2048;
        byte data[] = new byte[BUFFER];

        if (file.endsWith(".zip") || file.endsWith(".jar")) {
            try(ZipInputStream zis = new ZipInputStream(fin)) {
                //get the zipped file list entry
                ZipEntry ze = zis.getNextEntry();

                while (ze != null) {
                    String fileName = ze.getName();

                    String canonicalDestinationDirPath = new File(dest).getCanonicalPath();
                    File newFile = new File(dest + File.separator + fileName);
                    String canonicalDestinationFile = newFile.getCanonicalPath();

                    if (!canonicalDestinationFile.startsWith(canonicalDestinationDirPath + File.separator)) {
                        log.debug("Attempt to unzip entry is outside of the target dir");
                        throw new IOException("Entry is outside of the target dir: ");
                    }

                    if (ze.isDirectory()) {
                        newFile.mkdirs();
                        zis.closeEntry();
                        ze = zis.getNextEntry();
                        continue;
                    }

                    FileOutputStream fos = new FileOutputStream(newFile);

                    int len;
                    while ((len = zis.read(data)) > 0) {
                        fos.write(data, 0, len);
                    }

                    fos.close();
                    ze = zis.getNextEntry();
                    if(logFiles) {
                        log.info("File extracted: " + newFile.getAbsoluteFile());
                    }
                }

                zis.closeEntry();
            }
        } else if (file.endsWith(".tar.gz") || file.endsWith(".tgz") || file.endsWith(".tar")) {
            BufferedInputStream in = new BufferedInputStream(fin);
            TarArchiveInputStream tarIn;
            if(file.endsWith(".tar")){
                //Not compressed
                tarIn = new TarArchiveInputStream(in);
            } else {
                GzipCompressorInputStream gzIn = new GzipCompressorInputStream(in);
                 tarIn = new TarArchiveInputStream(gzIn);
            }

            TarArchiveEntry entry;
            /* Read the tar entries using the getNextEntry method **/
            while ((entry = (TarArchiveEntry) tarIn.getNextEntry()) != null) {
                if(logFiles) {
                    log.info("Extracting: " + entry.getName());
                }
                /* If the entry is a directory, create the directory. */

                if (entry.isDirectory()) {
                    File f = new File(dest + File.separator + entry.getName());
                    f.mkdirs();
                }
                /*
                 * If the entry is a file,write the decompressed file to the disk
                 * and close destination stream.
                 */
                else {
                    int count;
                    try(FileOutputStream fos = new FileOutputStream(dest + File.separator + entry.getName());
                        BufferedOutputStream destStream = new BufferedOutputStream(fos, BUFFER);) {
                        while ((count = tarIn.read(data, 0, BUFFER)) != -1) {
                            destStream.write(data, 0, count);
                        }

                        destStream.flush();
                        IOUtils.closeQuietly(destStream);
                    }
                }
            }

            // Close the input stream
            tarIn.close();
        } else if (file.endsWith(".gz")) {
            File extracted = new File(target.getParent(), target.getName().replace(".gz", ""));
            if (extracted.exists())
                extracted.delete();
            extracted.createNewFile();
            try (GZIPInputStream is2 = new GZIPInputStream(fin); OutputStream fos = FileUtils.openOutputStream(extracted)) {
                IOUtils.copyLarge(is2, fos);
                fos.flush();
            }
        } else {
            throw new IllegalStateException("Unable to infer file type (compression format) from source file name: " +
                    file);
        }
        target.delete();
    }

    /**
     * List all of the files and directories in the specified tar.gz file
     *
     * @param tarFile A .tar file
     * @return List of files and directories
     */
    public static List<String> tarListFiles(File tarFile) throws IOException {
        Preconditions.checkState(!tarFile.getPath().endsWith(".tar.gz"), ".tar.gz files should not use this method - use tarGzListFiles instead");
        return tarGzListFiles(tarFile, false);
    }

    /**
     * List all of the files and directories in the specified tar.gz file
     *
     * @param tarGzFile A tar.gz file
     * @return List of files and directories
     */
    public static List<String> tarGzListFiles(File tarGzFile) throws IOException {
        return tarGzListFiles(tarGzFile, true);
    }

    protected static List<String> tarGzListFiles(File file, boolean isTarGz) throws IOException {
        try(TarArchiveInputStream tin =
                    isTarGz ? new TarArchiveInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(file)))) :
                            new TarArchiveInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            ArchiveEntry entry;
            List<String> out = new ArrayList<>();
            while((entry = tin.getNextTarEntry()) != null){
                String name = entry.getName();
                out.add(name);
            }
            return out;
        }
    }

    /**
     * List all of the files and directories in the specified .zip file
     *
     * @param zipFile Zip file
     * @return List of files and directories
     */
    public static List<String> zipListFiles(File zipFile) throws IOException {
        List<String> out = new ArrayList<>();
        try (ZipFile zf = new ZipFile(zipFile)) {
            Enumeration entries = zf.entries();
            while (entries.hasMoreElements()) {
                ZipEntry ze = (ZipEntry) entries.nextElement();
                out.add(ze.getName());
            }
        }
        return out;
    }

    /**
     * Extract a single file from a .zip file. Does not support directories
     *
     * @param zipFile     Zip file to extract from
     * @param destination Destination file
     * @param pathInZip   Path in the zip to extract
     * @throws IOException If exception occurs while reading/writing
     */
    public static void zipExtractSingleFile(File zipFile, File destination, String pathInZip) throws IOException {
        try (ZipFile zf = new ZipFile(zipFile); InputStream is = new BufferedInputStream(zf.getInputStream(zf.getEntry(pathInZip)));
             OutputStream os = new BufferedOutputStream(new FileOutputStream(destination))) {
            IOUtils.copy(is, os);
        }
    }

    /**
     * Extract a single file from a tar.gz file. Does not support directories.
     * NOTE: This should not be used for batch extraction of files, due to the need to iterate over the entries until the
     * specified entry is found. Use {@link #unzipFileTo(String, String)} for batch extraction instead
     *
     * @param tarGz       A tar.gz file
     * @param destination The destination file to extract to
     * @param pathInTarGz The path in the tar.gz file to extract
     */
    public static void tarGzExtractSingleFile(File tarGz, File destination, String pathInTarGz) throws IOException {
        try(TarArchiveInputStream tin = new TarArchiveInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(tarGz))))) {
            ArchiveEntry entry;
            boolean extracted = false;
            while((entry = tin.getNextTarEntry()) != null){
                String name = entry.getName();
                if(pathInTarGz.equals(name)){
                    try(OutputStream os = new BufferedOutputStream(new FileOutputStream(destination))){
                        IOUtils.copy(tin, os);
                    }
                    extracted = true;
                }
            }
            Preconditions.checkState(extracted, "No file was extracted. File not found? %s", pathInTarGz);
        }
    }
}
