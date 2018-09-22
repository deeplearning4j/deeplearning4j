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

package org.nd4j.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.base.Preconditions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * @author Adam Gibson
 */
@Slf4j
public class ArchiveUtils {

    protected ArchiveUtils() {
    }

    /**
     * Extracts files to the specified destination
     *
     * @param file the file to extract to
     * @param dest the destination directory
     * @throws IOException
     */
    public static void unzipFileTo(String file, String dest) throws IOException {
        File target = new File(file);
        if (!target.exists())
            throw new IllegalArgumentException("Archive doesnt exist");
        FileInputStream fin = new FileInputStream(target);
        int BUFFER = 2048;
        byte data[] = new byte[BUFFER];

        if (file.endsWith(".zip") || file.endsWith(".jar")) {
            try(ZipInputStream zis = new ZipInputStream(fin)) {
                //get the zipped file list entry
                ZipEntry ze = zis.getNextEntry();

                while (ze != null) {
                    String fileName = ze.getName();
                    File newFile = new File(dest + File.separator + fileName);

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
                    log.debug("File extracted: " + newFile.getAbsoluteFile());
                }

                zis.closeEntry();
            }
        } else if (file.endsWith(".tar.gz") || file.endsWith(".tgz")) {

            BufferedInputStream in = new BufferedInputStream(fin);
            GzipCompressorInputStream gzIn = new GzipCompressorInputStream(in);
            TarArchiveInputStream tarIn = new TarArchiveInputStream(gzIn);

            TarArchiveEntry entry;
            /* Read the tar entries using the getNextEntry method **/
            while ((entry = (TarArchiveEntry) tarIn.getNextEntry()) != null) {
                log.info("Extracting: " + entry.getName());
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
            try(GZIPInputStream is2 = new GZIPInputStream(fin); OutputStream fos = FileUtils.openOutputStream(extracted)) {
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
     * @param tarGzFile A tar.gz file
     * @return List of files and directories
     */
    public static List<String> tarGzListFiles(File tarGzFile) throws IOException {
        try(TarArchiveInputStream tin = new TarArchiveInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(tarGzFile))))) {
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
            List<String> out = new ArrayList<>();
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
