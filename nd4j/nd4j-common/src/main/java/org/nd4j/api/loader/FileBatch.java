/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.api.loader;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * FileBatch: stores the raw contents of multiple files in byte arrays (one per file) along with their original paths.
 * FileBatch can be stored to disk (or any output stream) in zip format.
 * Typical use cases include creating batches of data in their raw format for distributed training. This can reduce the
 * number of disk reads required (fewer files) and network transfers when reading from remote storage, due to the zip
 * format used for compression.
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class FileBatch implements Serializable {
    /**
     * Name of the file in the zip file that contains the original paths/filenames
     */
    public static final String ORIGINAL_PATHS_FILENAME = "originalUris.txt";

    private final List<byte[]> fileBytes;
    private final List<String> originalUris;

    /**
     * Read a FileBatch from the specified file. This method assumes the FileBatch was previously saved to
     * zip format using {@link #writeAsZip(File)} or {@link #writeAsZip(OutputStream)}
     *
     * @param file File to read from
     * @return The loaded FileBatch
     * @throws IOException If an error occurs during reading
     */
    public static FileBatch readFromZip(File file) throws IOException {
        try (FileInputStream fis = new FileInputStream(file)) {
            return readFromZip(fis);
        }
    }

    /**
     * Read a FileBatch from the specified input stream. This method assumes the FileBatch was previously saved to
     * zip format using {@link #writeAsZip(File)} or {@link #writeAsZip(OutputStream)}
     *
     * @param is Input stream to read from
     * @return The loaded FileBatch
     * @throws IOException If an error occurs during reading
     */
    public static FileBatch readFromZip(InputStream is) throws IOException {
        String originalUris = null;
        Map<Integer, byte[]> bytesMap = new HashMap<>();
        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(is))) {
            ZipEntry ze;
            while ((ze = zis.getNextEntry()) != null) {
                String name = ze.getName();
                byte[] bytes = IOUtils.toByteArray(zis);
                if (name.equals(ORIGINAL_PATHS_FILENAME)) {
                    originalUris = new String(bytes, 0, bytes.length, StandardCharsets.UTF_8);
                } else {
                    int idxSplit = name.indexOf("_");
                    int idxSplit2 = name.indexOf(".");
                    int fileIdx = Integer.parseInt(name.substring(idxSplit + 1, idxSplit2));
                    bytesMap.put(fileIdx, bytes);
                }
            }
        }

        List<byte[]> list = new ArrayList<>(bytesMap.size());
        for (int i = 0; i < bytesMap.size(); i++) {
            list.add(bytesMap.get(i));
        }

        List<String> origPaths = Arrays.asList(originalUris.split("\n"));
        return new FileBatch(list, origPaths);
    }

    /**
     * Create a FileBatch from the specified files
     *
     * @param files Files to create the FileBatch from
     * @return The created FileBatch
     * @throws IOException If an error occurs during reading of the file content
     */
    public static FileBatch forFiles(File... files) throws IOException {
        return forFiles(Arrays.asList(files));
    }

    /**
     * Create a FileBatch from the specified files
     *
     * @param files Files to create the FileBatch from
     * @return The created FileBatch
     * @throws IOException If an error occurs during reading of the file content
     */
    public static FileBatch forFiles(List<File> files) throws IOException {
        List<String> origPaths = new ArrayList<>(files.size());
        List<byte[]> bytes = new ArrayList<>(files.size());
        for (File f : files) {
            bytes.add(FileUtils.readFileToByteArray(f));
            origPaths.add(f.toURI().toString());
        }
        return new FileBatch(bytes, origPaths);
    }

    /**
     * Write the FileBatch to the specified File, in zip file format
     *
     * @param f File to write to
     * @throws IOException If an error occurs during writing
     */
    public void writeAsZip(File f) throws IOException {
        writeAsZip(new FileOutputStream(f));
    }

    /**
     * @param os Write the FileBatch to the specified output stream, in zip file format
     * @throws IOException If an error occurs during writing
     */
    public void writeAsZip(OutputStream os) throws IOException {
        try (ZipOutputStream zos = new ZipOutputStream(new BufferedOutputStream(os))) {

            //Write original paths as a text file:
            ZipEntry ze = new ZipEntry(ORIGINAL_PATHS_FILENAME);
            String originalUrisJoined = StringUtils.join(originalUris, "\n"); //Java String.join is Java 8
            zos.putNextEntry(ze);
            zos.write(originalUrisJoined.getBytes(StandardCharsets.UTF_8));

            for (int i = 0; i < fileBytes.size(); i++) {
                String ext = FilenameUtils.getExtension(originalUris.get(i));
                if (ext == null || ext.isEmpty())
                    ext = "bin";
                String name = "file_" + i + "." + ext;
                ze = new ZipEntry(name);
                zos.putNextEntry(ze);
                zos.write(fileBytes.get(i));
            }
        }
    }
}
