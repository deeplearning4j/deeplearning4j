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

package org.nd4j.common.loader;

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

@AllArgsConstructor
@Data
public class FileBatch implements Serializable {
    /**
     * Name of the file in the zip file that contains the original paths/filenames
     */
    public static final String ORIGINAL_PATHS_FILENAME = "originalUris.txt";

    /**
     * Maximum total decompressed size allowed when reading FileBatch from ZIP (default: 1GB).
     * This limit protects against zip bomb attacks.
     * Can be overridden via system property "nd4j.filebatch.maxZipSize" (in bytes).
     */
    public static final long DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE = 1024L * 1024L * 1024L; // 1GB

    /**
     * Maximum allowed compression ratio (uncompressed/compressed size).
     * Default is 100:1. Can be overridden via system property "nd4j.filebatch.maxCompressionRatio".
     */
    public static final double DEFAULT_MAX_COMPRESSION_RATIO = 100.0;

    /**
     * Maximum number of entries allowed in a FileBatch ZIP file.
     * Can be overridden via system property "nd4j.filebatch.maxZipEntries".
     */
    public static final int DEFAULT_MAX_ZIP_ENTRIES = 10000;

    private static long maxTotalUncompressedSize = getConfiguredMaxSize();
    private static double maxCompressionRatio = getConfiguredMaxRatio();
    private static int maxZipEntries = getConfiguredMaxEntries();

    private static long getConfiguredMaxSize() {
        String prop = System.getProperty("nd4j.filebatch.maxZipSize");
        if (prop != null) {
            try {
                return Long.parseLong(prop);
            } catch (NumberFormatException e) {
                // Use default
            }
        }
        return DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE;
    }

    private static double getConfiguredMaxRatio() {
        String prop = System.getProperty("nd4j.filebatch.maxCompressionRatio");
        if (prop != null) {
            try {
                return Double.parseDouble(prop);
            } catch (NumberFormatException e) {
                // Use default
            }
        }
        return DEFAULT_MAX_COMPRESSION_RATIO;
    }

    private static int getConfiguredMaxEntries() {
        String prop = System.getProperty("nd4j.filebatch.maxZipEntries");
        if (prop != null) {
            try {
                return Integer.parseInt(prop);
            } catch (NumberFormatException e) {
                // Use default
            }
        }
        return DEFAULT_MAX_ZIP_ENTRIES;
    }

    /**
     * Set the maximum total uncompressed size allowed when reading FileBatch ZIP files.
     * @param maxSize Maximum size in bytes (must be positive)
     */
    public static void setMaxTotalUncompressedSize(long maxSize) {
        if (maxSize <= 0) throw new IllegalArgumentException("Max size must be positive, got " + maxSize);
        maxTotalUncompressedSize = maxSize;
    }

    /**
     * Set the maximum compression ratio allowed for ZIP entries.
     * @param maxRatio Maximum ratio (must be >= 1.0)
     */
    public static void setMaxCompressionRatio(double maxRatio) {
        if (maxRatio < 1.0) throw new IllegalArgumentException("Max ratio must be >= 1.0, got " + maxRatio);
        maxCompressionRatio = maxRatio;
    }

    /**
     * Set the maximum number of entries allowed in FileBatch ZIP files.
     * @param maxEntries Maximum entries (must be positive)
     */
    public static void setMaxZipEntries(int maxEntries) {
        if (maxEntries <= 0) throw new IllegalArgumentException("Max entries must be positive, got " + maxEntries);
        maxZipEntries = maxEntries;
    }

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
        long totalBytesRead = 0;
        int entryCount = 0;

        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(is))) {
            ZipEntry ze;
            while ((ze = zis.getNextEntry()) != null) {
                // Check entry count limit
                entryCount++;
                if (entryCount > maxZipEntries) {
                    throw new IOException("Potential zip bomb detected: too many entries. " +
                            "Found " + entryCount + " entries, maximum allowed is " + maxZipEntries + ". " +
                            "If this is a legitimate FileBatch, increase the limit using " +
                            "FileBatch.setMaxZipEntries() or system property 'nd4j.filebatch.maxZipEntries'");
                }

                String name = ze.getName();

                // Check compression ratio if sizes are known
                long compressedSize = ze.getCompressedSize();
                long uncompressedSize = ze.getSize();
                if (compressedSize > 0 && uncompressedSize > 0) {
                    double ratio = (double) uncompressedSize / compressedSize;
                    if (ratio > maxCompressionRatio) {
                        throw new IOException("Potential zip bomb detected: suspicious compression ratio. " +
                                "Entry '" + name + "' has ratio " + String.format("%.1f", ratio) +
                                ":1 (compressed: " + compressedSize + " bytes, uncompressed: " + uncompressedSize + " bytes). " +
                                "Maximum allowed ratio is " + String.format("%.1f", maxCompressionRatio) + ":1. " +
                                "If this is a legitimate FileBatch, increase the limit using " +
                                "FileBatch.setMaxCompressionRatio() or system property 'nd4j.filebatch.maxCompressionRatio'");
                    }
                }

                // Read with size limit protection
                byte[] bytes = readZipEntryWithLimit(zis, name, maxTotalUncompressedSize - totalBytesRead);
                totalBytesRead += bytes.length;

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
     * Read a ZIP entry with size limit protection.
     */
    private static byte[] readZipEntryWithLimit(ZipInputStream zis, String entryName, long remainingAllowed)
            throws IOException {
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        long totalRead = 0;
        int bytesRead;

        while ((bytesRead = zis.read(buffer)) != -1) {
            totalRead += bytesRead;

            if (totalRead > remainingAllowed) {
                throw new IOException("Potential zip bomb detected while reading entry '" + entryName + "'. " +
                        "Entry exceeded remaining size allowance of " + remainingAllowed + " bytes. " +
                        "If this is a legitimate FileBatch, increase the limit using " +
                        "FileBatch.setMaxTotalUncompressedSize() or system property 'nd4j.filebatch.maxZipSize'");
            }

            if (totalRead > Integer.MAX_VALUE - 8) {
                throw new IOException("Entry '" + entryName + "' exceeds maximum supported size (2GB)");
            }

            bout.write(buffer, 0, bytesRead);
        }

        return bout.toByteArray();
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
