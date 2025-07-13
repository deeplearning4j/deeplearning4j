/*
 * ******************************************************************************
 * *
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * See the NOTICE file distributed with this work for additional
 * * information regarding copyright ownership.
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * * License for the specific language governing permissions and limitations
 * * under the License.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */

package org.nd4j.autodiff.samediff.serde;

import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Utility class for saving and loading SameDiff models to/from a single ZIP archive (.sdz).
 * This class acts as a wrapper around {@link SameDiffSerializer}, handling the
 * creation and extraction of ZIP archives containing the internal .sdnb shard files.
 */
@Slf4j
public class SDZSerializer {

    private static final String SDZ_EXTENSION = ".sdz";
    private static final String INTERNAL_SDNB_EXTENSION = ".sdnb";
    private static final String INTERNAL_BASE_NAME = "model";
    private static final byte[] SDNB_MAGIC = "SDNB".getBytes();
    private static final long SDNB_HEADER_SIZE = 32;

    /**
     * Saves the SameDiff model to a single ZIP archive (.sdz).
     * Internally uses SameDiffSerializer to create one or more .sdnb files in a
     * temporary directory, which are then zipped.
     *
     * @param sameDiff         The SameDiff instance to save.
     * @param outputZipFile    The path to the output ZIP file (should end with .sdz).
     * @param saveUpdaterState If true, include updater state in the internal shards.
     * @param metadata         Optional metadata passed to the internal SameDiffSerializer.
     * @throws IOException If saving or zipping fails.
     */
    @SneakyThrows
    public static void save(@NonNull SameDiff sameDiff, @NonNull File outputZipFile, boolean saveUpdaterState, Map<String, String> metadata) throws IOException {
        Preconditions.checkNotNull(sameDiff, "SameDiff instance cannot be null");
        Preconditions.checkNotNull(outputZipFile, "Output ZIP file path cannot be null.");

        Path tempDir = Files.createTempDirectory("sdz-serializer-save-");
        File tempDirFile = tempDir.toFile();
        log.info("Created temporary directory for saving: {}", tempDirFile.getAbsolutePath());

        try {
            File internalSavePath = new File(tempDirFile, INTERNAL_BASE_NAME);
            log.info("Saving internal .sdnb representation to temp directory: {}", internalSavePath.getAbsolutePath());
            SameDiffSerializer.saveAutoShard(sameDiff, internalSavePath, saveUpdaterState, metadata);

            List<File> filesToZip = collectValidSdnbFiles(tempDirFile);

            if (filesToZip.isEmpty()) {
                log.error("No valid SDNB files found in directory structure: {}", tempDirFile.getAbsolutePath());
                debugDirectoryContents(tempDirFile);
                throw new IOException("Failed to find any valid SDNB files after saving");
            }

            log.info("Found {} valid SDNB file(s) to add to ZIP archive", filesToZip.size());
            log.info("Creating final ZIP archive: {}", outputZipFile.getAbsolutePath());
            createZipArchive(outputZipFile, filesToZip);

        } finally {
            try {
                FileUtils.deleteDirectory(tempDirFile);
                log.debug("Cleaned up temporary save directory: {}", tempDirFile.getAbsolutePath());
            } catch (IOException e) {
                log.warn("Failed to delete temporary save directory: {}", tempDirFile, e);
            }
        }
        log.info("Successfully saved SameDiff model to ZIP archive: {}", outputZipFile.getAbsolutePath());
    }

    /**
     * Collects all valid SDNB files from the temporary directory.
     * Validates each file to ensure it has proper SDNB format before including.
     */
    private static List<File> collectValidSdnbFiles(File tempDirFile) {
        List<File> validFiles = new ArrayList<>();

        // Find all potential files recursively
        List<File> allFiles = new ArrayList<>();
        findAllFilesRecursively(tempDirFile, allFiles);

        // Validate each file
        for (File file : allFiles) {
            if (isValidSdnbFile(file)) {
                validFiles.add(file);
                log.debug("Added valid SDNB file: {}", file.getName());
            } else {
                log.debug("Skipping invalid file: {}", file.getName());
            }
        }

        return validFiles;
    }

    /**
     * Validates if a file is a properly formatted SDNB file.
     */
    private static boolean isValidSdnbFile(File file) {
        if (file == null || !file.exists() || !file.isFile()) {
            return false;
        }

        if (file.length() < SDNB_HEADER_SIZE) {
            return false;
        }

        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] magic = new byte[SDNB_MAGIC.length];
            int bytesRead = fis.read(magic);

            if (bytesRead != SDNB_MAGIC.length) {
                return false;
            }

            return Arrays.equals(SDNB_MAGIC, magic);
        } catch (IOException e) {
            log.debug("Error checking file magic for {}: {}", file.getName(), e.getMessage());
            return false;
        }
    }

    private static void findAllFilesRecursively(File directory, List<File> foundFiles) {
        if (!directory.isDirectory()) {
            return;
        }

        File[] files = directory.listFiles();
        if (files == null) {
            return;
        }

        for (File file : files) {
            if (file.isDirectory()) {
                findAllFilesRecursively(file, foundFiles);
            } else if (file.isFile()) {
                foundFiles.add(file);
            }
        }
    }

    private static void debugDirectoryContents(File directory) {
        log.error("Debug: Directory contents for {}", directory.getAbsolutePath());
        List<File> allFiles = new ArrayList<>();
        findAllFilesRecursively(directory, allFiles);

        for (File file : allFiles) {
            log.error("  File: {} (size: {}, valid SDNB: {})",
                    file.getName(), file.length(), isValidSdnbFile(file));
        }
    }

    /**
     * Loads a SameDiff model from a single ZIP archive (.sdz).
     * Extracts the internal .sdnb shard files to a temporary directory and then uses
     * the original SameDiffSerializer to load the model from those files.
     *
     * @param modelZipFile     Path to the .sdz model archive file.
     * @param loadUpdaterState If true, attempt to load updater state from the internal shards.
     * @return The loaded SameDiff instance.
     * @throws IOException If the file is not a valid ZIP, extraction fails, or loading fails.
     */
    @SneakyThrows
    public static SameDiff load(@NonNull File modelZipFile, boolean loadUpdaterState) throws IOException {
        Preconditions.checkNotNull(modelZipFile, "Model ZIP file path cannot be null.");
        Preconditions.checkArgument(modelZipFile.exists() && modelZipFile.isFile(),
                "Model ZIP file does not exist or is not a file: %s", modelZipFile.getAbsolutePath());

        if (!isZipFile(modelZipFile)) {
            throw new IOException("File is not a valid ZIP archive: " + modelZipFile.getAbsolutePath());
        }

        Path tempDir = Files.createTempDirectory("sdz-serializer-load-");
        log.debug("Using temporary directory for ZIP extraction: {}", tempDir);
        SameDiff loadedSameDiff;

        try {
            log.info("Extracting ZIP archive '{}' to temporary directory...", modelZipFile.getName());
            extractZip(modelZipFile, tempDir.toFile());

            File loadPath = determineLoadPath(tempDir.toFile());
            if (loadPath == null) {
                throw new IOException("Could not determine the internal model file path after extracting ZIP archive to: " + tempDir);
            }
            log.info("Determined internal load path: {}", loadPath.getAbsolutePath());

            log.info("Loading model using SameDiffSerializer from extracted files...");
            loadedSameDiff = SameDiffSerializer.load(loadPath, loadUpdaterState);

        } finally {
            try {
                FileUtils.deleteDirectory(tempDir.toFile());
                log.debug("Cleaned up temporary load directory: {}", tempDir);
            } catch (IOException e) {
                log.warn("Failed to delete temporary load directory: {}", tempDir, e);
            }
        }

        if (loadedSameDiff == null) {
            throw new IOException("SameDiffSerializer.load returned null after loading from extracted files.");
        }
        log.info("Successfully loaded SameDiff model from ZIP archive: {}", modelZipFile.getAbsolutePath());
        return loadedSameDiff;
    }

    /**
     * Determines the correct file path within the extraction directory
     * to pass to the original SameDiffSerializer.load method.
     * Prioritizes valid SDNB files over naming conventions.
     */
    private static File determineLoadPath(File extractedDir) {
        File[] allFiles = extractedDir.listFiles();
        if (allFiles == null || allFiles.length == 0) {
            log.error("No files found in extraction directory: {}", extractedDir.getAbsolutePath());
            return null;
        }

        log.debug("Files found in extraction directory: {}", Arrays.toString(allFiles));

        // First, find all valid SDNB files
        List<File> validSdnbFiles = new ArrayList<>();
        for (File file : allFiles) {
            if (isValidSdnbFile(file)) {
                validSdnbFiles.add(file);
            }
        }

        if (validSdnbFiles.isEmpty()) {
            log.error("No valid SDNB files found in extraction directory");
            debugDirectoryContents(extractedDir);
            return null;
        }

        // If only one valid file, use it
        if (validSdnbFiles.size() == 1) {
            log.debug("Found single valid SDNB file: {}", validSdnbFiles.get(0).getName());
            return validSdnbFiles.get(0);
        }

        // Multiple files - look for preferred patterns

        // Try single file with extension first
        File singleFile = new File(extractedDir, INTERNAL_BASE_NAME + INTERNAL_SDNB_EXTENSION);
        if (isValidSdnbFile(singleFile)) {
            log.debug("Using single file with extension: {}", singleFile.getName());
            return singleFile;
        }

        // Try single file without extension
        File noExtensionFile = new File(extractedDir, INTERNAL_BASE_NAME);
        if (isValidSdnbFile(noExtensionFile)) {
            log.debug("Using single file without extension: {}", noExtensionFile.getName());
            return noExtensionFile;
        }

        // Check for sharded files with extension
        File[] shardFiles = extractedDir.listFiles((dir, name) ->
                name.matches(INTERNAL_BASE_NAME + "\\.shard\\d+-of-\\d+\\" + INTERNAL_SDNB_EXTENSION + "$"));
        if (shardFiles != null && shardFiles.length > 0) {
            // Validate at least one shard file
            for (File shardFile : shardFiles) {
                if (isValidSdnbFile(shardFile)) {
                    log.debug("Found valid sharded files with extension, using base: {}", INTERNAL_BASE_NAME);
                    return new File(extractedDir, INTERNAL_BASE_NAME);
                }
            }
        }

        // Check for sharded files without extension
        File[] shardedFilesWithoutExtension = extractedDir.listFiles((dir, name) ->
                name.matches(INTERNAL_BASE_NAME + "\\.shard\\d+-of-\\d+$"));
        if (shardedFilesWithoutExtension != null && shardedFilesWithoutExtension.length > 0) {
            // Validate at least one shard file
            for (File shardFile : shardedFilesWithoutExtension) {
                if (isValidSdnbFile(shardFile)) {
                    log.debug("Found valid sharded files without extension, using base: {}", INTERNAL_BASE_NAME);
                    return new File(extractedDir, INTERNAL_BASE_NAME);
                }
            }
        }

        // If we have valid files but no preferred pattern, use the first valid one
        File firstValid = validSdnbFiles.get(0);
        log.warn("No preferred file pattern found, using first valid SDNB file: {}", firstValid.getName());
        return firstValid;
    }

    private static void createZipArchive(File outputZipFile, List<File> filesToAdd) throws IOException {
        List<File> existingFiles = new ArrayList<>();
        for (File file : filesToAdd) {
            if (file.exists() && file.isFile()) {
                existingFiles.add(file);
            } else {
                log.warn("File does not exist or is not a regular file: {}", file.getAbsolutePath());
            }
        }

        if (existingFiles.isEmpty()) {
            throw new IOException("No valid files to add to the ZIP archive");
        }

        File parent = outputZipFile.getParentFile();
        if(parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Could not create parent directory for ZIP file: " + parent.getAbsolutePath());
        }

        try (FileOutputStream fos = new FileOutputStream(outputZipFile);
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             ZipOutputStream zos = new ZipOutputStream(bos)) {

            for (File file : existingFiles) {
                if (!file.exists() || !file.isFile()) {
                    log.warn("File disappeared between initial check and ZIP addition: {}", file.getAbsolutePath());
                    continue;
                }

                String entryName = file.getName();
                log.debug("Adding ZIP entry: {} from {}", entryName, file.getAbsolutePath());
                ZipEntry zipEntry = new ZipEntry(entryName);
                zos.putNextEntry(zipEntry);

                try (FileInputStream fis = new FileInputStream(file);
                     BufferedInputStream bis = new BufferedInputStream(fis)) {
                    IOUtils.copy(bis, zos);
                }
                zos.closeEntry();
            }

            zos.flush();
        } catch (IOException e) {
            try {
                if(outputZipFile.exists()) {
                    outputZipFile.delete();
                }
            } catch (Exception ignored) {}
            throw new IOException("Failed to create ZIP archive: " + outputZipFile.getAbsolutePath(), e);
        }
    }

    private static void extractZip(File zipFile, File targetDir) throws IOException {
        String canonicalTargetPath = targetDir.getCanonicalPath();
        if (!targetDir.exists() && !targetDir.mkdirs()) {
            throw new IOException("Could not create target directory for extraction: " + targetDir.getAbsolutePath());
        }

        byte[] buffer = new byte[8192];

        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile)))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                File entryFile = new File(targetDir, entry.getName());

                String canonicalEntryPath = entryFile.getCanonicalPath();
                if (!canonicalEntryPath.startsWith(canonicalTargetPath + File.separator) && !canonicalEntryPath.equals(canonicalTargetPath)) {
                    throw new IOException("Zip Slip vulnerability detected! Entry is outside of the target dir: " + entry.getName());
                }

                if (entry.isDirectory()) {
                    if (!entryFile.isDirectory() && !entryFile.mkdirs()) {
                        throw new IOException("Failed to create directory within ZIP structure: " + entryFile.getAbsolutePath());
                    }
                } else {
                    File parent = entryFile.getParentFile();
                    if (!parent.isDirectory() && !parent.mkdirs()) {
                        throw new IOException("Failed to create parent directory for extracted file: " + parent.getAbsolutePath());
                    }

                    try (FileOutputStream fos = new FileOutputStream(entryFile);
                         BufferedOutputStream bos = new BufferedOutputStream(fos)) {
                        int len;
                        while((len = zis.read(buffer)) > 0){
                            bos.write(buffer, 0, len);
                        }
                    }
                }
                zis.closeEntry();
            }
        } catch (IOException e) {
            throw new IOException("Failed during ZIP extraction from " + zipFile.getAbsolutePath() + " to " + targetDir.getAbsolutePath(), e);
        }
        log.debug("Finished extracting ZIP archive to {}", targetDir.getAbsolutePath());
    }

    private static boolean isZipFile(File file) {
        if (file == null || !file.exists() || !file.isFile() || file.length() < 4) {
            return false;
        }

        byte[] magic = new byte[4];
        try (FileInputStream fis = new FileInputStream(file);
             DataInputStream dis = new DataInputStream(fis)) {
            dis.readFully(magic);
        } catch (IOException e) {
            return false;
        }

        return magic[0] == 0x50 && magic[1] == 0x4b && magic[2] == 0x03 && magic[3] == 0x04;
    }
}