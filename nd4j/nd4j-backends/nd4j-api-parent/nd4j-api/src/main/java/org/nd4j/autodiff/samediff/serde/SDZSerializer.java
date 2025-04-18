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
    private static final String INTERNAL_BASE_NAME = "model"; // Consistent name used inside the ZIP

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

        // Create a dedicated temporary directory for saving the internal files
        Path tempDir = Files.createTempDirectory("sdz-serializer-save-");
        File tempDirFile = tempDir.toFile();
        log.info("Created temporary directory for saving: {}", tempDirFile.getAbsolutePath());

        try {
            // 1. Save using the original serializer to the temporary directory
            File internalSavePath = new File(tempDirFile, INTERNAL_BASE_NAME); // Use consistent base name in temp dir
            log.info("Saving internal .sdnb representation to temp directory: {}", internalSavePath.getAbsolutePath());
            SameDiffSerializer.saveAutoShard(sameDiff, internalSavePath, saveUpdaterState, metadata);

            // 2. Create a list to collect all files we need to add to the zip
            List<File> filesToZip = new ArrayList<>();

            // Check if there are directly .sdnb files in the temp directory
            File[] topLevelFiles = tempDirFile.listFiles((dir, name) ->
                    name.startsWith(INTERNAL_BASE_NAME) && (name.endsWith(INTERNAL_SDNB_EXTENSION) ||
                            name.matches(INTERNAL_BASE_NAME + "\\.shard\\d+-of-\\d+" + INTERNAL_SDNB_EXTENSION + "$")));

            if (topLevelFiles != null && topLevelFiles.length > 0) {
                filesToZip.addAll(Arrays.asList(topLevelFiles));
                log.info("Found {} .sdnb files at top level", topLevelFiles.length);
            }

            // Also look for the file with no extension (the "model" file that we've seen in the logs)
            File modelFile = new File(tempDirFile, INTERNAL_BASE_NAME);
            if (modelFile.exists() && modelFile.isFile()) {
                log.info("Found model file without extension: {}", modelFile.getAbsolutePath());
                filesToZip.add(modelFile);
            }

            // Check for sharded files without extension
            File[] shardedFilesWithoutExtension = tempDirFile.listFiles((dir, name) ->
                    name.matches(INTERNAL_BASE_NAME + "\\.shard\\d+-of-\\d+$"));

            if (shardedFilesWithoutExtension != null && shardedFilesWithoutExtension.length > 0) {
                filesToZip.addAll(Arrays.asList(shardedFilesWithoutExtension));
                log.info("Found {} sharded files without extension", shardedFilesWithoutExtension.length);
            }

            // If still no files found, check inside the model directory if it exists
            if (filesToZip.isEmpty()) {
                File modelDir = new File(tempDirFile, INTERNAL_BASE_NAME);
                if (modelDir.exists() && modelDir.isDirectory()) {
                    log.info("Found 'model' directory, checking for files inside");
                    File[] modelDirFiles = modelDir.listFiles();
                    if (modelDirFiles != null && modelDirFiles.length > 0) {
                        filesToZip.addAll(Arrays.asList(modelDirFiles));
                        log.info("Found {} files inside model directory", modelDirFiles.length);
                    }
                }
            }

            // If still no files found, perform recursive search
            if (filesToZip.isEmpty()) {
                log.info("No files found in standard locations, performing recursive search");
                findAllFilesRecursively(tempDirFile, filesToZip);
                log.info("Recursive search found {} files", filesToZip.size());
            }

            if (filesToZip.isEmpty()) {
                log.error("No files found anywhere in directory structure: {}", tempDirFile.getAbsolutePath());
                // List all files for debugging
                List<File> allFiles = new ArrayList<>();
                findAllFilesRecursively(tempDirFile, allFiles);
                log.info("All files found in directory structure: {}", allFiles);
                throw new IOException("Failed to find any files after saving");
            }

            log.info("Found {} total file(s) to add to ZIP archive", filesToZip.size());

            // 3. Create the final ZIP archive
            log.info("Creating final ZIP archive: {}", outputZipFile.getAbsolutePath());
            createZipArchive(outputZipFile, filesToZip);

        } finally {
            // 4. Clean up the temporary directory
            try {
                FileUtils.deleteDirectory(tempDirFile);
                log.debug("Cleaned up temporary save directory: {}", tempDirFile.getAbsolutePath());
            } catch (IOException e) {
                log.warn("Failed to delete temporary save directory: {}", tempDirFile, e);
            }
        }
        log.info("Successfully saved SameDiff model to ZIP archive: {}", outputZipFile.getAbsolutePath());
    }

    // Helper method to recursively find all files
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
    @SneakyThrows // For Files.createTempDirectory
    public static SameDiff load(@NonNull File modelZipFile, boolean loadUpdaterState) throws IOException {
        Preconditions.checkNotNull(modelZipFile, "Model ZIP file path cannot be null.");
        Preconditions.checkArgument(modelZipFile.exists() && modelZipFile.isFile(), "Model ZIP file does not exist or is not a file: %s", modelZipFile.getAbsolutePath());

        if (!isZipFile(modelZipFile)) {
            throw new IOException("File is not a valid ZIP archive: " + modelZipFile.getAbsolutePath());
        }

        Path tempDir = Files.createTempDirectory("sdz-serializer-load-");
        log.debug("Using temporary directory for ZIP extraction: {}", tempDir);
        SameDiff loadedSameDiff;

        try {
            // 1. Extract ZIP contents to the temporary directory
            log.info("Extracting ZIP archive '{}' to temporary directory...", modelZipFile.getName());
            extractZip(modelZipFile, tempDir.toFile());

            // 2. Determine the path needed by the original SameDiffSerializer.load
            File loadPath = determineLoadPath(tempDir.toFile());
            if (loadPath == null) {
                throw new IOException("Could not determine the internal model file path after extracting ZIP archive to: " + tempDir);
            }
            log.info("Determined internal load path: {}", loadPath.getAbsolutePath());


            // 3. Load using the original serializer from the extracted files
            log.info("Loading model using SameDiffSerializer from extracted files...");
            loadedSameDiff = SameDiffSerializer.load(loadPath, loadUpdaterState);

        } finally {
            // 4. Clean up the temporary directory
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


    // --- Private Helper Methods ---

    /**
     * Determines the correct file path within the extraction directory
     * to pass to the original SameDiffSerializer.load method.
     * It looks for either a single "model.sdnb" file or the base "model"
     * name if sharded files like "model.shard0-of-N.sdnb" exist.
     */
    private static File determineLoadPath(File extractedDir) {
        // First try the original logic looking for .sdnb files
        File singleFile = new File(extractedDir, INTERNAL_BASE_NAME + INTERNAL_SDNB_EXTENSION);
        if (singleFile.exists() && singleFile.isFile()) {
            log.debug("Determined load path is single file with extension: {}", singleFile.getName());
            return singleFile; // Path to the single .sdnb file
        }

        // Check for sharded files with extension
        File[] shardFiles = extractedDir.listFiles((dir, name) ->
                name.matches(INTERNAL_BASE_NAME + "\\.shard\\d+-of-\\d+\\" + INTERNAL_SDNB_EXTENSION + "$"));
        if (shardFiles != null && shardFiles.length > 0) {
            log.debug("Determined load path is sharded base with extension: {}", INTERNAL_BASE_NAME);
            return new File(extractedDir, INTERNAL_BASE_NAME);
        }

        // NEW CODE: Also check for file without extension
        File noExtensionFile = new File(extractedDir, INTERNAL_BASE_NAME);
        if (noExtensionFile.exists() && noExtensionFile.isFile()) {
            log.debug("Determined load path is file without extension: {}", noExtensionFile.getName());
            return noExtensionFile;
        }

        // NEW CODE: Check for sharded files without extension
        File[] shardedFilesWithoutExtension = extractedDir.listFiles((dir, name) ->
                name.matches(INTERNAL_BASE_NAME + "\\.shard\\d+-of-\\d+$"));
        if (shardedFilesWithoutExtension != null && shardedFilesWithoutExtension.length > 0) {
            log.debug("Determined load path is sharded base without extension: {}", INTERNAL_BASE_NAME);
            return new File(extractedDir, INTERNAL_BASE_NAME);
        }

        // Check if model is a directory (another possible structure)
        File modelDir = new File(extractedDir, INTERNAL_BASE_NAME);
        if (modelDir.exists() && modelDir.isDirectory()) {
            log.debug("Found 'model' directory - will use as base path");
            return modelDir;
        }

        // Log if no suitable files found
        log.error("Could not find expected '{}' or '{}' or '{}' file/pattern in extraction directory: {}",
                INTERNAL_BASE_NAME + INTERNAL_SDNB_EXTENSION,
                INTERNAL_BASE_NAME + ".shard*-of-*.sdnb",
                INTERNAL_BASE_NAME,
                extractedDir.getAbsolutePath());

        // Debug: List all files in the extraction directory
        File[] allFiles = extractedDir.listFiles();
        if (allFiles != null && allFiles.length > 0) {
            log.info("Files found in extraction directory: {}", Arrays.toString(allFiles));
        } else {
            log.info("No files found in extraction directory");
        }

        return null; // Indicate failure
    }


    private static void createZipArchive(File outputZipFile, List<File> filesToAdd) throws IOException {
        // Check file existence upfront
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

        // Ensure parent directory exists
        File parent = outputZipFile.getParentFile();
        if(parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Could not create parent directory for ZIP file: " + parent.getAbsolutePath());
        }

        try (FileOutputStream fos = new FileOutputStream(outputZipFile);
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             ZipOutputStream zos = new ZipOutputStream(bos)) {

            for (File file : existingFiles) {
                // Double-check file still exists right before adding
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

            // Make sure everything is written
            zos.flush();
        } catch (IOException e) {
            // Attempt to delete potentially corrupted zip file on failure
            try { if(outputZipFile.exists()) outputZipFile.delete(); } catch (Exception ignored) {}
            throw new IOException("Failed to create ZIP archive: " + outputZipFile.getAbsolutePath(), e);
        }
    }

    /** Extracts all entries from a ZIP archive to a target directory. */
    private static void extractZip(File zipFile, File targetDir) throws IOException {
        // Basic Zip Slip prevention: Ensure target directory is indeed the intended one
        String canonicalTargetPath = targetDir.getCanonicalPath();
        if (!targetDir.exists() && !targetDir.mkdirs()) {
            throw new IOException("Could not create target directory for extraction: " + targetDir.getAbsolutePath());
        }

        byte[] buffer = new byte[8192]; // Reusable buffer for copying

        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile)))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                File entryFile = new File(targetDir, entry.getName());

                // Zip Slip Check: Ensure extracted file stays within the target directory
                String canonicalEntryPath = entryFile.getCanonicalPath();
                if (!canonicalEntryPath.startsWith(canonicalTargetPath + File.separator) && !canonicalEntryPath.equals(canonicalTargetPath)) {
                    throw new IOException("Zip Slip vulnerability detected! Entry is outside of the target dir: " + entry.getName());
                }

                if (entry.isDirectory()) {
                    if (!entryFile.isDirectory() && !entryFile.mkdirs()) {
                        throw new IOException("Failed to create directory within ZIP structure: " + entryFile.getAbsolutePath());
                    }
                } else {
                    // Ensure parent directory exists for the file
                    File parent = entryFile.getParentFile();
                    if (!parent.isDirectory() && !parent.mkdirs()) {
                        throw new IOException("Failed to create parent directory for extracted file: " + parent.getAbsolutePath());
                    }

                    // Extract file content
                    try (FileOutputStream fos = new FileOutputStream(entryFile); BufferedOutputStream bos = new BufferedOutputStream(fos)) {
                        int len;
                        while((len = zis.read(buffer)) > 0){
                            bos.write(buffer, 0, len);
                        }
                    }
                }
                zis.closeEntry();
            }
        } catch (IOException e) {
            // Clean up potentially partially extracted files? Difficult to do reliably.
            throw new IOException("Failed during ZIP extraction from " + zipFile.getAbsolutePath() + " to " + targetDir.getAbsolutePath(), e);
        }
        log.debug("Finished extracting ZIP archive to {}", targetDir.getAbsolutePath());
    }

    /** Checks if a file is likely a ZIP archive based on magic number. */
    private static boolean isZipFile(File file) {
        if (file == null || !file.exists() || !file.isFile() || file.length() < 4) return false;
        byte[] magic = new byte[4];
        try (FileInputStream fis = new FileInputStream(file); DataInputStream dis = new DataInputStream(fis)) { dis.readFully(magic); } catch (IOException e) { return false; }
        // Standard ZIP magic number PK\03\04
        return magic[0] == 0x50 && magic[1] == 0x4b && magic[2] == 0x03 && magic[3] == 0x04;
    }

}