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
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.config.ND4JSystemProperties;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.zip.CRC32;
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
     * Maximum total decompressed size allowed when extracting ZIP files (default: 10GB).
     * This limit protects against zip bomb attacks that could exhaust disk space.
     * Can be overridden via system property "nd4j.sdz.maxZipSize" (in bytes).
     */
    public static final long DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE = 10L * 1024L * 1024L * 1024L; // 10GB

    /**
     * Maximum allowed compression ratio (uncompressed/compressed size).
     * Default is 100:1. Can be overridden via system property "nd4j.sdz.maxCompressionRatio".
     */
    public static final double DEFAULT_MAX_COMPRESSION_RATIO = 100.0;

    /**
     * Maximum number of entries allowed in a model ZIP file.
     * Can be overridden via system property "nd4j.sdz.maxZipEntries".
     */
    public static final int DEFAULT_MAX_ZIP_ENTRIES = 1000;

    private static long maxTotalUncompressedSize = getConfiguredMaxSize();
    private static double maxCompressionRatio = getConfiguredMaxRatio();
    private static int maxZipEntries = getConfiguredMaxEntries();

    private static long getConfiguredMaxSize() {
        String prop = System.getProperty("nd4j.sdz.maxZipSize");
        if (prop != null) {
            try {
                return Long.parseLong(prop);
            } catch (NumberFormatException e) {
                log.warn("Invalid value for nd4j.sdz.maxZipSize: {}, using default", prop);
            }
        }
        return DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE;
    }

    private static double getConfiguredMaxRatio() {
        String prop = System.getProperty("nd4j.sdz.maxCompressionRatio");
        if (prop != null) {
            try {
                return Double.parseDouble(prop);
            } catch (NumberFormatException e) {
                log.warn("Invalid value for nd4j.sdz.maxCompressionRatio: {}, using default", prop);
            }
        }
        return DEFAULT_MAX_COMPRESSION_RATIO;
    }

    private static int getConfiguredMaxEntries() {
        String prop = System.getProperty("nd4j.sdz.maxZipEntries");
        if (prop != null) {
            try {
                return Integer.parseInt(prop);
            } catch (NumberFormatException e) {
                log.warn("Invalid value for nd4j.sdz.maxZipEntries: {}, using default", prop);
            }
        }
        return DEFAULT_MAX_ZIP_ENTRIES;
    }

    /**
     * Set the maximum total uncompressed size allowed when extracting SDZ files.
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
     * Set the maximum number of entries allowed in SDZ files.
     * @param maxEntries Maximum entries (must be positive)
     */
    public static void setMaxZipEntries(int maxEntries) {
        if (maxEntries <= 0) throw new IllegalArgumentException("Max entries must be positive, got " + maxEntries);
        maxZipEntries = maxEntries;
    }

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
     * Saves the SameDiff model to a ZIP archive with graph optimization applied.
     * The model is first optimized using the default optimization passes, then saved.
     * This produces a more efficient model for inference.
     *
     * @param sameDiff         The SameDiff instance to save.
     * @param outputZipFile    The path to the output ZIP file (should end with .sdz).
     * @param saveUpdaterState If true, include updater state in the internal shards.
     * @param metadata         Optional metadata passed to the internal SameDiffSerializer.
     * @param requiredOutputs  The output variable names that must be preserved during optimization.
     *                         These are the outputs you will use for inference.
     * @throws IOException If saving or zipping fails.
     */
    @SneakyThrows
    public static void saveOptimized(@NonNull SameDiff sameDiff, @NonNull File outputZipFile,
                                     boolean saveUpdaterState, Map<String, String> metadata,
                                     @NonNull List<String> requiredOutputs) throws IOException {
        saveOptimized(sameDiff, outputZipFile, saveUpdaterState, metadata, requiredOutputs,
                GraphOptimizer.defaultOptimizations());
    }

    /**
     * Saves the SameDiff model to a ZIP archive with custom graph optimizations applied.
     * The model is first optimized using the provided optimization passes, then saved.
     *
     * @param sameDiff         The SameDiff instance to save.
     * @param outputZipFile    The path to the output ZIP file (should end with .sdz).
     * @param saveUpdaterState If true, include updater state in the internal shards.
     * @param metadata         Optional metadata passed to the internal SameDiffSerializer.
     * @param requiredOutputs  The output variable names that must be preserved during optimization.
     * @param optimizations    The list of optimization passes to apply.
     * @throws IOException If saving or zipping fails.
     */
    @SneakyThrows
    public static void saveOptimized(@NonNull SameDiff sameDiff, @NonNull File outputZipFile,
                                     boolean saveUpdaterState, Map<String, String> metadata,
                                     @NonNull List<String> requiredOutputs,
                                     @NonNull List<OptimizerSet> optimizations) throws IOException {
        Preconditions.checkNotNull(sameDiff, "SameDiff instance cannot be null");
        Preconditions.checkNotNull(outputZipFile, "Output ZIP file path cannot be null.");
        Preconditions.checkNotNull(requiredOutputs, "Required outputs cannot be null");
        Preconditions.checkArgument(!requiredOutputs.isEmpty(), "At least one required output must be specified");

        log.info("Applying graph optimizations before saving...");
        log.info("Required outputs: {}", requiredOutputs);
        log.info("Number of optimization passes: {}", optimizations.size());

        // Apply graph optimization - this creates a new optimized SameDiff instance
        SameDiff optimizedSd = GraphOptimizer.optimize(sameDiff, requiredOutputs, optimizations);

        int originalOps = sameDiff.getOps().size();
        int optimizedOps = optimizedSd.getOps().size();
        int originalVars = sameDiff.getVariables().size();
        int optimizedVars = optimizedSd.getVariables().size();

        log.info("Optimization complete. Original ops: {}, Optimized ops: {} (reduced by {})",
                originalOps, optimizedOps, originalOps - optimizedOps);
        log.info("Original variables: {}, Optimized variables: {} (reduced by {})",
                originalVars, optimizedVars, originalVars - optimizedVars);

        // Add optimization metadata
        Map<String, String> fullMetadata = new java.util.HashMap<>();
        if (metadata != null) {
            fullMetadata.putAll(metadata);
        }
        fullMetadata.put("optimized", "true");
        fullMetadata.put("optimization_timestamp", String.valueOf(System.currentTimeMillis()));
        fullMetadata.put("original_ops", String.valueOf(originalOps));
        fullMetadata.put("optimized_ops", String.valueOf(optimizedOps));
        fullMetadata.put("original_variables", String.valueOf(originalVars));
        fullMetadata.put("optimized_variables", String.valueOf(optimizedVars));
        fullMetadata.put("required_outputs", String.join(",", requiredOutputs));

        // Save the optimized model with metadata
        save(optimizedSd, outputZipFile, saveUpdaterState, fullMetadata);
    }

    /**
     * Convenience method to save an optimized model with a single output.
     *
     * @param sameDiff         The SameDiff instance to save.
     * @param outputZipFile    The path to the output ZIP file.
     * @param saveUpdaterState If true, include updater state.
     * @param requiredOutput   The single output variable name to preserve.
     * @throws IOException If saving fails.
     */
    @SneakyThrows
    public static void saveOptimized(@NonNull SameDiff sameDiff, @NonNull File outputZipFile,
                                     boolean saveUpdaterState, @NonNull String requiredOutput) throws IOException {
        saveOptimized(sameDiff, outputZipFile, saveUpdaterState, null,
                java.util.Collections.singletonList(requiredOutput));
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

        // Disable DSP and CUDA graphs during model loading. Loading model constants to GPU
        // is peak memory usage — DSP compilation and CUDA graph capture add memory that causes OOM.
        boolean dspWasEnabled = InferenceSession.isDynamicShapePlanEnabled();
        String prevCudaGraphs = System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
        InferenceSession.setDynamicShapePlanEnabled(false);
        System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, "false");

        long loadStart = System.currentTimeMillis();
        SameDiff loadedSameDiff;

        // Use ZipFile for random access to entries. With STORED compression, getInputStream()
        // reads directly from the underlying file without decompression — much faster than
        // ZipInputStream which must decompress sequentially.
        Path tempDir = null;
        try {
            // Extract SDNB entries to temp files using ZipFile (random access, large buffer)
            tempDir = Files.createTempDirectory("sdz-serializer-load-");
            File tempDirFile = tempDir.toFile();

            try (java.util.zip.ZipFile zipFile = new java.util.zip.ZipFile(modelZipFile)) {
                java.util.Enumeration<? extends ZipEntry> entries = zipFile.entries();
                byte[] extractBuffer = new byte[1024 * 1024]; // 1MB buffer for extraction
                int entryCount = 0;

                while (entries.hasMoreElements()) {
                    ZipEntry entry = entries.nextElement();
                    entryCount++;
                    if (entryCount > maxZipEntries) {
                        throw new IOException("Too many ZIP entries: " + entryCount + ", max: " + maxZipEntries);
                    }
                    if (entry.isDirectory()) continue;

                    // Zip Slip protection
                    File entryFile = new File(tempDirFile, entry.getName());
                    if (!entryFile.getCanonicalPath().startsWith(tempDirFile.getCanonicalPath() + File.separator)) {
                        throw new IOException("Zip Slip: " + entry.getName());
                    }

                    // Extract using ZipFile.getInputStream (random access, no sequential decompression)
                    try (InputStream zis = zipFile.getInputStream(entry);
                         FileOutputStream fos = new FileOutputStream(entryFile);
                         BufferedOutputStream bos = new BufferedOutputStream(fos, 1024 * 1024)) {
                        long totalWritten = 0;
                        int len;
                        while ((len = zis.read(extractBuffer)) > 0) {
                            totalWritten += len;
                            if (totalWritten > maxTotalUncompressedSize) {
                                throw new IOException("Uncompressed size exceeds limit: " + maxTotalUncompressedSize);
                            }
                            bos.write(extractBuffer, 0, len);
                        }
                    }
                }
            }

            File loadPath = determineLoadPath(tempDirFile);
            if (loadPath == null) {
                throw new IOException("No valid SDNB files found in ZIP: " + modelZipFile.getAbsolutePath());
            }

            loadedSameDiff = SameDiffSerializer.load(loadPath, loadUpdaterState);

        } finally {
            if (tempDir != null) {
                try {
                    FileUtils.deleteDirectory(tempDir.toFile());
                } catch (IOException e) {
                    log.warn("Failed to delete temporary load directory: {}", tempDir, e);
                }
            }
            // Restore DSP and CUDA graph settings
            InferenceSession.setDynamicShapePlanEnabled(dspWasEnabled);
            if (prevCudaGraphs != null) {
                System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, prevCudaGraphs);
            } else {
                System.clearProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
            }
        }

        if (loadedSameDiff == null) {
            throw new IOException("SameDiffSerializer.load returned null after loading from extracted files.");
        }
        long loadMs = System.currentTimeMillis() - loadStart;
        log.info("Loaded SameDiff model from SDZ in {}ms: {}", loadMs, modelZipFile.getName());
        return loadedSameDiff;
    }

    /**
     * Loads a SameDiff model from a ZIP archive with intelligent background transfer monitoring.
     * This overload uses ModelLoadingContext for optimized model loading:
     * <ul>
     *   <li>Pre-analyzes model size from manifest</li>
     *   <li>Selects optimal target device (GPU/CPU) based on available memory</li>
     *   <li>Schedules background async transfers for better performance</li>
     *   <li>Logs transfer metrics and statistics on close</li>
     * </ul>
     *
     * @param modelZipFile     Path to the .sdz model archive file.
     * @param loadUpdaterState If true, attempt to load updater state from the internal shards.
     * @param context          The ModelLoadingContext for optimized loading and transfer monitoring.
     * @return The loaded SameDiff instance.
     * @throws IOException If the file is not a valid ZIP, extraction fails, or loading fails.
     */
    @SneakyThrows
    public static SameDiff load(@NonNull File modelZipFile, boolean loadUpdaterState, @NonNull ModelLoadingContext context) throws IOException {
        Preconditions.checkNotNull(modelZipFile, "Model ZIP file path cannot be null.");
        Preconditions.checkNotNull(context, "ModelLoadingContext cannot be null.");
        Preconditions.checkArgument(modelZipFile.exists() && modelZipFile.isFile(),
                "Model ZIP file does not exist or is not a file: %s", modelZipFile.getAbsolutePath());

        if (!isZipFile(modelZipFile)) {
            throw new IOException("File is not a valid ZIP archive: " + modelZipFile.getAbsolutePath());
        }

        // Disable DSP and CUDA graphs during model loading. Loading model constants to GPU
        // is peak memory usage — DSP compilation and CUDA graph capture add memory that causes OOM.
        boolean dspWasEnabled = InferenceSession.isDynamicShapePlanEnabled();
        String prevCudaGraphs = System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
        InferenceSession.setDynamicShapePlanEnabled(false);
        System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, "false");

        log.info("Loading model with intelligent context: target={}, totalSize={}",
                context.getTargetDevice().getDeviceId(),
                context.getSizeInfo().toSummaryString());

        long loadStart = System.currentTimeMillis();
        Path tempDir = null;
        SameDiff loadedSameDiff;

        try {
            // Extract using ZipFile for random access (same as non-context load)
            tempDir = Files.createTempDirectory("sdz-serializer-load-");
            File tempDirFile = tempDir.toFile();

            try (java.util.zip.ZipFile zipFile = new java.util.zip.ZipFile(modelZipFile)) {
                java.util.Enumeration<? extends ZipEntry> entries = zipFile.entries();
                byte[] extractBuffer = new byte[1024 * 1024];
                int entryCount = 0;

                while (entries.hasMoreElements()) {
                    ZipEntry entry = entries.nextElement();
                    entryCount++;
                    if (entryCount > maxZipEntries) {
                        throw new IOException("Too many ZIP entries: " + entryCount);
                    }
                    if (entry.isDirectory()) continue;

                    File entryFile = new File(tempDirFile, entry.getName());
                    if (!entryFile.getCanonicalPath().startsWith(tempDirFile.getCanonicalPath() + File.separator)) {
                        throw new IOException("Zip Slip: " + entry.getName());
                    }

                    try (InputStream zis = zipFile.getInputStream(entry);
                         FileOutputStream fos = new FileOutputStream(entryFile);
                         BufferedOutputStream bos = new BufferedOutputStream(fos, 1024 * 1024)) {
                        long totalWritten = 0;
                        int len;
                        while ((len = zis.read(extractBuffer)) > 0) {
                            totalWritten += len;
                            if (totalWritten > maxTotalUncompressedSize) {
                                throw new IOException("Uncompressed size exceeds limit");
                            }
                            bos.write(extractBuffer, 0, len);
                        }
                    }
                }
            }

            File loadPath = determineLoadPath(tempDirFile);
            if (loadPath == null) {
                throw new IOException("No valid SDNB files found in ZIP: " + modelZipFile.getAbsolutePath());
            }

            loadedSameDiff = SameDiffSerializer.load(loadPath, loadUpdaterState, context);

        } finally {
            if (tempDir != null) {
                try {
                    FileUtils.deleteDirectory(tempDir.toFile());
                } catch (IOException e) {
                    log.warn("Failed to delete temporary load directory: {}", tempDir, e);
                }
            }
            // Restore DSP and CUDA graph settings
            InferenceSession.setDynamicShapePlanEnabled(dspWasEnabled);
            if (prevCudaGraphs != null) {
                System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, prevCudaGraphs);
            } else {
                System.clearProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
            }
        }

        if (loadedSameDiff == null) {
            throw new IOException("SameDiffSerializer.load returned null after loading from extracted files.");
        }
        long loadMs = System.currentTimeMillis() - loadStart;
        log.info("Loaded SameDiff model from SDZ with context in {}ms: {}", loadMs, modelZipFile.getName());
        return loadedSameDiff;
    }

    /**
     * Loads a SameDiff model with automatic intelligent loading.
     * Creates a ModelLoadingContext automatically, analyzes the model, and selects
     * the optimal device for loading.
     *
     * @param modelZipFile     Path to the .sdz model archive file.
     * @param loadUpdaterState If true, attempt to load updater state from the internal shards.
     * @param useIntelligentLoading If true, uses ModelLoadingContext for optimized loading.
     * @return The loaded SameDiff instance.
     * @throws IOException If loading fails.
     */
    @SneakyThrows
    public static SameDiff load(@NonNull File modelZipFile, boolean loadUpdaterState, boolean useIntelligentLoading) throws IOException {
        if (!useIntelligentLoading) {
            return load(modelZipFile, loadUpdaterState);
        }

        // Use intelligent loading with automatic context
        try (ModelLoadingContext context = ModelLoadingContext.forModel(modelZipFile)) {
            return load(modelZipFile, loadUpdaterState, context);
        }
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

            // Use STORED (no compression) for SDNB files. Neural network weights are
            // high-entropy binary data that compresses poorly (~1% reduction). Skipping
            // compression eliminates CPU overhead on both save and load.
            zos.setMethod(ZipOutputStream.STORED);

            byte[] buffer = new byte[65536];
            for (File file : existingFiles) {
                if (!file.exists() || !file.isFile()) {
                    log.warn("File disappeared between initial check and ZIP addition: {}", file.getAbsolutePath());
                    continue;
                }

                String entryName = file.getName();
                log.debug("Adding ZIP entry (STORED): {} from {}", entryName, file.getAbsolutePath());

                // STORED entries require size and CRC32 upfront
                long fileSize = file.length();
                CRC32 crc = new CRC32();
                try (FileInputStream crcFis = new FileInputStream(file);
                     BufferedInputStream crcBis = new BufferedInputStream(crcFis, 65536)) {
                    int len;
                    while ((len = crcBis.read(buffer)) > 0) {
                        crc.update(buffer, 0, len);
                    }
                }

                ZipEntry zipEntry = new ZipEntry(entryName);
                zipEntry.setMethod(ZipEntry.STORED);
                zipEntry.setSize(fileSize);
                zipEntry.setCompressedSize(fileSize);
                zipEntry.setCrc(crc.getValue());
                zos.putNextEntry(zipEntry);

                try (FileInputStream fis = new FileInputStream(file);
                     BufferedInputStream bis = new BufferedInputStream(fis, 65536)) {
                    int len;
                    while ((len = bis.read(buffer)) > 0) {
                        zos.write(buffer, 0, len);
                    }
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
        long totalBytesExtracted = 0;
        int entryCount = 0;

        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile)))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                // Check entry count limit
                entryCount++;
                if (entryCount > maxZipEntries) {
                    throw new IOException("Potential zip bomb detected: too many entries. " +
                            "Found " + entryCount + " entries, maximum allowed is " + maxZipEntries + ". " +
                            "If this is a legitimate SDZ file, increase the limit using " +
                            "SDZSerializer.setMaxZipEntries() or system property 'nd4j.sdz.maxZipEntries'");
                }

                String entryName = entry.getName();
                File entryFile = new File(targetDir, entryName);

                // Zip Slip protection
                String canonicalEntryPath = entryFile.getCanonicalPath();
                if (!canonicalEntryPath.startsWith(canonicalTargetPath + File.separator) && !canonicalEntryPath.equals(canonicalTargetPath)) {
                    throw new IOException("Zip Slip vulnerability detected! Entry is outside of the target dir: " + entryName);
                }

                // Check compression ratio if sizes are known
                long compressedSize = entry.getCompressedSize();
                long uncompressedSize = entry.getSize();
                if (compressedSize > 0 && uncompressedSize > 0) {
                    double ratio = (double) uncompressedSize / compressedSize;
                    if (ratio > maxCompressionRatio) {
                        throw new IOException("Potential zip bomb detected: suspicious compression ratio. " +
                                "Entry '" + entryName + "' has ratio " + String.format("%.1f", ratio) +
                                ":1 (compressed: " + compressedSize + " bytes, uncompressed: " + uncompressedSize + " bytes). " +
                                "Maximum allowed ratio is " + String.format("%.1f", maxCompressionRatio) + ":1. " +
                                "If this is a legitimate SDZ file, increase the limit using " +
                                "SDZSerializer.setMaxCompressionRatio() or system property 'nd4j.sdz.maxCompressionRatio'");
                    }
                }

                // Check if claimed uncompressed size would exceed limit
                if (uncompressedSize > 0 && (totalBytesExtracted + uncompressedSize) > maxTotalUncompressedSize) {
                    throw new IOException("Potential zip bomb detected: total uncompressed size would exceed limit. " +
                            "Entry '" + entryName + "' claims " + uncompressedSize + " bytes, " +
                            "which would bring total to " + (totalBytesExtracted + uncompressedSize) + " bytes. " +
                            "Maximum allowed is " + maxTotalUncompressedSize + " bytes (" + (maxTotalUncompressedSize / (1024 * 1024)) + " MB). " +
                            "If this is a legitimate SDZ file, increase the limit using " +
                            "SDZSerializer.setMaxTotalUncompressedSize() or system property 'nd4j.sdz.maxZipSize'");
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

                    // Extract with size limit protection
                    long entryBytesWritten = 0;
                    try (FileOutputStream fos = new FileOutputStream(entryFile);
                         BufferedOutputStream bos = new BufferedOutputStream(fos)) {
                        int len;
                        while ((len = zis.read(buffer)) > 0) {
                            entryBytesWritten += len;
                            totalBytesExtracted += len;

                            // Check limit during extraction
                            if (totalBytesExtracted > maxTotalUncompressedSize) {
                                // Clean up partial file
                                bos.close();
                                fos.close();
                                entryFile.delete();
                                throw new IOException("Potential zip bomb detected while extracting entry '" + entryName + "'. " +
                                        "Total extracted size " + totalBytesExtracted + " bytes exceeded maximum allowed " +
                                        maxTotalUncompressedSize + " bytes. " +
                                        "If this is a legitimate SDZ file, increase the limit using " +
                                        "SDZSerializer.setMaxTotalUncompressedSize() or system property 'nd4j.sdz.maxZipSize'");
                            }

                            bos.write(buffer, 0, len);
                        }
                    }
                }
                zis.closeEntry();
            }
        } catch (IOException e) {
            if (e.getMessage() != null && e.getMessage().contains("zip bomb")) {
                throw e; // Re-throw zip bomb exceptions as-is
            }
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