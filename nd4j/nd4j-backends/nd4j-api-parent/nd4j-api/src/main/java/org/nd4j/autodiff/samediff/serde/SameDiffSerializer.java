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

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.*; // Import base package
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.graph.*; // Assuming FlatBuffers generated classes are here
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.exception.ND4JUnknownDataTypeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.shade.guava.primitives.Ints; // Use shaded Guava

import java.io.*;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.nd4j.linalg.api.buffer.DataType.FLOAT16;


/**
 * SameDiffSerializer: Serializes/deserializes SameDiff instances.
 * Supports sharding and large models (>2GB) using a format where FlatBuffers
 * handle metadata and large array data is appended as raw binary blobs within
 * each shard file, referenced by a manifest map within the same file.
 * File extension: .sdnb (SameDiff Native Blob)
 */
@Slf4j
public class SameDiffSerializer {

    // --- Constants ---
    private static final byte[] FILE_MAGIC = "SDNB".getBytes();
    private static final int FILE_VERSION = 1; // Version for this specific format
    // Header: MAGIC(4) + VERSION(4) + ManifestOffset(8) + ManifestLength(8) + FlatBufferOffset(8) = 32 bytes
    private static final long HEADER_SIZE = FILE_MAGIC.length + 4 + 8 + 8 + 8;

    // Threshold to decide whether to store array inline in FlatBuffer or append raw bytes
    private static final long APPEND_THRESHOLD_BYTES = 1 * 1024 * 1024; // Append arrays >= 1MB

    // Max size for a single shard file (including header, metadata, blobs, manifest)
    // Keep well below 2GB to avoid internal FlatBuffers Java reader int offset issues.
    private static final long MAX_SHARD_FILE_BYTES = (long) (1.0 * 1024.0 * 1024.0 * 1024.0); // 1 GiB limit per shard file

    // For writing/reading raw array chunks directly to file stream/channel
    private static final int RAW_IO_CHUNK_SIZE_BYTES = 8 * 1024 * 1024; // 8MB chunks

    // Metadata keys
    public static final String META_ND4J_FORMAT_VERSION = "nd4j.format.version";
    public static final String META_ND4J_FORMAT_TYPE = "nd4j.format.type";
    public static final String META_FRAMEWORK_VERSION = "framework.version";
    public static final String META_CREATION_TIME = "creation.time";
    public static final String META_SHARD_INDEX = "shard.index";
    public static final String META_SHARD_TYPE = "shard.type";
    private static final String FORMAT_TYPE_APPENDED = "APPENDED_BLOB";

    // Used by enrich metadata - prevent repeated reflection
    private static volatile String frameworkVersion = null;
    private static final Object fwVersionLock = new Object();


    // --- Public Save API ---

    /**
     * Saves the SameDiff model, automatically determining if sharding is needed.
     * Uses the "Metadata (FlatBuffers) + Appended Raw Data" format.
     * Recommended method for most users.
     */
    public static void save(@NonNull SameDiff sameDiff, @NonNull File baseFile, boolean saveUpdaterState, Map<String, String> metadata) throws IOException {
        saveAutoShard(sameDiff, baseFile, saveUpdaterState, metadata);
    }

    /**
     * Saves the SameDiff model, automatically determining if sharding is needed based on variable sizes.
     * Calls either the standard single-file save or the multi-file saveSharded internally.
     */
    @SneakyThrows
    public static void saveAutoShard(@NonNull SameDiff sameDiff, @NonNull File baseFile,
                                     boolean saveUpdaterState, Map<String, String> metadata) {
        Preconditions.checkNotNull(sameDiff, "SameDiff instance cannot be null");
        Preconditions.checkNotNull(baseFile, "Base file path cannot be null.");

        long totalAppendableBytes = 0;
        int appendableCount = 0;
        List<SDVariable> appendableVars = new ArrayList<>();
        // ** This map will hold arrays IF saving as single file **
        Map<String, INDArray> arraysToAppendForSingleFile = new LinkedHashMap<>();

        // Iterate through variables to estimate size and identify arrays for potential appending
        for (SDVariable var : sameDiff.variables()) {
            // Include only VARIABLE and CONSTANT types with actual data
            if ((var.getVariableType() == VariableType.VARIABLE || var.getVariableType() == VariableType.CONSTANT)
                    && var.getArr() != null && !var.getArr().isEmpty()) { // Check non-null and non-empty
                INDArray arr = var.getArr();
                long sizeBytes = arr.length() * arr.dataType().width();
                if (sizeBytes < 0) sizeBytes = Long.MAX_VALUE; // Overflow safety

                if (sizeBytes >= APPEND_THRESHOLD_BYTES) {
                    try {
                        totalAppendableBytes = Math.addExact(totalAppendableBytes, sizeBytes);
                        appendableCount++;
                        appendableVars.add(var);
                        // ** If saving as single file, this array will be appended **
                        arraysToAppendForSingleFile.put(var.name(), arr);
                    } catch (ArithmeticException e) {
                        totalAppendableBytes = Long.MAX_VALUE; // Treat overflow as needing sharding
                        break; // No need to check further if total size overflows
                    }
                }
            }
        }
        log.info("Calculated total appendable data size for auto-sharding: {} bytes (~{:.2f} GB) from {} arrays.",
                totalAppendableBytes, totalAppendableBytes / (1024.0 * 1024.0 * 1024.0), appendableCount);

        // Estimate base metadata size + size needed for manifest entries if saved as single file
        long estimatedOverhead = calculateBaseMetadataSizeEstimate(sameDiff)
                + calculateManifestSizeEstimate(arraysToAppendForSingleFile.size());
        long estimatedTotalSizeSingleFile = -1;
        try {
            estimatedTotalSizeSingleFile = Math.addExact(totalAppendableBytes, estimatedOverhead);
        } catch (ArithmeticException e) {
            estimatedTotalSizeSingleFile = Long.MAX_VALUE; // Overflow means > limit
        }


        // Check individual large variable sizes against shard limit (relevant for both paths)
        for (SDVariable var : appendableVars) {
            INDArray arr = var.getArr();
            long varSizeBytes = arr.length() * arr.dataType().width();
            if (varSizeBytes < 0) varSizeBytes = Long.MAX_VALUE;
            // Estimate size if this var was in its own shard (min overhead)
            long singleVarOverhead = HEADER_SIZE + calculateBaseMetadataSizeEstimate(null) // Minimal base meta
                    + calculateVariableMetadataSizeEstimate(var)
                    + calculateManifestSizeEstimate(1); // Manifest for 1 entry
            long estimatedSizeInShard = -1;
            try {
                estimatedSizeInShard = Math.addExact(singleVarOverhead, varSizeBytes);
            } catch (ArithmeticException e) {
                estimatedSizeInShard = Long.MAX_VALUE;
            }

            if (estimatedSizeInShard > MAX_SHARD_FILE_BYTES) {
                throw new IOException(String.format(
                        "Variable '%s' size (%d bytes) + estimated overhead potentially exceeds shard limit (%d bytes). Cannot save model.",
                        var.name(), varSizeBytes, MAX_SHARD_FILE_BYTES));
            }
        }

        // Decide sharding based on estimated total size vs. limit
        boolean requiresSharding = estimatedTotalSizeSingleFile > MAX_SHARD_FILE_BYTES;

        log.info("Auto-shard calculation: Estimated total size for single file: {} bytes (~{:.2f} GB). Shard limit: {} bytes. Requires sharding? {}",
                estimatedTotalSizeSingleFile >= 0 ? estimatedTotalSizeSingleFile : "Overflow",
                estimatedTotalSizeSingleFile >= 0 ? estimatedTotalSizeSingleFile / (1024.0 * 1024.0 * 1024.0) : Double.POSITIVE_INFINITY,
                MAX_SHARD_FILE_BYTES,
                requiresSharding);

        if (!requiresSharding) {
            log.info("Model size allows for single file save. Saving to: {}", baseFile.getAbsolutePath());
            // *** Call the modified saveInternal, passing the map of arrays to append ***
            saveInternal(sameDiff, baseFile, saveUpdaterState, metadata, arraysToAppendForSingleFile);
        } else {
            // Calculate estimated shard count just for logging info if needed
            long availablePayloadPerShard = Math.max(1, MAX_SHARD_FILE_BYTES - calculateBaseMetadataSizeEstimate(null) - calculateManifestSizeEstimate(50)); // Rough estimate
            int numVarShards = (totalAppendableBytes == Long.MAX_VALUE) ? Integer.MAX_VALUE / 2 : Math.max(1, (int) Math.ceil((double) totalAppendableBytes / availablePayloadPerShard));
            int totalEstimatedShards = 1 + numVarShards;

            log.info("Model size requires sharding. Saving with dynamically determined shards (estimated {}) using base name: {}", totalEstimatedShards, baseFile.getAbsolutePath());
            // Pass estimated count for logging, actual count determined inside saveSharded
            saveSharded(sameDiff, baseFile, saveUpdaterState, totalEstimatedShards, metadata);
        }
    }

    /**
     * Saves the SameDiff model across multiple files (shards) using the SDNB format per shard.
     * The number of shards is dynamically determined based on MAX_SHARD_FILE_BYTES.
     */
    @SneakyThrows
    public static void saveSharded(@NonNull SameDiff sameDiff, @NonNull File baseFile,
                                   boolean saveUpdaterState, int estimatedTotalShards,
                                   Map<String, String> metadata) throws IOException {
        Preconditions.checkNotNull(sameDiff, "SameDiff instance cannot be null");
        Preconditions.checkNotNull(baseFile, "Base file path cannot be null");
        Preconditions.checkArgument(estimatedTotalShards >= 1, "Estimated total shards must be at least 1"); // Although dynamic, estimate is useful

        File parentDir = baseFile.getParentFile();
        if (parentDir == null) parentDir = new File(".");
        parentDir = parentDir.getAbsoluteFile();
        if (!parentDir.exists() && !parentDir.mkdirs())
            throw new IOException("Could not create parent directories: " + parentDir.getAbsolutePath());
        Preconditions.checkState(parentDir.isDirectory(), "Parent path is not a directory: %s", parentDir.getAbsolutePath());
        String baseName = baseFile.getName();
        int dotIdx = baseName.lastIndexOf('.');
        if (dotIdx > 0) baseName = baseName.substring(0, dotIdx);

        Map<String, String> baseMetadata = enrichMetadata(metadata); // Assumed helper

        // --- Save Graph Shard (Shard 0) ---
        Map<String, String> shard0Metadata = new HashMap<>(baseMetadata);
        shard0Metadata.put(META_SHARD_INDEX, "0");
        shard0Metadata.put(META_SHARD_TYPE, "graph");
        SameDiff graphShard = createGraphShard(sameDiff); // Assumed helper
        String shard0NamePattern = String.format("%s.shard0-of-%s.sdnb", baseName, "N"); // Temp name
        File shard0File = new File(parentDir, shard0NamePattern);
        log.info("Saving graph structure shard to temporary name: {}", shard0File.getName());
        // Use saveInternal, which won't find large arrays in graphShard and thus won't append any blobs.
        saveInternal(graphShard, shard0File, false, shard0Metadata, null); // Assumed exists, passing null for external arrays
        log.info("Graph structure shard saved temporarily.");
        // --- End Graph Structure Shard ---


        // --- Distribute Variable Data based on SIZE across Variable Shards ---
        List<SDVariable> allVarsToDistribute = getVariablesWithDataSorted(sameDiff); // Gets vars WITH data from original SD
        List<File> variableShardFiles = new ArrayList<>(); // Track saved shard file objects

        if (!allVarsToDistribute.isEmpty()) {
            log.info("Saving variable data dynamically across shards based on file size limit ({} bytes)...", MAX_SHARD_FILE_BYTES);
            int currentShardIndex = 1; // Variable shards start from 1
            SameDiff currentVarShard = SameDiff.create(); // Initialize first shard SD object (stubs only)
            currentVarShard.setLogExecution(false);
            Map<String, INDArray> currentShardArraysToAppend = new LinkedHashMap<>(); // Map to hold ACTUAL arrays for this shard
            Map<String, GradientUpdater> currentShardUpdaterMap = new HashMap<>();
            long currentShardFileBytesEstimate = HEADER_SIZE + calculateBaseMetadataSizeEstimate(currentVarShard); // Assumed helper

            for (int varIdx = 0; varIdx < allVarsToDistribute.size(); varIdx++) {
                SDVariable var = allVarsToDistribute.get(varIdx); // Variable from original SameDiff
                INDArray arr = var.getArr(); // *** The ACTUAL array data ***
                if (arr == null) continue; // Should not happen due to getVariablesWithDataSorted filter

                long varSizeBytes = arr.isEmpty() ? 0 : arr.length() * arr.dataType().width();
                if (varSizeBytes < 0) varSizeBytes = Long.MAX_VALUE; // Overflow safety

                boolean appendData = varSizeBytes >= APPEND_THRESHOLD_BYTES;
                long varMetadataEstimate = calculateVariableMetadataSizeEstimate(var); // Assumed helper
                long estimatedContributionToFile; // Includes metadata + data (inline or appended)

                if (appendData) {
                    estimatedContributionToFile = varMetadataEstimate + varSizeBytes;
                } else {
                    // Small arrays will be added WITH data to currentVarShard later
                    estimatedContributionToFile = varMetadataEstimate + varSizeBytes + calculateInlineArrayOverheadEstimate(arr); // Assumed helper
                }
                if (estimatedContributionToFile < 0) estimatedContributionToFile = Long.MAX_VALUE; // Overflow safety


                // --- Check if adding this variable requires starting a new shard FIRST ---
                boolean startNewShard = false;
                if (!currentVarShard.getVariables().isEmpty()) { // Check only if current shard has variables assigned
                    // Estimate manifest size IF this var is large and appended
                    long nextManifestEstimate = calculateManifestSizeEstimate(currentShardArraysToAppend.size() + (appendData ? 1 : 0)); // Assumed helper
                    long projectedFileSize = -1;
                    try {
                        projectedFileSize = Math.addExact(currentShardFileBytesEstimate, estimatedContributionToFile);
                        projectedFileSize = Math.addExact(projectedFileSize, nextManifestEstimate);
                    } catch (ArithmeticException e) {
                        projectedFileSize = Long.MAX_VALUE;
                    }

                    if (projectedFileSize > MAX_SHARD_FILE_BYTES) {
                        log.debug("Variable '{}' (idx {}) est. contribution {} would exceed limit for shard {}. Projected FILE size: {} > Limit: {}. Finalizing current shard first.",
                                var.name(), varIdx, estimatedContributionToFile, currentShardIndex, projectedFileSize, MAX_SHARD_FILE_BYTES);
                        startNewShard = true;
                    }
                }
                // Also check single large var constraint
                if (!startNewShard && appendData) {
                    long baseOverhead = HEADER_SIZE + calculateBaseMetadataSizeEstimate(null) + calculateManifestSizeEstimate(1); // Assumed helper
                    long singleVarShardSizeEst = -1;
                    try {
                        singleVarShardSizeEst = Math.addExact(baseOverhead, varMetadataEstimate);
                        singleVarShardSizeEst = Math.addExact(singleVarShardSizeEst, varSizeBytes);
                    } catch (ArithmeticException e) { singleVarShardSizeEst = Long.MAX_VALUE; }
                    if (singleVarShardSizeEst > MAX_SHARD_FILE_BYTES) {
                        throw new IOException(String.format("Variable '%s' (%d bytes) + overhead exceeds shard limit (%d).", var.name(), varSizeBytes, MAX_SHARD_FILE_BYTES));
                    }
                }
                // --- End Check ---

                // If starting a new shard, save the PREVIOUS one NOW
                if (startNewShard) {
                    log.info("Saving completed variable shard {} before starting new one.", currentShardIndex);
                    try {
                        // *** Pass the map of arrays to append for the shard being saved ***
                        saveVariableShardHelper(currentVarShard, currentShardArraysToAppend, // Assumed helper exists
                                currentShardUpdaterMap, baseName, currentShardIndex,
                                baseMetadata, parentDir, variableShardFiles, saveUpdaterState);
                        log.info(">>> Successfully saved variable shard {} <<<", currentShardIndex);
                    } catch (Exception e) {
                        log.error(">>> FAILED to save variable shard {} within loop <<<", currentShardIndex, e);
                        throw new IOException("Failed saving intermediate shard " + currentShardIndex, e);
                    }

                    // Reset for the NEW shard
                    currentShardIndex++;
                    currentVarShard = SameDiff.create(); // New empty shard (stubs only)
                    currentVarShard.setLogExecution(false);
                    currentShardArraysToAppend = new LinkedHashMap<>(); // New empty map for arrays
                    currentShardUpdaterMap = new HashMap<>();
                    currentShardFileBytesEstimate = HEADER_SIZE + calculateBaseMetadataSizeEstimate(currentVarShard); // Reset estimate
                    log.debug("Initialized new variable shard {}", currentShardIndex);
                }
                // --- End Check and Save ---

                // --- Add variable definition (stub) to the current shard's SameDiff ---
                log.trace("Adding variable definition '{}' ({} bytes, append={}) to shard {}", var.name(), varSizeBytes, appendData, currentShardIndex);
                // Create stub without array data in currentVarShard
                SDVariable stub = new SDVariable(var.name(), var.getVariableType(), currentVarShard, var.getShape(), var.dataType());
                currentVarShard.addVariable(stub);
                // Copy structural metadata (control deps etc.) from original Variable metadata to the stub's metadata
                Variable oMeta = sameDiff.getVariables().get(var.name());
                Variable sMeta = currentVarShard.getVariables().get(var.name());
                if (oMeta != null && sMeta != null) {
                    sMeta.setControlDeps(copyList(oMeta.getControlDeps())); // Assumed helper
                    sMeta.setControlDepsForOp(copyList(oMeta.getControlDepsForOp())); // Assumed helper
                    sMeta.setControlDepsForVar(copyList(oMeta.getControlDepsForVar())); // Assumed helper
                }

                // --- Handle array data ---
                if (!appendData) {
                    // Add small data inline TO THE SHARD'S SAMEDIFF OBJECT
                    // Add with data
                    if (var.getVariableType() == VariableType.CONSTANT) {
                        currentVarShard.constant(var.name(), arr.dup());
                    } else { // VARIABLE or ARRAY
                        currentVarShard.var(var.name(), arr.dup());
                    }
                } else {
                    // Store the ACTUAL array data in the map for this shard
                    currentShardArraysToAppend.put(var.name(), arr);
                }

                // Collect updater state if requested
                if (var.getVariableType() == VariableType.VARIABLE && saveUpdaterState &&
                        sameDiff.getUpdaterMap() != null && sameDiff.getUpdaterMap().containsKey(var.name())) {
                    currentShardUpdaterMap.put(var.name(), sameDiff.getUpdaterMap().get(var.name()));
                }

                // Accumulate ESTIMATED FILE size
                try {
                    currentShardFileBytesEstimate = Math.addExact(currentShardFileBytesEstimate, estimatedContributionToFile);
                } catch (ArithmeticException e) {
                    currentShardFileBytesEstimate = Long.MAX_VALUE;
                    log.warn("Estimated file size for shard {} overflowed long.", currentShardIndex);
                }
                // --- End Adding Variable ---

            } // End loop over variables

            // Save the final (last accumulated) shard if it contains variables or arrays to append
            if (currentVarShard != null && (!currentVarShard.getVariables().isEmpty() || !currentShardArraysToAppend.isEmpty())) {
                log.debug("Saving final accumulated variable shard {}", currentShardIndex);
                try {
                    // *** Pass the final map of arrays ***
                    saveVariableShardHelper(currentVarShard, currentShardArraysToAppend, // Assumed helper exists
                            currentShardUpdaterMap, baseName, currentShardIndex,
                            baseMetadata, parentDir, variableShardFiles, saveUpdaterState);
                    log.info(">>> Successfully saved FINAL variable shard {} <<<", currentShardIndex);
                } catch (Exception e) {
                    log.error(">>> FAILED to save FINAL variable shard {} <<<", currentShardIndex, e);
                    throw new IOException("Failed saving final shard " + currentShardIndex, e);
                }
            } else {
                log.debug("Final variable shard {} was empty, not saving.", currentShardIndex);
            }
            // --- End Variable Data Distribution ---

        } else {
            log.warn("No variables with arrays found to distribute; no variable shards were created.");
        }

        // --- Finalize: Rename files ---
        int finalTotalShards = variableShardFiles.size() + 1; // Graph shard + variable shards
        log.info("Renaming shard files to reflect final count of {}.", finalTotalShards);
        renameShardFiles(baseName, parentDir, shard0File, variableShardFiles, finalTotalShards); // Assumed helper exists
        log.info("Finished saving dynamically sharded model based on size.");
    }


    /**
     * Estimates the size of the manifest map when serialized using Java Object Serialization.
     * This provides a rough estimate used for projecting shard file size during dynamic sharding,
     * helping to decide when a shard is nearing its maximum size limit *before* adding the
     * potentially large manifest itself.
     *
     * @param numEntries The number of entries (large appended arrays) expected in the manifest for the current shard.
     * @return An estimated size in bytes for the serialized manifest.
     */
    private static long calculateManifestSizeEstimate(int numEntries) {
        // Estimate based on typical Java Serialization overhead. This doesn't need to be exact,
        // just good enough to prevent the final file size from grossly exceeding the limit
        // due to an unexpectedly large manifest.

        // Base overhead for stream headers, map object class descriptor, etc.
        long baseOverhead = 2048; // Conservative guess: 2KB fixed overhead

        // Average overhead per entry:
        // - String key object + chars (assume avg 40 chars * 2 bytes/char + obj overhead = ~128 bytes?)
        // - Pair object overhead (~32 bytes?)
        // - 2x Long objects (in Pair) (2 * 8 bytes data + 2* obj overhead = ~48 bytes?)
        // - Map entry object overhead (~32 bytes?)
        // Total per entry estimate is roughly 128 + 32 + 48 + 32 = ~240 bytes.
        // Let's use a slightly rounded-up 256 bytes per entry for buffer.
        long bytesPerEntry = 256;

        long estimatedSize = -1;
        try {
            // Use multiplyExact and addExact to catch potential overflow with huge numEntries
            long entryTotal = Math.multiplyExact((long) numEntries, bytesPerEntry);
            estimatedSize = Math.addExact(baseOverhead, entryTotal);
        } catch (ArithmeticException e) {
            log.warn("Overflow calculating estimated manifest size for {} entries. Returning Long.MAX_VALUE.", numEntries);
            return Long.MAX_VALUE; // Return max value if estimate calculation overflows
        }

        return estimatedSize;
    }

    /**
     * Load a SameDiff instance, automatically detecting if it's sharded or single-file
     * based on the presence of shard files or the file header.
     *
     * @param modelFile        Path to the base model file (e.g., "model.sdnb" if single file,
     *                         or the base name like "model.bin" used for sharding which expects
     *                         "model.bin.shard0-of-N.sdnb" etc. in the same directory).
     * @param loadUpdaterState If true, attempt to load updater state.
     * @return The loaded SameDiff instance.
     * @throws IOException If loading fails or the format is inconsistent.
     */
    public static SameDiff load(@NonNull File modelFile, boolean loadUpdaterState) throws IOException {
        Preconditions.checkNotNull(modelFile, "Model file path cannot be null.");
        File parentDir = modelFile.getParentFile();
        if (parentDir == null) parentDir = new File(".");
        parentDir = parentDir.getAbsoluteFile();
        String baseName = modelFile.getName();
        int dotIdx = baseName.lastIndexOf('.');
        if (dotIdx > 0) baseName = baseName.substring(0, dotIdx);

        final String filePrefix = baseName + ".shard";
        File[] matchingFiles = parentDir.listFiles((dir, name) -> name.startsWith(filePrefix) && name.endsWith(".sdnb")); // Look for .sdnb

        if (matchingFiles != null && matchingFiles.length > 0) {
            // Shard files exist, delegate to loadSharded
            log.info("Shard files detected for base name '{}'. Attempting sharded load.", baseName);
            return loadSharded(modelFile, loadUpdaterState);
        } else {
            // No shard files found, attempt to load the provided file as a single-file model
            log.info("No shard files detected. Attempting to load '{}' as a single-file SDNB model.", modelFile.getAbsolutePath());
            if (!modelFile.exists()) {
                throw new FileNotFoundException("Model file does not exist: " + modelFile.getAbsolutePath());
            }
            if (!isValidSdnbFile(modelFile)) { // Check if it has the correct magic number/header
                throw new IOException("File format not recognized as SDNB (magic number mismatch or too small): " + modelFile.getAbsolutePath());
            }
            // Loading a single file means creating a new SameDiff instance.
            return loadInternal(modelFile, loadUpdaterState,null);
        }
    }

    /**
     * Loads a sharded SameDiff model saved using the SDNB format per shard.
     */
    /**
     * Loads a sharded SameDiff model saved using the SDNB format per shard.
     * Corrected to properly merge data from variable shards into the main instance.
     */
    public static SameDiff loadSharded(@NonNull File baseFile, boolean loadUpdaterState) throws IOException {
        Preconditions.checkNotNull(baseFile, "Base file path cannot be null.");
        File parentDir = baseFile.getParentFile();
        if (parentDir == null) parentDir = new File(".");
        parentDir = parentDir.getAbsoluteFile();
        Preconditions.checkState(parentDir.isDirectory(), "Parent path is not a directory: %s", parentDir.getAbsolutePath());
        log.info("Searching for shard files in directory: {}", parentDir.getAbsolutePath());
        String baseName = baseFile.getName();
        int dotIdx = baseName.lastIndexOf('.');
        if (dotIdx > 0) baseName = baseName.substring(0, dotIdx);

        int numShards = detectShardCount(parentDir, baseName); // Use helper
        log.info("Detected {} total shards for base model '{}'", numShards, baseName);

        // --- Load Shard 0 (Graph Structure) ---
        String shard0Name = String.format("%s.shard0-of-%d.sdnb", baseName, numShards);
        File shard0File = new File(parentDir, shard0Name);
        Preconditions.checkState(shard0File.exists(), "Required graph shard file does not exist: %s", shard0File.getAbsolutePath());
        if (!shard0File.isFile() || shard0File.length() <= HEADER_SIZE || !shard0File.canRead()) { // Check header size too
            throw new IOException("Graph shard file is empty, not readable, or too small: " + shard0File.getAbsolutePath());
        }
        log.info("Loading graph structure from shard 0: {}", shard0File.getName());
        SameDiff result = null;
        try {
            // *** Load shard 0 by passing null for existingSD ***
            result = loadInternal(shard0File, false, null); // Creates the base SameDiff instance
            if (result == null) {
                throw new IOException("Loading graph shard 0 returned a null SameDiff instance.");
            }
        } catch (Exception e) {
            log.error("Failed to load graph shard file: {}", shard0File.getAbsolutePath(), e);
            throw new IOException("Failed to load graph shard from " + shard0File.getAbsolutePath(), e);
        }

        // --- Load Variable Shards (Shards 1 to N-1) ---
        for (int shardIdx = 1; shardIdx < numShards; shardIdx++) {
            String shardName = String.format("%s.shard%d-of-%d.sdnb", baseName, shardIdx, numShards);
            File shardFile = new File(parentDir, shardName);
            if (!shardFile.exists() || !shardFile.isFile() || shardFile.length() <= HEADER_SIZE || !shardFile.canRead()) { // Check header size
                log.warn("Variable shard file {} is missing, empty, too small, or not readable. Skipping.", shardFile.getName());
                continue;
            }
            log.info("Loading and merging data from variable shard {} (file: {})", shardIdx, shardFile.getName());
            try {
                // The returned value is the same 'result' instance, potentially modified.
                loadInternal(shardFile, loadUpdaterState, result);
            } catch (Exception e) {
                log.error("Failed to load or merge variable shard {}: {}", shardIdx, shardFile.getAbsolutePath(), e);
                // Consider if failure here should be fatal or allow partial loads
                throw new IOException("Failed to load or merge variable shard " + shardIdx + " from " + shardFile.getAbsolutePath(), e); // Fail hard for consistency
            }
        }

        log.info("Finished loading sharded model from base path: {}", baseFile.getPath());
        for(String varName : result.variableNames()) {
            if(result.getVariable(varName).getVariableType() == VariableType.VARIABLE && result.getVariable(varName).getArr() == null){
                log.warn("Variable '{}' still has null array after loading all shards.", varName);
            }
        }
        return result;
    }


    /**
     * Internal save method for the SDNB format (Metadata FB + Appended Blobs).
     * Can save a graph shard (no appended data expected) or a variable shard.
     * For variable shards, it serializes metadata based on the stub `sameDiff` object
     * but appends raw data from the `externalArraysToAppend` map.
     *

     *
     * @param sameDiff             The SameDiff instance containing metadata (stubs, ops, small arrays).
     * @param file                 The file to save to.
     * @param saveUpdaterState     Whether to include updater state in the metadata FlatBuffer.
     * @param metadata             File metadata.
     * @param externalArraysToAppend Map of {VarName -> INDArray} for large arrays whose raw data
     * should be appended. If null or empty, no data is appended
     * (used for graph shard or shards with only small arrays).
     * These arrays are NOT expected to be attached to the `sameDiff` object itself.
     * @throws IOException If any file I/O or serialization error occurs.
     */
    private static void saveInternal(
            @NonNull SameDiff sameDiff, @NonNull File file,
            boolean saveUpdaterState, Map<String, String> metadata,
            Map<String, INDArray> externalArraysToAppend // Map of arrays to append
    ) throws IOException {

        Map<String, Pair<Long, Long>> largeArrayManifest = new LinkedHashMap<>();
        Set<String> largeArrayNamesForMetadata = new HashSet<>();
        Set<String> smallInlineArrayNamesForMetadata = new HashSet<>();

        // 1. Identify arrays for metadata serialization (Same logic as before)
        for (SDVariable var : getVariablesWithDataSorted(sameDiff)) {
            INDArray arr = var.getArr();
            if (arr == null) continue;
            long sizeBytes = arr.isEmpty() ? 0 : arr.length() * arr.dataType().width();
            if (sizeBytes < 0) sizeBytes = Long.MAX_VALUE;
            if (externalArraysToAppend != null && externalArraysToAppend.containsKey(var.name())) {
                log.warn("Variable '{}' has data attached to the metadata SameDiff instance AND is also marked for external appending. Prioritizing external definition for metadata.", var.name());
                largeArrayNamesForMetadata.add(var.name());
            } else if (sizeBytes < APPEND_THRESHOLD_BYTES && !arr.isEmpty()) {
                smallInlineArrayNamesForMetadata.add(var.name());
            } else if (sizeBytes >= APPEND_THRESHOLD_BYTES) {
                log.error("Consistency Warning: Variable '{}' has a large array ({} bytes) attached to the metadata SameDiff instance but was NOT provided in the externalArraysToAppend map. Marking as 'large' for metadata, but its data WILL NOT BE SAVED unless also provided externally.", var.name(), sizeBytes);
                largeArrayNamesForMetadata.add(var.name());
            }
        }
        if (externalArraysToAppend != null) {
            largeArrayNamesForMetadata.addAll(externalArraysToAppend.keySet());
        }
        log.debug("File {}: Metadata serialization: {} large names (expect appended), {} small names (expect inline).",
                file.getName(), largeArrayNamesForMetadata.size(), smallInlineArrayNamesForMetadata.size());


        // 2. Serialize Metadata FlatBuffer (Same logic as before)
        ByteBuffer metadataBuffer = serializeMetadataFlatBuffer(sameDiff, saveUpdaterState, metadata,
                largeArrayNamesForMetadata, smallInlineArrayNamesForMetadata);
        int metadataLength = metadataBuffer.remaining();
        if (metadataLength <= 0 && (sameDiff.variables().size() > 0 || sameDiff.getOps().size() > 0)) {
            log.warn("Serialization produced empty metadata buffer for non-empty SameDiff instance {}. File may be invalid.", file.getName());
        }


        // 3. Write File using RandomAccessFile
        long manifestOffset = -1;
        long manifestLength = -1;
        try (RandomAccessFile raf = new RandomAccessFile(file, "rw")) {
            raf.setLength(0); // Clear file if it exists

            // --- Write Header ---
            raf.write(FILE_MAGIC);                      // Pos 0-3
            raf.writeInt(FILE_VERSION);                 // Pos 4-7
            long manifestOffsetPos = raf.getFilePointer(); // Pos 8
            raf.writeLong(-1L);                         // Placeholder for manifest offset
            long manifestLengthPos = raf.getFilePointer(); // Pos 16
            raf.writeLong(-1L);                         // Placeholder for manifest length
            // Calculate metadata offset *before* writing it
            long metadataOffset = raf.getFilePointer() + 8; // Expected start = Pos 24 + 8 = 32
            raf.writeLong(metadataOffset);              // Actual Metadata Offset @ pos 24
            long posAfterHeader = raf.getFilePointer(); // Should be HEADER_SIZE (32)
            log.debug("Header written. Pos after header: {}, Calculated metadataOffset: {}", posAfterHeader, metadataOffset);
            Preconditions.checkState(posAfterHeader == HEADER_SIZE, "Header write size error");
            Preconditions.checkState(posAfterHeader == metadataOffset, "Header write position error. Pos=" + posAfterHeader + ", Expected metadataOffset=" + metadataOffset);


            // --- Write FlatBuffer Metadata ---
            log.debug("Writing FlatBuffer metadata. Expected start offset: {}, Expected length: {}", metadataOffset, metadataLength);
            long bytesActuallyWritten = 0;
            if (metadataLength > 0) {
                log.info("Attempting to write {} bytes of metadata.", metadataLength); // Elevated to INFO
                byte[] metadataBytes = new byte[metadataLength];

                // Log buffer state BEFORE get()
                log.debug("Metadata Buffer BEFORE get: pos={}, limit={}, remaining={}",
                        metadataBuffer.position(), metadataBuffer.limit(), metadataBuffer.remaining());
                int initialPosition = metadataBuffer.position();

                try {
                    metadataBuffer.get(metadataBytes); // Read from FB buffer into heap array
                } catch (Exception e) {
                    log.error("Error during metadataBuffer.get(metadataBytes)!", e);
                    throw new IOException("Failed to read bytes from FlatBuffer ByteBuffer", e);
                }

                // Log buffer state AFTER get()
                int finalPosition = metadataBuffer.position();
                int bytesReadFromBuffer = finalPosition - initialPosition;
                log.debug("Metadata Buffer AFTER get: pos={}, limit={}, remaining={}, Bytes Read={}",
                        finalPosition, metadataBuffer.limit(), metadataBuffer.remaining(), bytesReadFromBuffer);

                // Check if get() actually read the expected number of bytes
                if (bytesReadFromBuffer != metadataLength) {
                    log.error("ByteBuffer.get() did not read expected number of bytes! Expected {}, Read {}",
                            metadataLength, bytesReadFromBuffer);
                    // Throw an error here as this indicates a fundamental problem reading the FB buffer
                    throw new IOException("Failed to read the expected number of bytes ("+ metadataLength +") from the FlatBuffer ByteBuffer, only read " + bytesReadFromBuffer);
                }

                // Log content check (optional, potentially slow - check first few bytes)
                boolean hasData = false;
                int checkLen = Math.min(16, metadataBytes.length);
                for(int i = 0; i < checkLen; i++) {
                    if (metadataBytes[i] != 0) {
                        hasData = true; break;
                    }
                }
                log.debug("First {} bytes of metadataBytes have non-zero data? {}", checkLen, hasData);


                // Perform the write operation
                try {
                    raf.write(metadataBytes); // Write heap array to file
                } catch (IOException e) {
                    log.error("IOException during raf.write(metadataBytes)!", e);
                    throw e; // Rethrow
                }

                // Check file pointer advancement
                long currentPosAfterWrite = raf.getFilePointer();
                bytesActuallyWritten = currentPosAfterWrite - metadataOffset; // Use metadataOffset (32) as start
                log.debug("raf.write(metadataBytes) executed. Bytes reportedly written in this step: {}. New position: {}",
                        bytesActuallyWritten, currentPosAfterWrite);

                if (bytesActuallyWritten != metadataLength) {
                    log.error("RAF write mismatch! Expected to write {}, raf reports {} written in metadata block.", metadataLength, bytesActuallyWritten);
                    // Throw an exception as this indicates a serious IO problem.
                    throw new IOException("RAF failed to write the expected number of metadata bytes. Expected=" + metadataLength + ", Written=" + bytesActuallyWritten);
                }

            } else {
                log.warn("Metadata buffer length is zero for {}. FlatBuffer section will be empty.", file.getName());
                bytesActuallyWritten = 0; // Explicitly zero if nothing written
            }
            long metadataEndOffset = raf.getFilePointer(); // Get position AFTER potential write
            log.debug("FlatBuffer metadata written block finished. File position now: {}", metadataEndOffset);

            // Perform the final check
            log.debug("CHECKSTATE: metadataEndOffset={}, metadataOffset={}, metadataLength={}",
                    metadataEndOffset, metadataOffset, metadataLength);
            Preconditions.checkState(metadataEndOffset == metadataOffset + metadataLength,
                    "Metadata write position error. Check failed: " + metadataEndOffset + " == " + metadataOffset + " + " + metadataLength +
                            " (Position after metadata write block: " + metadataEndOffset + ", Bytes reportedly written: " + bytesActuallyWritten + ")");


            // --- Write Raw Data Blobs & Update Manifest ---
            long currentWriteOffset = metadataEndOffset; // Start appending after metadata
            FileChannel channel = raf.getChannel();

            if (externalArraysToAppend != null && !externalArraysToAppend.isEmpty()) {
                // ... (Same refined logic as in the previous response for writing blobs) ...
                log.info("Appending raw data for {} large arrays (provided externally) to file {}...", externalArraysToAppend.size(), file.getName());

                for (Map.Entry<String, INDArray> entry : externalArraysToAppend.entrySet()) {
                    String name = entry.getKey();
                    INDArray arr = entry.getValue();

                    if (arr == null || arr.isEmpty()) {
                        log.warn("Skipping append for variable '{}': Provided array is null or empty.", name);
                        largeArrayManifest.put(name, Pair.of(currentWriteOffset, 0L));
                        continue;
                    }

                    DataBuffer buffer = arr.data();
                    long lengthBytes = arr.length() * buffer.getElementSize();
                    if (lengthBytes <= 0) {
                        log.error("Calculated invalid length ({}) for variable '{}'. Skipping append.", lengthBytes, name);
                        largeArrayManifest.put(name, Pair.of(currentWriteOffset, 0L));
                        continue;
                    }

                    largeArrayManifest.put(name, Pair.of(currentWriteOffset, lengthBytes));
                    log.info("SAVE [{}]: Appending {} bytes. File Offset={}, Order={}, DType={}",
                            name, lengthBytes, currentWriteOffset, arr.ordering(), arr.dataType());

                    ByteBuffer dataNio = buffer.asNio();
                    boolean usedDirectNio = false;

                    if (dataNio != null) {
                        dataNio.order(ByteOrder.nativeOrder());
                        long arrOffsetBytes = arr.offset() * buffer.getElementSize();
                        if (arrOffsetBytes >= 0 && arrOffsetBytes <= Integer.MAX_VALUE &&
                                lengthBytes >= 0 && lengthBytes <= Integer.MAX_VALUE &&
                                (arrOffsetBytes + lengthBytes) >= 0 && (arrOffsetBytes + lengthBytes) <= Integer.MAX_VALUE)
                        {
                            log.debug("SAVE [{}]: Attempting Direct NIO write path.", name);
                            try {
                                dataNio.position((int) arrOffsetBytes);
                                dataNio.limit((int) (arrOffsetBytes + lengthBytes));
                                long totalWritten = 0;
                                long nioWriteStartTime = System.currentTimeMillis();
                                channel.position(currentWriteOffset);

                                while (totalWritten < lengthBytes) {
                                    long writtenThisCall = channel.write(dataNio);
                                    if (writtenThisCall < 0) throw new IOException("FileChannel write error (direct) for " + name);
                                    totalWritten += writtenThisCall;
                                    if (writtenThisCall == 0 /* ... */) { Thread.sleep(1); /* Check timeout */ }
                                    else { nioWriteStartTime = System.currentTimeMillis(); }
                                }
                                if (totalWritten != lengthBytes) throw new IOException("NIO Write incomplete (direct) for " + name + "...");

                                currentWriteOffset = channel.position();
                                raf.seek(currentWriteOffset);
                                usedDirectNio = true;
                                log.info("SAVE [{}]: Successfully used Direct NIO write path ({} bytes).", name, totalWritten);
                            } catch (IOException | InterruptedException e) {
                                if(e instanceof InterruptedException) Thread.currentThread().interrupt();
                                log.warn("SAVE [{}]: Direct NIO write failed, attempting fallback. Error: {}", name, e.getMessage());
                                channel.position(currentWriteOffset); raf.seek(currentWriteOffset); usedDirectNio = false;
                            } catch (Exception e) {
                                log.warn("SAVE [{}]: Unexpected error during Direct NIO write, attempting fallback.", name, e);
                                channel.position(currentWriteOffset); raf.seek(currentWriteOffset); usedDirectNio = false;
                            }
                        } else {
                            log.warn("SAVE [{}]: Variable offset/length (BufferOffset={}, Length={}) exceeds Integer.MAX_VALUE for direct NIO. Using fallback.", name, arrOffsetBytes, lengthBytes);
                            usedDirectNio = false;
                        }
                    } else {
                        log.warn("SAVE [{}]: Direct NIO buffer not available (buffer.asNio() returned null). Using fallback.", name);
                        usedDirectNio = false;
                    }

                    if (!usedDirectNio) {
                        log.warn("SAVE [{}]: Using chunked fallback write path. Attempting NIO write per chunk with NATIVE ORDER.", name);
                        long bytesWritten = 0;
                        long sourceElementOffset = arr.offset();
                        long totalElementsInArray = arr.length();
                        long fallbackWriteStartTime = System.currentTimeMillis();

                        while (bytesWritten < lengthBytes) {
                            long elementsProcessed = bytesWritten / buffer.getElementSize();
                            long elementsRemaining = totalElementsInArray - elementsProcessed;
                            if (elementsRemaining <= 0 && bytesWritten < lengthBytes) throw new IOException("Inconsistency during fallback write for " + name);
                            int elementsInChunk = (int) Math.min(RAW_IO_CHUNK_SIZE_BYTES / buffer.getElementSize(), elementsRemaining);
                            if (elementsInChunk <= 0 && elementsRemaining > 0) elementsInChunk = 1; else if (elementsInChunk <= 0) throw new IOException("Inconsistency: elementsRemaining is zero but loop continued for " + name);
                            long currentGetElementOffset = sourceElementOffset + elementsProcessed;

                            INDArray chunkDup = null;
                            try {
                                INDArray chunkView = arr.reshape(arr.length()).get(NDArrayIndex.interval(currentGetElementOffset, currentGetElementOffset + elementsInChunk));
                                chunkDup = chunkView.dup(arr.ordering());
                                DataBuffer chunkDataBuffer = chunkDup.data();
                                long chunkLengthBytes = chunkDataBuffer.length() * chunkDataBuffer.getElementSize();
                                log.trace("SAVE [{}]: Processing chunk. Elements: {}, Bytes: {}, Chunk Dup Offset: {}", name, elementsInChunk, chunkLengthBytes, chunkDup.offset());

                                if (chunkLengthBytes > 0) {
                                    ByteBuffer nioBufferView = chunkDataBuffer.asNio();
                                    long nioBufferOffsetBytes = 0; // Assume chunkDup offset is 0

                                    if (nioBufferView != null && chunkLengthBytes <= Integer.MAX_VALUE) {
                                        log.trace("SAVE [{}]: Using NIO write within fallback for chunk.", name);
                                        try {
                                            nioBufferView.order(ByteOrder.nativeOrder());
                                            nioBufferView.position((int) nioBufferOffsetBytes);
                                            nioBufferView.limit((int) (nioBufferOffsetBytes + chunkLengthBytes));
                                            long totalWrittenThisChunk = 0;
                                            long currentChunkWritePos = currentWriteOffset + bytesWritten;
                                            channel.position(currentChunkWritePos);

                                            while (totalWrittenThisChunk < chunkLengthBytes) {
                                                long writtenNow = channel.write(nioBufferView);
                                                if (writtenNow < 0) throw new IOException("FileChannel write error (fallback chunk) for " + name);
                                                totalWrittenThisChunk += writtenNow;

                                            }
                                            if (totalWrittenThisChunk != chunkLengthBytes) throw new IOException("Fallback NIO Write incomplete for chunk of " + name + "...");

                                            bytesWritten += totalWrittenThisChunk;
                                            log.trace("SAVE [{}]: Fallback NIO write successful for chunk ({} bytes). Total written so far: {}", name, totalWrittenThisChunk, bytesWritten);
                                        } catch (IOException e) {
                                            throw new IOException("Failed fallback NIO write for chunk of " + name, e);
                                        }
                                    } else {
                                        throw new IOException("Unsupported condition: Fallback save path requires direct NIO buffer for chunks, but it's not available for variable '" + name + "'.");
                                    }
                                } else {
                                    log.trace("SAVE [{}]: Skipping empty chunk.", name);
                                }
                            } finally {
                                if (chunkDup != null && chunkDup.closeable()) chunkDup.close();
                            }
                            if (System.currentTimeMillis() - fallbackWriteStartTime > 300000) throw new IOException("Timeout during fallback write for variable '" + name + "'.");
                        } // End while loop for chunks

                        if (bytesWritten != lengthBytes) throw new IOException("Fallback chunking write incomplete for " + name + "...");

                        currentWriteOffset = channel.position();
                        raf.seek(currentWriteOffset);
                        log.info("SAVE [{}]: Fallback chunking write path completed successfully ({} bytes).", name, bytesWritten);
                    } // End if (!usedDirectNio)

                    raf.seek(currentWriteOffset); // Final sync before next variable or manifest
                } // End loop over external arrays to append
                manifestOffset = currentWriteOffset;
            } else {
                manifestOffset = metadataEndOffset;
                log.info("No external large arrays provided or map was empty for file {}. No data appended.", file.getName());
            }


            // --- Write Manifest (Same as before) ---
            log.debug("Writing manifest map ({} entries) for file {} at offset {}", largeArrayManifest.size(), file.getName(), manifestOffset);
            byte[] manifestBytes;
            try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
                 ObjectOutputStream oos = new ObjectOutputStream(baos)) {
                oos.writeObject(largeArrayManifest); manifestBytes = baos.toByteArray();
            } catch (IOException e) {
                throw new IOException("Failed to serialize manifest", e);
            }
            manifestLength = manifestBytes.length;
            raf.seek(manifestOffset); raf.write(manifestBytes);
            long finalFileSize = raf.getFilePointer();
            log.debug("Manifest written ({} bytes). Final file size for {}: {}", manifestLength, file.getName(), finalFileSize);


            // --- Patch Header (Same as before) ---
            raf.seek(manifestOffsetPos); raf.writeLong(manifestOffset);
            raf.seek(manifestLengthPos); raf.writeLong(manifestLength);
            log.debug("Patched file header for {}: Manifest Offset={}, Manifest Length={}", file.getName(), manifestOffset, manifestLength);

        } // End try-with-resources (closes raf and channel)
    }


    /**
     * Internal load method for the SDNB format. Handles loading from a single file
     * or loading data into an existing SameDiff instance from a variable shard file.
     *
     * @param file             The .sdnb file to load.
     * @param loadUpdaterState If true, attempt to load/merge updater state.
     * @param existingSD       If null, creates a new SameDiff instance (for shard 0 or single file).
     * If non-null, loads data (appended arrays, small inline arrays, updater state)
     * from this file into the provided existingSD instance.
     * @return The created or populated SameDiff instance.
     * @throws IOException If loading fails.
     */
    private static SameDiff loadInternal(@NonNull File file, boolean loadUpdaterState, SameDiff existingSD) throws IOException {
        log.debug("Loading internal format from file: {}", file.getAbsolutePath());
        Map<String, Pair<Long, Long>> manifest = null;
        ByteBuffer metadataBuffer = null;
        long manifestOffset = -1, manifestLength = -1, metadataOffset = -1, metadataLength = -1;

        try (FileInputStream fis = new FileInputStream(file); FileChannel channel = fis.getChannel()) {
            long fileSize = channel.size();
            if (fileSize < HEADER_SIZE)
                throw new IOException("File too small to be a valid SDNB file: " + file.getAbsolutePath() + " (size: " + fileSize + " bytes)");

            // --- Read Header ---
            ByteBuffer headerBuffer = ByteBuffer.allocate((int) HEADER_SIZE).order(ByteOrder.BIG_ENDIAN); // Header is Big Endian
            int headerRead = channel.read(headerBuffer, 0);
            if (headerRead != HEADER_SIZE)
                throw new IOException("Failed to read complete header from: " + file.getAbsolutePath());
            headerBuffer.flip();
            byte[] magicRead = new byte[FILE_MAGIC.length];
            headerBuffer.get(magicRead);
            if (!Arrays.equals(FILE_MAGIC, magicRead))
                throw new IOException("Invalid magic number in file: " + file.getAbsolutePath());

            int version = headerBuffer.getInt();
            manifestOffset = headerBuffer.getLong();
            manifestLength = headerBuffer.getLong();
            metadataOffset = headerBuffer.getLong();
            // --- Validate Header Offsets/Lengths ---
            if (metadataOffset != HEADER_SIZE || manifestOffset < metadataOffset || manifestLength < 0 || manifestOffset > fileSize || manifestOffset + manifestLength > fileSize)
                throw new IOException(String.format("Invalid header offsets/lengths in file %s. " +
                                "MetaOffset=%d (expected %d), ManifestOffset=%d, ManifestLength=%d, FileSize=%d",
                        file.getAbsolutePath(), metadataOffset, HEADER_SIZE, manifestOffset, manifestLength, fileSize));

            metadataLength = manifestOffset - metadataOffset;
            if (metadataLength < 0) // Should be caught by manifestOffset < metadataOffset, but check explicitly
                throw new IOException("Invalid metadata length (negative): " + metadataLength);
            if (metadataLength > Integer.MAX_VALUE)
                throw new IOException("Metadata length > 2GB not supported for direct ByteBuffer allocation.");


            // --- Read Manifest ---
            if (manifestLength > 0) {
                if (manifestLength > Integer.MAX_VALUE)
                    throw new IOException("Manifest length > 2GB not supported for direct ByteBuffer allocation.");
                ByteBuffer manifestNio = ByteBuffer.allocate((int) manifestLength);
                int manifestRead = channel.read(manifestNio, manifestOffset);
                if (manifestRead != manifestLength)
                    throw new IOException("Failed to read complete manifest from: " + file.getAbsolutePath());
                manifestNio.flip();
                try (ByteArrayInputStream bais = new ByteArrayInputStream(manifestNio.array(), manifestNio.position(), manifestNio.remaining());
                     ObjectInputStream ois = new ObjectInputStream(bais)) {
                    manifest = (Map<String, Pair<Long, Long>>) ois.readObject();
                } catch (Exception e) {
                    throw new IOException("Failed to deserialize manifest from file: " + file.getAbsolutePath(), e);
                }
            } else {
                manifest = Collections.emptyMap(); // Handle case of no appended data
                log.debug("Manifest length is zero in file {}. No appended data expected.", file.getName());
            }


            // --- Read Metadata Buffer ---
            // Only read if length > 0 to avoid allocating 0-byte buffer
            FlatGraph fg = null; // Parsed FlatGraph metadata
            if (metadataLength > 0) {
                metadataBuffer = ByteBuffer.allocateDirect((int) metadataLength).order(ByteOrder.LITTLE_ENDIAN); // FlatBuffers standard
                int metaRead = channel.read(metadataBuffer, metadataOffset);
                if (metaRead != metadataLength)
                    throw new IOException("Failed to read complete metadata FlatBuffer from: " + file.getAbsolutePath());
                metadataBuffer.flip(); // Prepare for reading
                // Parse FlatGraph once, used by multiple steps below
                try {
                    fg = FlatGraph.getRootAsFlatGraph(metadataBuffer.duplicate()); // Use duplicate to preserve original buffer position
                    if (fg == null) throw new IOException("Failed to get FlatGraph root from metadata ByteBuffer.");
                } catch (Exception e) {
                    throw new IOException("Error parsing FlatBuffer metadata from file: " + file.getAbsolutePath(), e);
                }
            } else {
                log.warn("Metadata length is zero in file {}. Cannot deserialize graph structure or load inline arrays.", file.getName());
                // If existingSD is null, we can't proceed without metadata.
                // If existingSD is not null, we might still be able to load appended data if manifest is present.
                if (existingSD == null) {
                    throw new IOException("Cannot create new SameDiff instance: metadata is empty in file " + file.getAbsolutePath());
                }
            }


            // --- Create or Use SameDiff Instance ---
            SameDiff targetSD;
            if (existingSD == null) {
                // --- Case 1: Create NEW SameDiff instance (Shard 0 or single file) ---
                if (metadataBuffer == null || metadataLength == 0) {
                    throw new IOException("Cannot create new SameDiff instance: metadata is empty in file " + file.getAbsolutePath());
                }
                log.debug("Deserializing graph structure into NEW SameDiff instance from file {}", file.getName());
                targetSD = deserializeFromFlatBuffers(metadataBuffer.duplicate(), loadUpdaterState, manifest); // Pass manifest to identify non-inline
                log.debug("Initial SameDiff structure deserialized for file {}. Vars: {}, Ops: {}", file.getName(), targetSD.variables().size(), targetSD.getOps().size());
            } else {
                // --- Case 2: Populate EXISTING SameDiff instance (Variable Shard) ---
                log.debug("Populating EXISTING SameDiff instance from variable shard file {}", file.getName());
                targetSD = existingSD; // Use the passed-in instance as the target

                // Load small inline arrays defined in *this shard's* metadata
                if (fg != null) { // Check if metadata was successfully parsed
                    loadSmallInlineArraysIntoExisting(targetSD, fg, manifest);
                    // Load and merge updater state if requested
                    if (loadUpdaterState) {
                        loadAndUpdateUpdaterState(targetSD, fg);
                    }
                } else {
                    log.warn("Cannot load small inline arrays or updater state for existing SameDiff: metadata was empty or failed to parse in file {}", file.getName());
                }
            }

            // --- Load Appended Array Data (Common to both cases) ---
            // Pass the correct targetSD instance (either newly created or existing)
            // Also pass the metadataBuffer for lookups within loadAppendedArrayData
            if (manifest != null && !manifest.isEmpty()) {
                log.info("Loading appended array data for {} variables from file {} into {} SameDiff instance...",
                        manifest.size(), file.getName(), (existingSD == null ? "NEW" : "EXISTING"));
                if (metadataBuffer == null) {
                    throw new IOException("Cannot load appended array data: metadata buffer is required for lookups but is missing or empty in file " + file.getAbsolutePath());
                }
                loadAppendedArrayData(targetSD, manifest, channel, metadataBuffer.duplicate()); // Pass duplicate buffer
            } else {
                log.info("No appended array data found in manifest for file {}.", file.getName());
            }

            return targetSD; // Return the newly created or populated instance

        } catch (IOException e) {
            // Add file context to exceptions
            log.error("IOException during loadInternal for file: {}", file.getAbsolutePath(), e);
            throw e; // Re-throw
        } catch (Exception e) {
            // Catch other potential runtime exceptions
            log.error("Unexpected exception during loadInternal for file: {}", file.getAbsolutePath(), e);
            throw new IOException("Unexpected error loading file " + file.getAbsolutePath(), e);
        }
    }


    /**
     * Helper method for loadInternal. Parses the metadata FlatGraph from a variable shard
     * and loads any *small, inline* arrays found directly into the existing SameDiff instance.
     * It skips variables that are present in the manifest (as they are handled by loadAppendedArrayData).
     * Includes detailed logging for layer_0_b trace.
     */
    private static void loadSmallInlineArraysIntoExisting(
            @NonNull SameDiff targetSD,
            @NonNull FlatGraph fg,
            @NonNull Map<String, Pair<Long, Long>> manifest) throws IOException {

        log.debug("Checking for small inline arrays in metadata for shard...");
        int loadedCount = 0;
        int skippedManifestCount = 0;
        int skippedExistingCount = 0;
        int errorCount = 0;

        if (fg == null) {
            log.error("LOAD_INLINE: FlatGraph object is null. Cannot load inline arrays.");
            return; // Cannot proceed
        }

        for (int i = 0; i < fg.variablesLength(); i++) {
            FlatVariable fv = fg.variables(i);
            if (fv == null) continue;
            String name = fv.name();
            if (name == null || name.isEmpty()) continue;


            // Skip if this variable's data is expected to be appended (handled elsewhere)
            if (manifest.containsKey(name)) {
                skippedManifestCount++;
                continue; // Skip inline load attempt
            }

            // Check if the variable exists in the target SameDiff graph structure
            if (!targetSD.hasVariable(name)) {
                log.warn("LOAD_INLINE: Variable '{}' found in variable shard metadata but not in the main graph structure. Skipping inline load.", name);
                errorCount++;
                continue;
            }

            // Check if the variable *already* has an array in the target SD
            SDVariable targetVar = targetSD.getVariable(name);
            // Check if this FlatVariable actually contains inline array data
            FlatArray fa = fv.ndarray();

            if (fa != null) {
                log.debug("LOAD_INLINE: Found inline FlatArray metadata for '{}'. Attempting deserialization.", name);
                try {
                    // Use the potentially simplified deserializeSmallNdArrayFromInlineBuffer
                    INDArray smallArr = deserializeSmallNdArrayFromInlineBuffer(fa,name); // Call the corrected version
                    if (smallArr != null) {
                        log.info("LOAD_INLINE: Successfully deserialized inline array for '{}'. Shape: {}", name, Arrays.toString(smallArr.shape()));

                        // Perform consistency checks (dtype, shape)
                        DataType expectedDtype = targetVar.dataType();
                        if (expectedDtype != null && expectedDtype != DataType.UNKNOWN && smallArr.dataType() != expectedDtype) {
                            log.warn("LOAD_INLINE: Data type mismatch for small inline array '{}'. Expected {}, Found {}. Attempting cast.", name, expectedDtype, smallArr.dataType());
                            try {
                                smallArr = smallArr.castTo(expectedDtype);
                            }
                            catch (Exception castEx) {
                                log.error("LOAD_INLINE: Failed to cast array '{}' to {}.", name, expectedDtype, castEx);
                                errorCount++;
                                continue;
                            }
                        }
                        long[] expectedShape = targetVar.getShape();
                        if(expectedShape != null && !Arrays.equals(expectedShape, smallArr.shape())){
                            log.error("LOAD_INLINE: Shape mismatch for small inline array '{}'. Expected {}, Found {}. Cannot load array.", name, Arrays.toString(expectedShape), Arrays.toString(smallArr.shape()));
                            errorCount++;
                            continue; // Skip loading this array
                        }

                        // Associate the array
                        targetSD.setArrayForVariable(name, smallArr);
                        // Verification after setting
                        INDArray checkArr = targetSD.getArrForVarName(name);
                        if(checkArr == null) {
                            log.error("LOAD_INLINE: Set array for '{}' but getArrForVarName is null immediately after!", name);
                            // This indicates a problem with setArrayForVariable or the ArrayHolder
                            errorCount++;
                        } else {
                            log.info("LOAD_INLINE: Associated inline array for variable '{}'.", name);
                            loadedCount++;
                        }
                    } else {
                        // This is a critical failure point if smallArr should exist
                        log.error("LOAD_INLINE: deserializeSmallNdArrayFromInlineBuffer returned NULL for inline variable '{}'. Data will be missing!", name);
                        errorCount++;
                    }
                } catch (Exception e) {
                    log.error("LOAD_INLINE: Failed during deserialization or setting of inline array for variable '{}'.", name, e);
                    errorCount++;
                }
            } else {
                // fa == null
                log.trace("LOAD_INLINE: No inline FlatArray metadata found for '{}' (fa == null).", name);
                // This implies serializeSmall... failed or returned 0 for this variable during save.
            }
        } // End loop
        log.debug("Finished processing small inline arrays for shard. Loaded: {}, Skipped (in manifest): {}, Skipped (already present): {}, Errors: {}",
                loadedCount, skippedManifestCount, skippedExistingCount, errorCount);
    } // end loadSmallInlineArraysIntoExisting


    /**
     * Helper to convert byte array segment to Hex String for logging.
     */
    private static String bytesToHex(byte[] bytes, int offset, int length) {
        if (bytes == null) return "null";
        StringBuilder sb = new StringBuilder();
        Formatter formatter = new Formatter(sb);
        int end = Math.min(bytes.length, offset + length);
        int start = Math.max(0, offset);
        for (int i = start; i < end; i++) {
            formatter.format("%02X", bytes[i]);
            if (i < end - 1) sb.append(",");
        }
        formatter.close();
        return sb.toString();
    }

    /**
     * Helper method for loadInternal. Parses the metadata FlatGraph from a variable shard
     * and loads/merges any updater state found into the existing SameDiff instance.
     * CORRECTED: Updated call to deserializeSmallNdArrayFromInlineBuffer.
     *
     * @param targetSD The existing SameDiff instance to populate/update.
     * @param fg       The parsed FlatGraph object from the variable shard's metadata.
     */
    @SneakyThrows // For reflection field access
    private static void loadAndUpdateUpdaterState(
            @NonNull SameDiff targetSD,
            @NonNull FlatGraph fg) {

        if (fg == null) {
            log.warn("Cannot load updater state: FlatGraph metadata is null.");
            return;
        }

        if (fg.updaterStateLength() == 0) {
            log.debug("No updater state found in this shard's metadata.");
            return;
        }
        if (targetSD.getTrainingConfig() == null || targetSD.getTrainingConfig().getUpdater() == null) {
            log.warn("Cannot load updater state: TrainingConfig or IUpdater is missing from the target SameDiff instance (loaded from graph shard).");
            return;
        }

        log.debug("Loading and merging updater state from shard metadata...");
        int loadedCount = 0;
        int errorCount = 0;

        // Get access to the target updater map via reflection
        Field updaterMapField = SameDiff.class.getDeclaredField("updaterMap");
        updaterMapField.setAccessible(true);
        Map<String, GradientUpdater> targetUpdaterMap = (Map<String, GradientUpdater>) updaterMapField.get(targetSD);

        // Initialize the map if it's null (first time loading state)
        boolean mapInitialized = true;
        if (targetUpdaterMap == null) {
            targetUpdaterMap = new HashMap<>();
            updaterMapField.set(targetSD, targetUpdaterMap);
            mapInitialized = false; // Mark that we need to set the initializedTraining flag later
            log.debug("Initialized new updaterMap in target SameDiff instance.");
        }

        for (int i = 0; i < fg.updaterStateLength(); i++) {
            UpdaterState us = fg.updaterState(i);
            if (us == null) continue;
            String paramName = us.paramName();
            if (paramName == null || !targetSD.hasVariable(paramName)) {
                log.warn("Skipping updater state for parameter '{}': Variable not found in target SameDiff.", paramName);
                errorCount++;
                continue;
            }

            // Check if state for this parameter already exists (shouldn't happen with current save logic, but check)
            if (targetUpdaterMap.containsKey(paramName)) {
                log.warn("Updater state for parameter '{}' already exists in target SameDiff. Skipping state from this shard to avoid overwrite.", paramName);
                // Or implement merging logic if overwriting/combining state is desired/possible
                continue;
            }

            Map<String, INDArray> stateMap = new HashMap<>();
            boolean stateLoadSuccess = true;
            for (int j = 0; j < us.updaterStateKeysLength(); j++) {
                String key = us.updaterStateKeys(j);
                FlatArray faState = us.updaterStateValues(j);
                if (key == null || faState == null) {
                    log.warn("Skipping null key or null FlatArray in updater state for param '{}', key '{}'", paramName, key);
                    continue;
                }
                try {
                    // *** CORRECTED CALL to include varName ***
                    String stateVarName = paramName + "_" + key; // Construct a name for logging
                    INDArray stateArr = deserializeSmallNdArrayFromInlineBuffer(faState, stateVarName);

                    if (stateArr != null) {
                        stateMap.put(key, stateArr);
                    } else {
                        log.warn("Deserializing updater state array returned null for param '{}', key '{}'. Skipping this state entry.", paramName, key);
                        stateLoadSuccess = false; // Mark failure if any part fails
                    }
                } catch (Exception e) {
                    log.error("Error deserializing updater state array for param '{}', key '{}'", paramName, key, e);
                    stateLoadSuccess = false; // Mark failure
                }
            }

            // Instantiate and add the GradientUpdater only if all parts loaded successfully
            if (stateLoadSuccess && !stateMap.isEmpty()) {
                try {
                    // Use the IUpdater from the target SameDiff's TrainingConfig to instantiate
                    GradientUpdater gu = targetSD.getTrainingConfig().getUpdater().instantiate(stateMap, false); // Assuming instantiate exists
                    targetUpdaterMap.put(paramName, gu);
                    loadedCount++;
                    log.trace("Loaded updater state for parameter '{}'.", paramName);
                } catch (Exception e) {
                    log.error("Failed to instantiate GradientUpdater for parameter '{}' from loaded state.", paramName, e);
                    errorCount++;
                }
            } else if (!stateMap.isEmpty()) {
                // Log error only if stateMap is not empty but load failed somewhere
                log.error("Skipping updater state for parameter '{}' due to errors loading constituent arrays.", paramName);
                errorCount++;
            } else {
                // stateMap is empty - either no state or all failed/skipped silently
                log.trace("No valid updater state entries loaded for parameter '{}'.", paramName);
            }
        } // End loop over updater states in FlatGraph

        // If we loaded any state successfully *and* the map was newly created, set the initializedTraining flag
        if (loadedCount > 0 && !mapInitialized) {
            try {
                Field initField = SameDiff.class.getDeclaredField("initializedTraining");
                initField.setAccessible(true);
                initField.set(targetSD, true);
            } catch (Exception e) {
                log.error("Failed to set initializedTraining flag via reflection after loading updater state.", e);
            }
        }

        log.debug("Finished processing updater state for shard. Loaded states: {}, Errors/Skipped: {}", loadedCount, errorCount);
    } // end loadAndUpdateUpdaterState

    /**
     * Helper to deserialize FlatBuffer metadata and populate a NEW SameDiff instance
     * with stubs and small inline arrays. Large arrays remain empty.
     *
     * @param bbIn             ByteBuffer containing FlatBuffer metadata (position 0, correct limit, Little Endian)
     * @param loadUpdaterState If true, load updater state metadata and small state arrays.
     * @param manifest         Manifest mapping large array names (used ONLY to identify which arrays AREN'T inline).
     * @return A new SameDiff instance populated with metadata and small arrays.
     * @throws IOException If parsing fails.
     */
    private static SameDiff deserializeFromFlatBuffers(
            @NonNull ByteBuffer bbIn, boolean loadUpdaterState,
            @NonNull Map<String, Pair<Long, Long>> manifest) throws IOException { // Removed SameDiff sd param

        Preconditions.checkNotNull(bbIn, "Input ByteBuffer cannot be null");
        bbIn.order(ByteOrder.LITTLE_ENDIAN);
        bbIn.rewind();

        FlatGraph fg;
        try {
            fg = FlatGraph.getRootAsFlatGraph(bbIn);
            if (fg == null) throw new IOException("Failed to get FlatGraph root from ByteBuffer.");
        } catch (Exception e) {
            throw new IOException("Error parsing FlatBuffer metadata", e);
        }

        // *** Create NEW SameDiff instance ***
        SameDiff sd = SameDiff.create();
        sd.setLogExecution(false); // Typically disable logging for internal instance

        Map<String, String> fileMetadata = new HashMap<>();
        try { // Extract header metadata
            for (int i = 0; i < fg.metadataKeysLength(); i++) {
                String key = fg.metadataKeys(i);
                String val = fg.metadataValues(i);
                if (key != null) fileMetadata.put(key, val);
            }
        } catch (Exception e) {
            log.warn("Error reading FlatBuffer metadata entries", e);
        }

        Map<Pair<Integer, Integer>, SDVariable> variablesByNodeAndOutNum = new HashMap<>();

        // --- Reconstruct Variables (Stubs + Small Inline) ---
        int numVarsInFb = fg.variablesLength();
        log.debug("Deserializing {} variable definitions from FlatBuffer metadata into new SameDiff instance.", numVarsInFb);
        for (int i = 0; i < numVarsInFb; i++) {
            try {
                FlatVariable fv = fg.variables(i);
                if (fv == null) continue;
                String name = fv.name();
                if (name == null || name.isEmpty()) continue;
                // Since sd is new, no need to check sd.hasVariable(name)

                DataType dtype = FlatBuffersMapper.getDataTypeFromByte(fv.dtype());
                VariableType vt = FlatBuffersMapper.fromVarType(fv.variabletype());
                long[] shape = null;
                if (fv.shapeLength() > 0) {
                    shape = new long[fv.shapeLength()];
                    for (int j = 0; j < shape.length; j++)
                        shape[j] = fv.shape(j);
                }

                SDVariable var = new SDVariable(name, vt, sd, shape, dtype);
                sd.addVariable(var); // Adds to sd.variables map
                Variable varMeta = sd.getVariables().get(name);

                log.trace("Added variable stub: {}", name);

                // Restore control dependencies
                if (fv.controlDepsLength() > 0) {
                    List<String> l = new ArrayList<>();
                    for (int j = 0; j < fv.controlDepsLength(); j++) l.add(fv.controlDeps(j));
                    varMeta.setControlDeps(l);
                } else {
                    varMeta.setControlDeps(null);
                }
                if (fv.controlDepForOpLength() > 0) {
                    List<String> l = new ArrayList<>();
                    for (int j = 0; j < fv.controlDepForOpLength(); j++)
                        l.add(fv.controlDepForOp(j));
                    varMeta.setControlDepsForOp(l);
                } else {
                    varMeta.setControlDepsForOp(null);
                }
                if (fv.controlDepsForVarLength() > 0) {
                    List<String> l = new ArrayList<>();
                    for (int j = 0; j < fv.controlDepsForVarLength(); j++)
                        l.add(fv.controlDepsForVar(j));
                    varMeta.setControlDepsForVar(l);
                } else {
                    varMeta.setControlDepsForVar(null);
                }

                FlatArray fa = fv.ndarray();
                // Load SMALL inline arrays only if NOT in manifest
                if (!manifest.containsKey(name) && fa != null) {
                    try {
                        INDArray smallArr = deserializeSmallNdArrayFromInlineBuffer(fa,name);
                        if (smallArr != null) {
                            sd.setArrayForVariable(name, smallArr);
                            log.trace("Loaded small inline array for variable '{}'", name);
                        }
                    } catch (Exception e) {
                        log.warn("Failed inline load for presumed small array '{}'.", name, e);
                    }
                } else if (manifest.containsKey(name)) {
                    log.trace("Variable '{}' marked for appended data loading.", name);
                }

                IntPair idPair = fv.id();
                if (idPair != null)
                    variablesByNodeAndOutNum.put(new Pair<>(idPair.first(), idPair.second()), var);

            } catch (Exception e) {
                throw new IOException("Error processing FlatVariable at index " + i, e);
            }
        }
        log.debug("After variable loop, new sd.variables().size() = {}", sd.variables().size());

        // --- Reconstruct Ops ---
        int numOpsInFb = fg.nodesLength();
        log.debug("Deserializing {} ops from FlatBuffer metadata.", numOpsInFb);
        for (int i = 0; i < numOpsInFb; i++) {
            try {
                FlatNode fn = fg.nodes(i);
                if (fn == null) continue;
                String opOwnName = fn.name();
                if (opOwnName == null || opOwnName.isEmpty()) continue;
                // Op should not exist in the newly created sd
                if (sd.getOps().containsKey(opOwnName)) {
                    log.warn("Op '{}' unexpectedly already exists in new SameDiff.", opOwnName);
                    continue;
                }

                DifferentialFunction df = FlatBuffersMapper.fromFlatNode(fn);
                df.setSameDiff(sd);
                df.setOwnName(opOwnName);
                SameDiffOp sdo = SameDiffOp.builder().name(opOwnName).op(df).build();
                sd.getOps().put(opOwnName, sdo);
                log.trace("Added op: {}", opOwnName);

                // Link Inputs
                List<String> inputNames = new ArrayList<>();
                for (int j = 0; j < fn.inputPairedLength(); j++) {
                    IntPair pair = fn.inputPaired(j);
                    if (pair == null) continue;
                    SDVariable inVar = variablesByNodeAndOutNum.get(new Pair<>(pair.first(), pair.second()));
                    if (inVar != null && inVar.name() != null)
                        inputNames.add(inVar.name());
                }
                sdo.setInputsToOp(inputNames);
                for (String inName : inputNames) {
                    Variable inMeta = sd.getVariables().get(inName);
                    if (inMeta != null) {
                        if (inMeta.getInputsForOp() == null)
                            inMeta.setInputsForOp(new ArrayList<>());
                        if (!inMeta.getInputsForOp().contains(opOwnName))
                            inMeta.getInputsForOp().add(opOwnName);
                    }
                }

                // Link Outputs
                List<String> outputNames = new ArrayList<>();
                for (int j = 0; j < fn.outputNamesLength(); j++) {
                    String on = fn.outputNames(j);
                    if (on != null) outputNames.add(on);
                }
                sdo.setOutputsOfOp(outputNames);

                for (String outName : outputNames) {
                    Variable outMeta = sd.getVariables().get(outName);
                    if (outMeta != null) {
                        outMeta.setOutputOfOp(opOwnName);
                    } else {
                        log.error("Output stub '{}' not found for op '{}'.", outName, opOwnName);
                        /* Handle error */
                    }
                }

                // Link Control Dependencies ...
                List<String> cdList = new ArrayList<>();
                for (int j = 0; j < fn.controlDepsLength(); j++)
                    cdList.add(fn.controlDeps(j));

                if (!cdList.isEmpty())
                    sdo.setControlDeps(cdList);

                List<String> vcdList = new ArrayList<>();
                for (int j = 0; j < fn.varControlDepsLength(); j++)
                    vcdList.add(fn.varControlDeps(j));

                if (!vcdList.isEmpty())
                    sdo.setVarControlDeps(vcdList);

                List<String> cdfList = new ArrayList<>();
                for (int j = 0; j < fn.controlDepForLength(); j++) {
                    cdfList.add(fn.controlDepFor(j));
                }
                if (!cdfList.isEmpty()) sdo.setControlDepFor(cdfList);

                df.configureWithSameDiff(sd);
            } catch (Exception e) {
                throw new IOException("Error processing FlatNode at index " + i, e);
            }
        }
        log.debug("After op loop, new sd.ops().size() = {}", sd.getOps().size());

        // --- Loss Vars, Training Config, Updater State ---
        if (fg.lossVariablesLength() > 0) {
            for (int i = 0; i < fg.lossVariablesLength(); i++)
                sd.addLossVariable(fg.lossVariables(i));
        }
        String tcJson = fg.trainingConfig();
        if (tcJson != null && !tcJson.isEmpty())
            sd.setTrainingConfig(TrainingConfig.fromJson(tcJson));
        if (loadUpdaterState && fg.updaterStateLength() > 0 && sd.getTrainingConfig() != null) {
            Map<String, GradientUpdater> updaterMap = new HashMap<>(); // Initialize map for this instance
            boolean loadedAnyUpdater = false;
            for (int i = 0; i < fg.updaterStateLength(); i++) {
                UpdaterState us = fg.updaterState(i);
                if (us == null) continue;
                String paramName = us.paramName();
                if (paramName == null || !sd.hasVariable(paramName)) continue;
                Map<String, INDArray> stateMap = new HashMap<>();
                for (int j = 0; j < us.updaterStateKeysLength(); j++) {
                    String key = us.updaterStateKeys(j);
                    FlatArray faState = us.updaterStateValues(j);
                    if (key == null || faState == null) continue;
                    INDArray stateArr = deserializeSmallNdArrayFromInlineBuffer(faState,key);
                    if (stateArr != null) stateMap.put(key, stateArr);
                }
                if (!stateMap.isEmpty() && sd.getTrainingConfig().getUpdater() != null) {
                    GradientUpdater gu = sd.getTrainingConfig().getUpdater().instantiate(stateMap, false);
                    updaterMap.put(paramName, gu);
                    loadedAnyUpdater = true;
                }
            }
            if (loadedAnyUpdater) {
                try {
                    Field umf = SameDiff.class.getDeclaredField("updaterMap");
                    umf.setAccessible(true);
                    umf.set(sd, updaterMap);
                    Field itf = SameDiff.class.getDeclaredField("initializedTraining");
                    itf.setAccessible(true);
                    itf.set(sd, true);
                } catch (Exception e) {
                    log.error("Failed to set updater map.", e);
                }
            }
        }
        log.debug("Finished deserializeFromFlatBuffers. Final variable count: {}, Op count: {}", sd.variables().size(), sd.getOps().size());
        return sd; // Return the newly created and populated SameDiff instance
    }



    /**
     * Helper to load appended raw data into the SameDiff variables using FileChannel.
     * Retrieves shape, dtype, AND ordering information from the provided FlatBuffers metadata buffer.
     * MODIFIED WITH INTENSIVE LOGGING AND VERIFICATION.
     *
     * @param targetSD       The SameDiff instance to populate (can be newly created or existing).
     * @param manifest       Map mapping variable names to {offset, length} in the file channel.
     * @param channel        FileChannel positioned at the start of the file (seeking will be done internally).
     * @param metadataBuffer ByteBuffer (positioned at 0, with limit=metadataLength) containing the
     *                       parsed FlatBuffers metadata for the entire shard. Must be Little Endian.
     * @throws IOException If reading fails or metadata is inconsistent.
     */
    private static void loadAppendedArrayData(
            @NonNull SameDiff targetSD,
            @NonNull Map<String, Pair<Long, Long>> manifest,
            @NonNull FileChannel channel,
            @NonNull ByteBuffer metadataBuffer // Used to look up metadata like order
    ) throws IOException {

        byte[] tempChunk = null; // Lazy init for fallback reading path
        // Ensure buffer is ready for reading from the start
        metadataBuffer.order(ByteOrder.LITTLE_ENDIAN); // FlatBuffers standard
        metadataBuffer.position(0); // Rewind buffer before parsing

        FlatGraph fg = null; // Will hold parsed metadata
        try {
            fg = FlatGraph.getRootAsFlatGraph(metadataBuffer); // Get root of metadata graph once
            if (fg == null) throw new IOException("Failed to get FlatGraph root from metadata ByteBuffer.");
        } catch (Exception e) {
            throw new IOException("Error parsing metadata FlatBuffer within loadAppendedArrayData", e);
        }

        log.info("Attempting to load raw data for {} variables listed in manifest.", manifest.size());

        for (Map.Entry<String, Pair<Long, Long>> entry : manifest.entrySet()) {
            String name = entry.getKey();
            long offset = entry.getValue().getFirst(); // Offset in the FileChannel where raw data starts
            long lengthBytes = entry.getValue().getSecond(); // Length of the raw data blob

            log.debug("Processing manifest entry: Var='{}', Offset={}, Length={}", name, offset, lengthBytes);

            // --- Check Variable Existence in Target SD ---
            if (!targetSD.hasVariable(name)) {
                log.error("FATAL: Manifest contains entry for variable '{}' but it was not found in the target SameDiff instance's graph structure. Cannot load data.", name);
                // Depending on requirements, you might continue or throw an exception. Throwing is safer.
                throw new IOException("Variable '" + name + "' from manifest not found in SameDiff graph structure.");
            }
            SDVariable var = targetSD.getVariable(name);
            if (var == null) {
                log.error("FATAL: targetSD.hasVariable(\"{}\") returned true, but targetSD.getVariable(\"{}\") returned null. Inconsistent state.", name, name);
                throw new IllegalStateException("Inconsistent variable state for '" + name + "' in target SameDiff.");
            }
            log.trace("Variable '{}' found in target SameDiff.", name);

            // --- Check if Data Already Loaded ---
            if (var.getArr() != null) {
                // This could happen if the same variable (e.g., a constant) was somehow included
                // in multiple shards' manifests or loaded as small inline AND appended.
                log.warn("Variable '{}' already has an array in target SameDiff instance. Skipping append load for this entry (Offset={}, Length={}). Check save logic if this is unexpected.", name, offset, lengthBytes);
                continue;
            }

            // --- Get Metadata from SDVariable AND FlatVariable ---
            DataType dtype = var.dataType();
            long[] shape = var.getShape();
            char order = 'c'; // Default to 'c'

            // Find corresponding metadata in the FlatBuffer for *this shard*
            FlatVariable fv = findFlatVariableMeta(fg, name); // findFlatVariableMeta should already exist
            if (fv != null) {
                // Double check dtype/shape consistency if needed (optional)
                DataType fbDtype = FlatBuffersMapper.getDataTypeFromByte(fv.dtype());
                if (dtype != null && dtype != DataType.UNKNOWN && fbDtype != dtype) {
                    log.warn("DataType mismatch for '{}': Target SD has {}, Shard Metadata has {}. Using Target SD type.", name, dtype, fbDtype);
                } else if (dtype == null || dtype == DataType.UNKNOWN) {
                    dtype = fbDtype; // Use type from shard metadata if target was unknown
                }

                // Shape check (more critical)
                long[] fbShape = null;
                if (fv.shapeLength() > 0) {
                    fbShape = new long[fv.shapeLength()];
                    for (int j = 0; j < fbShape.length; j++) fbShape[j] = fv.shape(j);
                }
                if (shape != null && fbShape != null && !Arrays.equals(shape, fbShape)) {
                    log.error("Shape mismatch for '{}': Target SD has {}, Shard Metadata has {}. Cannot safely load data.", name, Arrays.toString(shape), Arrays.toString(fbShape));
                    throw new IOException("Shape mismatch in metadata for variable '" + name + "'.");
                } else if (shape == null && fbShape != null) {
                    shape = fbShape; // Use shape from shard metadata if target was unknown/null
                }

                // Get Order
                FlatArray fa = fv.ndarray(); // Check ndarray part of variable metadata
                if (fa != null) {
                    // FlatBuffers schema uses 0 for 'c', 1 for 'f'
                    order = fa.byteOrder() == 1 ? 'f' : 'c'; // Get order from FlatArray metadata
                    log.trace("Determined order '{}' for variable '{}' from shard metadata.", order, name);
                } else {
                    log.warn("FlatArray metadata missing within FlatVariable for appended variable '{}'. Assuming default 'c' order.", name);
                }
            } else {
                // This means the variable exists in the graph (targetSD) but has no specific entry
                // in *this shard's* metadata FlatBuffer. This shouldn't happen if the variable's
                // data is listed in *this shard's* manifest. Indicates a potential inconsistency during save.
                log.error("FATAL: FlatVariable metadata missing entirely in this shard for variable '{}' which is listed in the manifest. Save process might be flawed.", name);
                throw new IOException("Missing FlatVariable metadata for manifested variable '" + name + "'.");
            }
            // --- End Metadata Determination ---

            // --- Validate Metadata and Manifest Length ---
            if (dtype == null || dtype == DataType.UNKNOWN) {
                log.error("FATAL: Could not determine DataType for appended var '{}'. Skipping.", name);
                throw new IOException("Unknown DataType for variable '" + name + "'.");
            }
            if (shape == null) {
                log.error("FATAL: Could not determine Shape for appended var '{}'. Skipping.", name);
                throw new IOException("Unknown Shape for variable '" + name + "'.");
            }
            if (lengthBytes <= 0) {
                // Allow loading of empty arrays if shape implies 0 elements
                long expectedElements = ArrayUtil.prodLong(shape);
                if (expectedElements == 0 && lengthBytes == 0) {
                    log.warn("Manifest entry for '{}' has zero length, and shape {} implies zero elements. Will create empty array.", name, Arrays.toString(shape));
                } else {
                    log.error("FATAL: Invalid zero/negative manifest length ({}) for appended var '{}' with non-empty shape {}. Skipping.", lengthBytes, name, Arrays.toString(shape));
                    throw new IOException("Invalid manifest length for variable '" + name + "'.");
                }
            }

            int elementSize = dtype.width();
            if (elementSize <= 0) {
                log.error("FATAL: Invalid element size {} for dtype {} of variable '{}'. Skipping.", elementSize, dtype, name);
                throw new ND4JUnknownDataTypeException("Invalid element size for DataType " + dtype);
            }

            long expectedElements = ArrayUtil.prodLong(shape);
            if (expectedElements < 0) { // Shape overflow
                log.warn("Shape {} for var '{}' overflows long. Cannot verify manifest length {}. Proceeding with caution.", Arrays.toString(shape), name, lengthBytes);
            } else {
                long expectedLengthBytes = expectedElements * elementSize;
                if (lengthBytes != expectedLengthBytes) {
                    log.error("FATAL: Manifest length mismatch for var '{}'. Manifest: {}, Calculated from Shape*Dtype: {}. (Shape: {}, DType: {}). Skipping.",
                            name, lengthBytes, expectedLengthBytes, Arrays.toString(shape), dtype);
                    throw new IOException("Manifest length mismatch for variable '" + name + "'.");
                }
            }
            if (dtype == DataType.COMPRESSED || dtype == DataType.UTF8) {
                log.error("FATAL: Append loading for type {} for variable '{}' is not supported. Skipping.", dtype, name);
                throw new UnsupportedOperationException("Append loading not supported for DataType " + dtype);
            }


            log.debug("Preparing to load {} bytes for variable '{}' (dtype={}, shape={}, order={}) from file offset {}",
                    lengthBytes, name, dtype, Arrays.toString(shape), order, offset);

            // --- Create Target Array ---
            INDArray resultArr = null;
            DataBuffer targetBuffer = null;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                // Handle case of empty array creation
                if (lengthBytes == 0 && expectedElements == 0) {
                    log.debug("Creating empty INDArray for var '{}'", name);
                    resultArr = Nd4j.create(dtype, shape, Nd4j.getStrides(shape, order), order);
                } else if (lengthBytes > 0) {
                    log.debug("Creating uninitialized INDArray for var '{}'", name);
                    resultArr = Nd4j.createUninitialized(dtype, shape, order);
                }

                if (resultArr == null)
                    throw new IOException("Nd4j.createUninitialized/create returned null for " + name);
                targetBuffer = resultArr.data();
                if (targetBuffer == null && lengthBytes > 0) // Empty buffer is okay for empty array
                    throw new IOException("Target DataBuffer is null for " + name);

                log.trace("Successfully created target INDArray for '{}'. IsEmpty={}, Length={}, Shape={}", name, resultArr.isEmpty(), resultArr.length(), Arrays.toString(resultArr.shape()));

            } catch (Exception e) {
                log.error("FATAL: Failed to create INDArray for {}", name, e);
                throw new IOException("Failed to create INDArray for " + name, e);
            }

            // --- Read Data From Channel into Array Buffer ---
            // Skip reading if array is empty
            if (lengthBytes > 0) {
                long arrayOffsetBytes = resultArr.offset() * targetBuffer.getElementSize(); // Offset within the DataBuffer

                try {
                    channel.position(offset); // Position channel to where blob starts
                    log.trace("Channel positioned to offset {} for reading '{}'", offset, name);

                    ByteBuffer targetNio = targetBuffer.asNio();
                    if (targetNio != null && lengthBytes <= Integer.MAX_VALUE && arrayOffsetBytes <= Integer.MAX_VALUE && (arrayOffsetBytes + lengthBytes) <= Integer.MAX_VALUE) {
                        // --- Direct NIO Read Path ---
                        log.trace("Attempting Direct NIO read for '{}'", name);
                        targetNio.order(ByteOrder.nativeOrder()); // Ensure native order for direct memory access
                        targetNio.position((int) arrayOffsetBytes);
                        targetNio.limit((int) (arrayOffsetBytes + lengthBytes));
                        long totalRead = 0;
                        long startReadTime = System.currentTimeMillis();
                        while (totalRead < lengthBytes) {
                            int readBytes = channel.read(targetNio);
                            if (readBytes == -1) {
                                log.error("FATAL: EOFException encountered while reading data for '{}'. Expected {} bytes, read {} bytes.", name, lengthBytes, totalRead);
                                throw new EOFException("EOF encountered while reading data for variable '" + name + "' at offset " + offset);
                            }
                            totalRead += readBytes;
                            if (readBytes == 0) { // Avoid busy spin; yield or sleep briefly
                                try {
                                    Thread.sleep(1);
                                } catch (InterruptedException ie) {
                                    Thread.currentThread().interrupt();
                                    throw new IOException("Read interrupted", ie);
                                }
                            }
                            // Optional: Add timeout check
                            if (System.currentTimeMillis() - startReadTime > 300000) { // 5 min timeout per array? Adjust as needed
                                log.error("FATAL: Read timeout (>5min) while reading data for '{}'. Expected {} bytes, read {} bytes.", name, lengthBytes, totalRead);
                                throw new IOException("Read timeout for variable '" + name + "'.");
                            }
                        }
                        long endReadTime = System.currentTimeMillis();
                        if (totalRead != lengthBytes) { // Should be caught by EOF, but double-check
                            log.error("FATAL: Direct NIO read incomplete for '{}'. Expected {}, Read {}.", name, lengthBytes, totalRead);
                            throw new IOException("Direct NIO read incomplete for variable '" + name + "'.");
                        }
                        log.trace("Direct NIO read successful for '{}' ({} bytes in {} ms).", name, totalRead, endReadTime - startReadTime);
                        // --- End Direct NIO Read Path ---
                    } else {
                        // --- Fallback: Read via temporary byte[] chunks + Pointer.memcpy ---
                        log.warn("Using chunked byte[] read fallback for '{}'. Reason: {} large ({} bytes) or direct buffer unavailable.",
                                name, (lengthBytes > Integer.MAX_VALUE || arrayOffsetBytes > Integer.MAX_VALUE || (arrayOffsetBytes + lengthBytes) > Integer.MAX_VALUE) ? "Array > 2GB / offset issue" : "NIO buffer", lengthBytes);

                        if (tempChunk == null)
                            tempChunk = new byte[RAW_IO_CHUNK_SIZE_BYTES]; // Use constant defined elsewhere
                        long bytesReadCount = 0;
                        long targetBufferWriteOffsetBytes = arrayOffsetBytes; // Start writing at the correct offset in the buffer
                        Pointer targetPointer = targetBuffer.pointer();
                        if (targetPointer == null || targetPointer.isNull()) {
                            log.error("FATAL: Cannot get native pointer for target DataBuffer of variable '{}'. Cannot use fallback copy.", name);
                            throw new IOException("Cannot get native pointer for fallback copy for variable '" + name + "'.");
                        }
                        log.trace("Target buffer pointer obtained for fallback write for '{}'.", name);

                        long startReadTime = System.currentTimeMillis();
                        while (bytesReadCount < lengthBytes) {
                            int toRead = (int) Math.min(tempChunk.length, lengthBytes - bytesReadCount);
                            ByteBuffer tempNioWrapper = ByteBuffer.wrap(tempChunk, 0, toRead); // Wrap the chunk for channel read
                            int actuallyRead = 0;
                            long loopStartTime = System.currentTimeMillis();
                            while (tempNioWrapper.hasRemaining()) {
                                int chunkReadBytes = channel.read(tempNioWrapper);
                                if (chunkReadBytes == -1) {
                                    log.error("FATAL: EOFException encountered during chunked read for '{}'. Expected {} bytes, read {} bytes.", name, lengthBytes, bytesReadCount + actuallyRead);
                                    throw new EOFException("EOF encountered during chunked read for variable '" + name + "'.");
                                }
                                actuallyRead += chunkReadBytes;
                                if (chunkReadBytes == 0) { // Avoid busy spin
                                    try {
                                        Thread.sleep(1);
                                    } catch (InterruptedException ie) {
                                        Thread.currentThread().interrupt();
                                        throw new IOException("Read interrupted", ie);
                                    }
                                }
                                // Optional: Timeout check within inner loop
                                if (System.currentTimeMillis() - loopStartTime > 120000) { // 2 min timeout per chunk read? Adjust.
                                    log.error("FATAL: Read timeout (>2min) while reading chunk for '{}'. Expected {} bytes in chunk, read {}.", name, toRead, actuallyRead);
                                    throw new IOException("Read timeout during chunk read for variable '" + name + "'.");
                                }
                            }
                            if (actuallyRead == 0 && bytesReadCount < lengthBytes) {
                                // Should not happen if EOF is checked, but as safety
                                log.warn("Chunk read returned 0 bytes unexpectedly for '{}'. Retrying.", name);
                                continue;
                            }

                            // Copy the read chunk into the target DataBuffer via pointer
                            try (Pointer sourcePointer = new BytePointer(tempChunk)) {
                                // Calculate target position carefully
                                BytePointer targetWritePtr = new BytePointer(targetPointer).position(targetBufferWriteOffsetBytes);
                                Pointer.memcpy(targetWritePtr, sourcePointer, actuallyRead); // Use memcpy
                            }
                            bytesReadCount += actuallyRead;
                            targetBufferWriteOffsetBytes += actuallyRead; // Advance write position in target buffer

                            // Optional: Timeout check for overall array
                            if (System.currentTimeMillis() - startReadTime > 600000) { // 10 min overall timeout? Adjust.
                                log.error("FATAL: Read timeout (>10min) during chunked read fallback for '{}'. Expected {} bytes, read {} bytes.", name, lengthBytes, bytesReadCount);
                                throw new IOException("Overall read timeout during fallback for variable '" + name + "'.");
                            }
                        }
                        long endReadTime = System.currentTimeMillis();
                        if (bytesReadCount != lengthBytes) {
                            log.error("FATAL: Chunked read fallback incomplete for '{}'. Expected {}, Read {}.", name, lengthBytes, bytesReadCount);
                            throw new IOException("Chunked read fallback incomplete for variable '" + name + "'.");
                        }
                        log.trace("Chunked fallback read successful for '{}' ({} bytes in {} ms).", name, bytesReadCount, endReadTime - startReadTime);
                        // --- End Fallback ---
                    }
                } catch (IOException e) {
                    log.error("FATAL: IOException during raw data read for variable '{}' at offset {}", name, offset, e);
                    throw e; // Re-throw IOExceptions
                } catch (Exception e) {
                    log.error("FATAL: Unexpected error during raw data read for variable '{}'", name, e);
                    throw new IOException("Failed loading raw data for variable '" + name + "'", e);
                }
            } else {
                log.debug("Skipping data reading for empty array '{}'", name);
            }


            // --- Associate Array with SameDiff Instance & Verify ---
            log.debug("Associating loaded array with variable '{}' in target SameDiff instance.", name);
            try {
                // Step 1: Call the main method to associate the array
                targetSD.setArrayForVariable(name, resultArr);
                log.trace("Called targetSD.setArrayForVariable('{}', ...)", name);

                // Step 2: Explicitly set in the correct ArrayHolder
                SDVariable varToUpdate = targetSD.getVariable(name); // Get the variable object again
                if (varToUpdate == null) {
                    // This should have been caught earlier, but double-check
                    log.error("CRITICAL: Variable '{}' became null after loading array data!", name);
                    throw new IllegalStateException("Variable '" + name + "' missing after array load.");
                }

                if (varToUpdate.isConstant()) {
                    targetSD.getConstantArrays().setArray(name, resultArr);
                    log.trace("Set array in constantArrays for {}", name);
                } else if (varToUpdate.getVariableType() == VariableType.VARIABLE) {
                    targetSD.getVariablesArrays().setArray(name, resultArr);
                    log.trace("Set array in variablesArrays for {}", name);
                } else {
                    // Handle other types if necessary, e.g., ARRAY
                    targetSD.getEagerArrays().setArray(name, resultArr);
                    log.trace("Set array in eagerArrays for {} (type: {})", name, varToUpdate.getVariableType());
                }

                // --- IMMEDIATE VERIFICATION ---
                INDArray checkArr = targetSD.getArrForVarName(name); // Use the getter
                if (checkArr == null) {
                    log.error("!!!!!!!! VERIFICATION FAILED !!!!!!!!");
                    log.error("CRITICAL: Array is NULL immediately after setting (via getArrForVarName) for variable '{}'!", name);
                    log.error("This indicates a likely bug in SameDiff.setArrayForVariable or SameDiff.getArrForVarName or the ArrayHolders.");
                    // Throw exception to halt the process, as state is inconsistent
                    throw new IllegalStateException("Verification failed: Array is null immediately after setting for variable '" + name + "'.");
                } else {
                    log.debug("Verification Step 1 PASSED: Array is non-NULL via getArrForVarName for '{}'. Shape: {}", name, Arrays.toString(checkArr.shape()));
                    // Optional deeper check: compare references or basic properties
                    if (checkArr != resultArr) {
                        log.warn("Verification Note: getArrForVarName returned a different instance than the loaded one for '{}'. This might be okay if it's a copy/view.", name);
                    } else {
                        log.trace("Verification Detail: getArrForVarName returned the same instance for '{}'.", name);
                    }
                }

                // Verify directly from the holder as well
                INDArray checkHolderArr = null;
                ArrayHolder holderToCheck = null;
                if (varToUpdate.isConstant()) holderToCheck = targetSD.getConstantArrays();
                else if (varToUpdate.getVariableType() == VariableType.VARIABLE)
                    holderToCheck = targetSD.getVariablesArrays();
                // else holderToCheck = targetSD.getEagerArrays(); // Check eager if needed

                if (holderToCheck != null) {
                    checkHolderArr = holderToCheck.getArray(name);
                    if (checkHolderArr == null) {
                        log.error("!!!!!!!! VERIFICATION FAILED !!!!!!!!");
                        log.error("CRITICAL: Array is NULL in the corresponding ArrayHolder ('{}') immediately after setting for '{}'!", holderToCheck.getClass().getSimpleName(), name);
                        throw new IllegalStateException("Verification failed: Array is null in ArrayHolder for variable '" + name + "'.");
                    } else {
                        log.debug("Verification Step 2 PASSED: Array is non-NULL in ArrayHolder for variable '{}'.", name);
                        if (checkHolderArr != resultArr) {
                            log.warn("Verification Note: ArrayHolder contained a different instance than the loaded one for '{}'.", name);
                        } else {
                            log.trace("Verification Detail: ArrayHolder contained the same instance for '{}'.", name);
                        }
                    }
                } else {
                    log.trace("Skipping holder check for var '{}' - type {} doesn't map to checked holders.", name, varToUpdate.getVariableType());
                }
                // --- END IMMEDIATE VERIFICATION ---

                log.debug("Successfully loaded and associated raw data for variable '{}'.", name);

            } catch (Exception e) {
                log.error("FATAL: Error during array association or verification for variable '{}'", name, e);
                throw new IOException("Failed to associate or verify array for variable '" + name + "'", e);
            }

        } // End loop through manifest entries
        log.info("Finished processing manifest and loading appended data.");
    }


    /**
     * Helper to find FlatVariable metadata within a FlatGraph structure.
     *
     * @param fg   The parsed FlatGraph object.
     * @param name The name of the variable to find.
     * @return The FlatVariable object, or null if not found.
     */
    private static FlatVariable findFlatVariableMeta(FlatGraph fg, String name) {
        if (fg == null || name == null) return null;
        // Iterate through the variables vector in the FlatGraph
        for (int i = 0; i < fg.variablesLength(); i++) {
            FlatVariable fv = fg.variables(i); // Access variable at index i
            // Use .equals() for string comparison, checking for nulls
            if (fv != null && name.equals(fv.name())) {
                return fv; // Found the matching variable metadata
            }
        }
        // This can be normal if looking up metadata for a variable in a different shard's FlatGraph
        log.trace("Metadata for variable '{}' not found within the provided FlatGraph metadata.", name);
        return null; // Not found
    }


    private static ByteBuffer serializeMetadataFlatBuffer(
            @NonNull SameDiff sameDiff,
            boolean saveUpdaterState,
            Map<String, String> metadata,
            @NonNull Set<String> largeArrayNamesToExcludeData,
            @NonNull Set<String> smallArrayNamesToIncludeData) throws IOException {

        ExecutorConfiguration configuration = ExecutorConfiguration.builder().outputMode(OutputMode.VARIABLE_SPACE)
                .executionMode(org.nd4j.autodiff.execution.conf.ExecutionMode.SEQUENTIAL)
                .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
                .build();
        Map<String, String> mergedMetadata = enrichMetadata(metadata); // Assume helper exists
        mergedMetadata.put(META_ND4J_FORMAT_TYPE, FORMAT_TYPE_APPENDED);
        mergedMetadata.put(META_ND4J_FORMAT_VERSION, String.valueOf(FILE_VERSION));
        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(1 * 1024 * 1024);

        // --- Metadata Vec ---
        int metadataKeysOffset = 0, metadataValuesOffset = 0;
        if (mergedMetadata != null && !mergedMetadata.isEmpty()) {
            int[] keyOffsets = new int[mergedMetadata.size()];
            int[] valOffsets = new int[mergedMetadata.size()];
            int i = 0;
            List<Map.Entry<String, String>> sortedMeta = new ArrayList<>(mergedMetadata.entrySet());
            sortedMeta.sort(Map.Entry.comparingByKey());
            for (Map.Entry<String, String> entry : sortedMeta) {
                keyOffsets[i] = bufferBuilder.createString(entry.getKey());
                valOffsets[i] = bufferBuilder.createString(entry.getValue() == null ? "" : entry.getValue());
                i++;
            }
            metadataKeysOffset = FlatGraph.createMetadataKeysVector(bufferBuilder, keyOffsets);
            metadataValuesOffset = FlatGraph.createMetadataValuesVector(bufferBuilder, valOffsets);
        }

        // --- Variables Vec & ID Mapping ---
        val flatVariables = new ArrayList<Integer>();
        // Use a stable list of variables present in *this specific* SameDiff instance (graphShard or varShard)
        val variableListForOps = new ArrayList<>(sameDiff.variables());
        val reverseMap = new LinkedHashMap<String, Integer>(); // VarName -> NodeID that produces it (or independent ID)
        val idxForOps = new IdentityHashMap<DifferentialFunction, Integer>(); // Op -> NodeID
        val idCounter = new AtomicInteger(0); // For assigning node IDs

        log.debug("Starting variable iteration for metadata FB ({} vars in this instance)", sameDiff.variables().size());
        for (SDVariable variable : variableListForOps) { // Iterate the stable list
            String varName = variable.name();
            if (varName == null || variable.getVariableType() == VariableType.SEQUENCE)
                continue;
            Variable vMeta = sameDiff.getVariables().get(varName);
            if (vMeta == null) {
                log.warn("Internal metadata missing for variable '{}'. Skipping variable serialization.", varName);
                continue;
            }

            // Assign Node ID (critical for linking ops later)
            int varIdx;
            int outputNum = 0;
            String producingOpName = vMeta.getOutputOfOp();
            DifferentialFunction producingOpFunc = null;
            if (producingOpName != null && sameDiff.getOps().containsKey(producingOpName)) {
                producingOpFunc = sameDiff.getOps().get(producingOpName).getOp();
            }
            if (producingOpFunc != null) {
                if (!idxForOps.containsKey(producingOpFunc)) {
                    varIdx = idCounter.incrementAndGet();
                    idxForOps.put(producingOpFunc, varIdx);
                } else {
                    varIdx = idxForOps.get(producingOpFunc);
                }
                String[] outNames = producingOpFunc.outputVariablesNames();
                outputNum = ArrayUtil.indexOf(outNames, varName);
                if (outputNum < 0) outputNum = 0;
            } else {
                varIdx = idCounter.incrementAndGet();
                outputNum = 0;
            } // Independent node ID
            reverseMap.put(varName, varIdx); // Map name to node ID

            int shapeOffset = 0;
            int nameOffset = bufferBuilder.createString(varName);
            int arrayOffset = 0;
            int idOffset = IntPair.createIntPair(bufferBuilder, varIdx, outputNum);
            byte varTypeByte = FlatBuffersMapper.toVarType(variable.getVariableType());
            DataType dtype = variable.dataType();
            if (dtype == DataType.UNKNOWN && variable.getArr() != null)
                dtype = variable.getArr().dataType();
            if (dtype == DataType.UNKNOWN)
                dtype = DataType.FLOAT;
            byte dtypeByte = FlatBuffersMapper.getDataTypeAsByte(dtype);
            long[] shape = variable.getShape();
            if (shape != null) shapeOffset = FlatVariable.createShapeVector(bufferBuilder, shape);

            // Array Data (Inline Small Only)
            if (smallArrayNamesToIncludeData.contains(varName)) {
                INDArray arr = variable.getArr();
                if (arr != null && !arr.isEmpty()) {
                    try {
                        arrayOffset = serializeSmallNdArrayToFlatBuffer(arr, bufferBuilder);
                        log.trace("Serialized small array inline for '{}', offset={}", varName, arrayOffset);
                    } catch (Exception e) {
                        log.warn("Error serializing small array inline for '{}'.", varName, e);
                        arrayOffset = 0;
                    }
                } else {
                    arrayOffset = 0;
                }
            } else {
                arrayOffset = 0;
            } // Large arrays have data appended

            int controlDepsOffset = 0, controlDepsForOpOffset = 0, controlDepsForVarOffset = 0;
            int[] cds = FlatBuffersMapper.mapOrNull(vMeta.getControlDeps(), bufferBuilder);
            if (cds != null) controlDepsOffset = FlatVariable.createControlDepsVector(bufferBuilder, cds);
            int[] cdsForOp = FlatBuffersMapper.mapOrNull(vMeta.getControlDepsForOp(), bufferBuilder);
            if (cdsForOp != null)
                controlDepsForOpOffset = FlatVariable.createControlDepForOpVector(bufferBuilder, cdsForOp);
            int[] cdsForVar = FlatBuffersMapper.mapOrNull(vMeta.getControlDepsForVar(), bufferBuilder);
            if (cdsForVar != null)
                controlDepsForVarOffset = FlatVariable.createControlDepsForVarVector(bufferBuilder, cdsForVar);
            flatVariables.add(FlatVariable.createFlatVariable(
                    bufferBuilder,
                    idOffset,
                    nameOffset,
                    dtypeByte,
                    shapeOffset,
                    arrayOffset,
                    -1,
                    varTypeByte,
                    controlDepsOffset,
                    controlDepsForOpOffset,
                    controlDepsForVarOffset));
        }
        log.debug("serializeMetadataFlatBuffer: Finished variable iteration. flatVariables.size() = {} (Expected ~{})", flatVariables.size(), sameDiff.variables().size());
        if (flatVariables.isEmpty() && sameDiff.variables().size() > 0) {
            log.warn("Variable processing loop resulted in empty flatVariables list, but original SameDiff had variables!");
        }
        int variablesVectorOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));


        // --- Ops Vec ---
        val flatNodes = new ArrayList<Integer>();
        log.debug("Starting op iteration for metadata FB ({} ops in this instance)", sameDiff.getOps().size());
        Map<String, Integer> forwardMap = new HashMap<>();
        Map<String, Integer> framesMap = new HashMap<>(); // Required by asFlatNode
        for (SameDiffOp op : sameDiff.getOps().values()) {
            DifferentialFunction df = op.getOp();
            if (df == null) {
                log.warn("Skipping op '{}' with null function.", op.getName());
                continue;
            }
            Integer fnId = idxForOps.get(df); // Get ID assigned via output variables
            if (fnId == null) { // Should only happen for ops with NO output vars? Or error in var loop?
                fnId = idCounter.incrementAndGet(); // Assign new ID if needed
                idxForOps.put(df, fnId);
                log.warn("Op '{}' ({}) was not assigned an ID via its outputs. Assigning new ID {}. Check graph structure/linking.", op.getName(), df.opName(), fnId);
            }
            try {
                flatNodes.add(FlatBuffersMapper.asFlatNode(sameDiff, df, bufferBuilder, variableListForOps, reverseMap, forwardMap, framesMap, idCounter, fnId)); // Assume FlatBuffersMapper exists and is correct
            } catch (Exception e) {
                throw new IOException("Failed to serialize node: " + op.getName(), e);
            }
        }
        log.debug("serializeMetadataFlatBuffer: Finished op iteration. flatNodes.size() = {} (Expected ~{})", flatNodes.size(), sameDiff.getOps().size());
        if (flatNodes.isEmpty() && sameDiff.getOps().size() > 0) {
            log.warn("Op processing loop resulted in empty flatNodes list, but original SameDiff had ops!");
        }
        int nodesVectorOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(flatNodes));

        // --- Other Graph Components ---
        int outputsVectorOffset = 0;
        if(sameDiff.outputs() != null && !sameDiff.outputs().isEmpty()) {
            int[] outputsOffsets = new int[sameDiff.outputs().size()];
            for(int i = 0; i < sameDiff.outputs().size(); i++) {
                outputsOffsets[i] = bufferBuilder.createString(sameDiff.outputs().get(i));
            }
            outputsVectorOffset = FlatGraph.createOutputsVector(bufferBuilder, outputsOffsets);
        } else {
            outputsVectorOffset = FlatGraph.createOutputsVector(bufferBuilder, new int[0]);
        }

        int placeholdersVectorOffset = createPlaceholdersVector(sameDiff, bufferBuilder); // Assume helper exists
        int lossVariablesVectorOffset = createLossVariablesVector(sameDiff, bufferBuilder); // Assume helper exists
        int trainingConfigStringOffset = createTrainingConfigOffset(sameDiff, bufferBuilder); // Assume helper exists
        // Use helper that takes boolean only
        int updaterStateVectorOffset = createUpdaterStateOffset(sameDiff, bufferBuilder, saveUpdaterState); // Assume helper exists
        int configurationTableOffset = configuration.getFlatConfiguration(bufferBuilder);


        // --- Finalize FlatBuffer ---
        log.debug("serializeMetadataFlatBuffer: Finalizing FlatBuffer. VarVecOffset={}, NodeVecOffset={}, PlaceholderVecOffset={}, LossVecOffset={}, UpdaterStateVecOffset={}",
                variablesVectorOffset, nodesVectorOffset, placeholdersVectorOffset, lossVariablesVectorOffset, updaterStateVectorOffset); // Log key offsets
        int fg = FlatGraph.createFlatGraph(bufferBuilder,
                0, // Graph ID - use 0 for metadata block?
                variablesVectorOffset, nodesVectorOffset, outputsVectorOffset, configurationTableOffset,
                placeholdersVectorOffset, lossVariablesVectorOffset, trainingConfigStringOffset, updaterStateVectorOffset,
                metadataKeysOffset, metadataValuesOffset);
        bufferBuilder.finish(fg);
        ByteBuffer resultBuffer = bufferBuilder.dataBuffer();
        log.debug("serializeMetadataFlatBuffer: Finished. Result buffer remaining size: {}", resultBuffer.remaining());
        if (resultBuffer.remaining() == 0 && (sameDiff.variables().size() > 0 || sameDiff.getOps().size() > 0)) {
            log.error("CRITICAL: serializeMetadataFlatBuffer produced an empty buffer for a non-empty SameDiff instance!");
        }
        return resultBuffer;
    }

    /**
     * Serializes a small INDArray (non-scalar, non-empty) to a FlatBuffer buffer vector.
     * Uses Native Endian byte order. Includes enhanced byte verification.
     * ADDED: FATAL validation of shape buffer extras to catch corrupt arrays from import.
     */
    public static int serializeSmallNdArrayToFlatBuffer(@NonNull INDArray arr, @NonNull FlatBufferBuilder builder) throws IOException {
        // Try to get a somewhat identifiable name/string for logging
        String varNameForLog = "InlineArray"; // Default
        try {
            String id = String.valueOf(arr.getId()); // Assuming getId() exists or adapt as needed
            varNameForLog = (id != null && !id.isEmpty() && !id.matches("unnamed_array_.*")) ? id : arr.shapeInfoToString();
        } catch (Exception e) {
            try { varNameForLog = arr.shapeInfoToString(); } catch (Exception e2) { /* ignore */ }
        }

        log.info("SERIALIZE_INLINE: Attempting for Var='{}', Shape={}, DType={}", varNameForLog, Arrays.toString(arr.shape()), arr.dataType());

        // CRITICAL VALIDATION: Check shape buffer integrity BEFORE any processing
        try {
            // Try to access the array's shape info through the Shape utility
            long[] shapeInfo = arr.shapeInfoDataBuffer().asLong();
            if (shapeInfo != null && shapeInfo.length > 0) {
                // The extras value is typically encoded in the shape buffer
                // Try to extract the data type from the shape buffer using ArrayOptionsHelper
                try {
                    // Use the shape info to validate the data type encoding
                    DataType extractedType = ArrayOptionsHelper.dataType(shapeInfo);
                    if (extractedType == null || extractedType == DataType.UNKNOWN) {
                        throw new IllegalStateException(String.format(
                                "FATAL SERIALIZATION ERROR: INVALID SHAPE BUFFER DETECTED for array '%s'. " +
                                        "ArrayOptionsHelper.dataType() returned NULL or UNKNOWN. " +
                                        "Full ShapeInfo: %s. " +
                                        "Expected DataType: %s. " +
                                        "This indicates CORRUPT array metadata from ONNX import or array creation process. " +
                                        "TERMINATING PROCESS TO PREVENT DATA CORRUPTION.",
                                varNameForLog, Arrays.toString(shapeInfo), arr.dataType()));
                    }

                    // Additional validation: extracted type should match array's reported type
                    if (extractedType != arr.dataType()) {
                        throw new IllegalStateException(String.format(
                                "FATAL SERIALIZATION ERROR: SHAPE BUFFER DATA TYPE MISMATCH for array '%s'. " +
                                        "Shape buffer indicates DataType: %s, " +
                                        "but array reports DataType: %s. " +
                                        "Full ShapeInfo: %s. " +
                                        "This indicates CORRUPT array metadata from ONNX import or array creation process. " +
                                        "TERMINATING PROCESS TO PREVENT DATA CORRUPTION.",
                                varNameForLog, extractedType, arr.dataType(), Arrays.toString(shapeInfo)));
                    }

                    log.debug("SERIALIZE_VALIDATION [{}]: Shape buffer validation PASSED. DataType: {}", varNameForLog, extractedType);

                } catch (ND4JUnknownDataTypeException e) {
                    throw new IllegalStateException(String.format(
                            "FATAL SERIALIZATION ERROR: INVALID DATA TYPE ENCODING in shape buffer for array '%s'. " +
                                    "Full ShapeInfo: %s. " +
                                    "ArrayOptionsHelper.dataType() threw: %s. " +
                                    "This indicates CORRUPT array metadata from ONNX import or array creation process. " +
                                    "TERMINATING PROCESS TO PREVENT DATA CORRUPTION.",
                            varNameForLog, Arrays.toString(shapeInfo), e.getMessage()), e);
                }
            } else {
                throw new IllegalStateException(String.format(
                        "FATAL SERIALIZATION ERROR: NULL OR EMPTY shape info buffer for array '%s'. " +
                                "This indicates SEVERELY CORRUPT array metadata. " +
                                "TERMINATING PROCESS TO PREVENT DATA CORRUPTION.",
                        varNameForLog));
            }
        } catch (Exception e) {
            if (e instanceof IllegalStateException) {
                throw e; // Re-throw our fatal errors
            }
            throw new IllegalStateException(String.format(
                    "FATAL SERIALIZATION ERROR: Exception accessing shape info for array '%s': %s. " +
                            "This indicates SEVERELY CORRUPT array state. " +
                            "TERMINATING PROCESS TO PREVENT DATA CORRUPTION.",
                    varNameForLog, e.getMessage()), e);
        }

        // Skip arrays we can't handle safely
        if (arr.dataType() == DataType.UTF8 || arr.dataType() == DataType.COMPRESSED || arr.dataType() == DataType.UNKNOWN) {
            log.warn("SERIALIZE_INLINE [{}]: Cannot serialize array of type {} inline. Skipping.", varNameForLog, arr.dataType());
            return 0;
        }

        long[] shape = arr.shape();
        int rank = arr.rank();
        boolean isScalar = rank == 0;

        // Check dimension validity
        if (!isScalar && shape != null) {
            for (long d : shape) {
                if (d < 0 || d > Integer.MAX_VALUE) {
                    log.error("SERIALIZE_INLINE [{}]: Invalid shape dimension: {}. Skipping serialization.", varNameForLog, Arrays.toString(shape));
                    return 0;
                }
            }
        } else if (shape == null && !isScalar && !arr.isEmpty()) {
            log.error("SERIALIZE_INLINE [{}]: Invalid null shape for non-scalar, non-empty inline array. Skipping serialization.", varNameForLog);
            return 0;
        }

        // Skip scalars for now
        if (isScalar) {
            log.debug("SERIALIZE_INLINE [{}]: Skipping scalar array serialization inline.", varNameForLog);
            return 0;
        }

        // Handle empty arrays (shape only)
        if (arr.isEmpty()) {
            log.debug("SERIALIZE_INLINE [{}]: Converting empty array to shape-only metadata.", varNameForLog);
            int shapeOffset = FlatArray.createShapeVector(builder, shape);
            byte dtype = FlatBuffersMapper.getDataTypeAsByte(arr.dataType());
            byte order = (byte)(arr.ordering() == 'c' ? 0 : 1);
            int finalOffset = FlatArray.createFlatArray(builder, shapeOffset, 0, dtype, order, 0, 0, 0, false);
            log.info("SERIALIZE_INLINE: SUCCESS (Empty) for Var='{}'. Returning offset {}.", varNameForLog, finalOffset);
            return finalOffset;
        }

        DataBuffer dataBuffer = arr.data();


        // --- Standard non-scalar, non-empty array handling ---
        try {
            int shapeOffset = FlatArray.createShapeVector(builder, shape);
            byte dtype = FlatBuffersMapper.getDataTypeAsByte(arr.dataType());
            byte order = (byte)(arr.ordering() == 'c' ? 0 : 1);
            int bufferOffset = 0; // FB offset for the data vector

            ByteBuffer nioBuffer = dataBuffer.asNio(); // Get NIO view
            long lengthBytes = arr.length() * dataBuffer.getElementSize();

            if (nioBuffer != null) {
                nioBuffer.order(ByteOrder.nativeOrder()); // Ensure NATIVE order
                long arrOffsetBytes = arr.offset() * dataBuffer.getElementSize(); // Use INDArray offset

                // Check if view is usable and within limits
                if (arrOffsetBytes >= 0 && arrOffsetBytes <= Integer.MAX_VALUE &&
                        lengthBytes >= 0 && lengthBytes <= Integer.MAX_VALUE &&
                        (arrOffsetBytes + lengthBytes) >= 0 && (arrOffsetBytes + lengthBytes) <= Integer.MAX_VALUE)
                {
                    // Create a new byte array to hold the native-ordered bytes
                    byte[] nativeBytes = new byte[(int)lengthBytes];
                    // Position the NIO buffer view correctly
                    nioBuffer.position((int)arrOffsetBytes);
                    nioBuffer.limit((int)(arrOffsetBytes + lengthBytes));
                    // Read the native-ordered bytes into the temporary array
                    try {
                        nioBuffer.get(nativeBytes);
                    } catch (Exception e) {
                        log.error("SERIALIZE_INLINE [{}]: Exception during nioBuffer.get(nativeBytes)!", varNameForLog, e);
                        return 0; // Fail if extraction fails
                    }

                    // *** ENHANCED BYTE CHECK ***
                    ByteBuffer checkNioBuffer = dataBuffer.asNio(); // Get a fresh view for checking
                    if (checkNioBuffer != null) {
                        boolean checkIterationPassed = true; // Local check status
                        int checkLength = nativeBytes.length;
                        checkNioBuffer.order(ByteOrder.nativeOrder());
                        checkNioBuffer.position((int) arrOffsetBytes); // Position to start of array data
                        checkNioBuffer.limit((int) (arrOffsetBytes + checkLength)); // Limit to array data

                        log.debug("SERIALIZE_CHECK [{}]: Performing byte-by-byte verification ({} bytes)...", varNameForLog, checkLength);
                        for (int j = 0; j < checkLength; j++) {
                            if (!checkNioBuffer.hasRemaining()) {
                                log.error("SERIALIZE_CHECK [{}]: Check buffer ran out unexpectedly at byte {}/{}", varNameForLog, j, checkLength);
                                checkIterationPassed = false; break;
                            }
                            byte expectedByte = checkNioBuffer.get(); // Read expected byte
                            if (nativeBytes[j] != expectedByte) { // Compare with extracted byte
                                log.error("SERIALIZE_CHECK [{}]: Byte mismatch at index {}! Expected:{}, Got:{}. Stopping check.",
                                        varNameForLog, j, String.format("%02X", expectedByte), String.format("%02X", nativeBytes[j]));
                                checkIterationPassed = false;
                                // Log context (optional, ensure bytesToHex exists and is safe)
                                try {
                                    int start = Math.max(0, j - 8);
                                    int end = Math.min(checkLength, j + 8);
                                    ByteBuffer contextBuffer = dataBuffer.asNio(); // Fresh buffer for context log
                                    byte[] expectedContext = new byte[end - start];
                                    if(contextBuffer != null) {
                                        contextBuffer.order(ByteOrder.nativeOrder());
                                        contextBuffer.position((int)arrOffsetBytes + start);
                                        contextBuffer.limit((int)arrOffsetBytes + end);
                                        contextBuffer.get(expectedContext);
                                        log.error("SERIALIZE_CHECK [{}]: Context (Expected): {}", varNameForLog, bytesToHex(expectedContext, 0, expectedContext.length));
                                    } else {
                                        log.error("SERIALIZE_CHECK [{}]: Context (Expected): Could not get buffer view for context log.", varNameForLog);
                                    }
                                    log.error("SERIALIZE_CHECK [{}]: Context (Actual Extracted):   {}", varNameForLog, bytesToHex(nativeBytes, start, end - start));
                                } catch (Exception logEx) { log.warn("SERIALIZE_CHECK [{}]: Error logging context.", varNameForLog, logEx); }
                                break; // Stop check on first mismatch
                            }
                        }
                        if(checkIterationPassed) {
                            log.info("SERIALIZE_CHECK [{}]: Full native byte extraction check PASSED ({} bytes).", varNameForLog, checkLength);
                        } else {
                            throw new IllegalStateException(String.format(
                                    "FATAL SERIALIZATION ERROR: Byte verification FAILED for array '%s'. " +
                                            "Extracted bytes do not match original buffer contents. " +
                                            "This indicates MEMORY CORRUPTION or BUFFER INCONSISTENCY. " +
                                            "TERMINATING PROCESS TO PREVENT DATA CORRUPTION.",
                                    varNameForLog));
                        }
                    } else {
                        log.warn("SERIALIZE_CHECK [{}]: Cannot perform full byte check because asNio() returned null for check. Proceeding without verification.", varNameForLog);
                        // If verification is critical, you might want to return 0 here instead.
                        // For now, proceeding but logging the warning.
                    }
                    // *** END ENHANCED BYTE CHECK ***

                    // Insert into FlatBuffer only if bytes were prepared
                    if (nativeBytes.length > 0) {
                        bufferOffset = FlatArray.createBufferVector(builder, nativeBytes);
                        log.trace("SERIALIZE_INLINE [{}]: Prepared FlatBuffer vector from {} native bytes.", varNameForLog, nativeBytes.length);
                    } else {
                        // This case should generally not happen if arr.isEmpty() check passed earlier
                        log.warn("SERIALIZE_INLINE [{}]: Extracted zero bytes for non-empty inline array. Buffer offset will be 0.", varNameForLog);
                        bufferOffset = 0;
                    }
                } else {
                    log.error("SERIALIZE_INLINE [{}]: Cannot serialize inline array: Offset/length ({}/{}) exceeds limits for NIO buffer.", varNameForLog, arrOffsetBytes, lengthBytes);
                    return 0; // Indicate failure
                }
            } else {
                log.error("SERIALIZE_INLINE [{}]: Cannot serialize inline array: Direct NIO buffer not available (asNio=null).", varNameForLog);
                return 0; // Indicate failure
            }

            // Create the final FlatArray object
            int finalOffset = FlatArray.createFlatArray(builder, shapeOffset, bufferOffset, dtype, order, 0, 0, 0, false);

            // Log success/failure based on whether data was actually embedded
            if (bufferOffset > 0) {
                log.info("SERIALIZE_INLINE: SUCCESS for Var='{}'. Returning offset {}.", varNameForLog, finalOffset);
            } else if (arr.length() > 0 && bufferOffset == 0) {
                // If array wasn't empty but we failed to get a buffer offset
                log.error("SERIALIZE_INLINE: FAILURE for Var='{}'. Failed to create buffer vector. Returning 0.", varNameForLog);
                return 0; // Return 0 to indicate failure
            } else {
                // Array was empty or buffer offset was legitimately 0 (should only be for empty arrays handled earlier)
                log.warn("SERIALIZE_INLINE: SUCCESS (No Data?) for Var='{}'. BufferOffset is 0. Returning offset {}.", varNameForLog, finalOffset);
            }
            return finalOffset;

        } catch (IllegalStateException e) {
            // Re-throw IllegalStateException (our fatal validation errors) without wrapping
            throw e;
        } catch (Exception e) {
            log.error("SERIALIZE_INLINE [{}]: Unhandled error serializing inline array: {}", varNameForLog, e.getMessage(), e);
            return 0; // Return 0 on any error
        }
    }

    /**
     * Deserializes an inline INDArray from FlatBuffer data. Expects Native Endian bytes.
     * MODIFIED: Reads bytes manually using fa.buffer(j) instead of fa.bufferAsByteBuffer()
     * to bypass potential issues with the buffer view method.
     *
     * @param fa      FlatBuffer FlatArray object containing inline data.
     * @param varName The name of the variable being deserialized (for logging).
     * @return Deserialized INDArray or null on failure.
     */
    private static INDArray deserializeSmallNdArrayFromInlineBuffer(FlatArray fa, String varName) throws IOException {
        if (fa == null) {
            log.trace("LOAD_INLINE [{}]: FlatArray object is null. Returning null.", varName);
            return null;
        }

        // Handle empty array case first (based on shape, no data buffer expected/needed)
        if (fa.shapeLength() > 0 && fa.bufferLength() == 0 && fa.bufferChunksLength() == 0 && !fa.isExternal()) {
            // This case means shape info exists, but data buffer vector is explicitly empty.
            try {
                long[] shape = new long[fa.shapeLength()];
                for (int i = 0; i < shape.length; i++) { shape[i] = fa.shape(i); }
                byte dtypeByte = fa.dtype();
                DataType dataType = FlatBuffersMapper.getDataTypeFromByte(dtypeByte);
                if (dataType == null || dataType == DataType.UNKNOWN) { log.debug("LOAD_INLINE [{}]: Empty FlatArray has unrecognized dtype ({}). Defaulting to FLOAT.", varName, dtypeByte); dataType = DataType.FLOAT; }
                char order = fa.byteOrder() == 0 ? 'c' : 'f';
                long numElements = ArrayUtil.prod(shape); // Calculate elements from shape
                if (numElements != 0) {
                    log.warn("LOAD_INLINE [{}]: Shape {} implies {} elements, but FlatBuffer data length is 0. Creating empty array.", varName, Arrays.toString(shape), numElements);
                } else {
                    log.trace("LOAD_INLINE [{}]: Creating empty array from shape-only FlatArray. Shape {}, Dtype {}, Order {}", varName, Arrays.toString(shape), dataType, order);
                }
                return Nd4j.create(dataType, shape, order); // Create based on shape
            } catch (Exception e) { log.error("LOAD_INLINE [{}]: Failed to create empty array from shape-only FlatArray: {}", varName, e.getMessage(), e); return null; }
        }


        // Handle inline buffer with data - Must have bufferLength > 0 now
        if (fa.bufferLength() > 0 && fa.bufferChunksLength() == 0 && !fa.isExternal()) {
            String shapeForLog = "N/A";
            try {
                // Step 1: Handle scalar array (rank 0) - Assuming not saved
                if (fa.shapeLength() == 0) {
                    log.warn("LOAD_INLINE [{}]: Deserializing scalar from inline buffer - this was skipped during recommended save. Returning null.", varName);
                    return null;
                }

                // Step 2: Non-scalar arrays - Get metadata
                long[] shape = new long[fa.shapeLength()];
                for (int i = 0; i < shape.length; i++) { shape[i] = fa.shape(i); }
                shapeForLog = Arrays.toString(shape);

                byte dtypeByte = fa.dtype();
                DataType dataType = FlatBuffersMapper.getDataTypeFromByte(dtypeByte);
                if (dataType == null || dataType == DataType.UNKNOWN) { log.warn("LOAD_INLINE [{}]: Unrecognized dtype ({}) in inline FlatArray for shape {}. Defaulting to FLOAT.", varName, dtypeByte, shapeForLog); dataType = DataType.FLOAT; }
                char order = fa.byteOrder() == 0 ? 'c' : 'f';

                // Calculate expected size
                int elementSize = dataType.width();
                long totalElements = ArrayUtil.prod(shape);
                long expectedBytes = totalElements * elementSize;

                // Get buffer length reported by FlatBuffers metadata
                int fbBufferLength = fa.bufferLength();
                log.trace("LOAD_INLINE [{}]: Shape={}, DType={}, Order={}, ExpectedBytes={}, FlatBufferLength={}",
                        varName, shapeForLog, dataType, order, expectedBytes, fbBufferLength);

                // Validate size from metadata against expected size
                if (expectedBytes != fbBufferLength) {
                    log.error("LOAD_INLINE [{}]: FlatBuffer data length mismatch! Shape {} requires {} bytes, but FlatBuffer metadata reports length {}.",
                            varName, shapeForLog, expectedBytes, fbBufferLength);
                    return null; // Cannot proceed if sizes don't match
                }

                // If empty array based on shape, return empty array
                if (totalElements == 0) {
                    log.trace("LOAD_INLINE [{}]: Shape {} implies zero elements. Returning empty array.", varName, shapeForLog);
                    return Nd4j.create(dataType, shape, order);
                }

                // *** Read bytes manually using fa.buffer(j) ***
                log.debug("LOAD_INLINE [{}]: Reading {} bytes manually using fa.buffer(j)...", varName, expectedBytes);
                byte[] readBytes = new byte[(int)expectedBytes]; // Cast to int safe due to previous checks on shape dimensions
                boolean readSuccess = true;
                try {
                    for(int j=0; j<expectedBytes; j++) {
                        // Get unsigned byte and cast to signed byte for Java array
                        readBytes[j] = (byte)fa.buffer(j);
                    }
                } catch (Exception e) {
                    log.error("LOAD_INLINE [{}]: Exception during manual byte reading fa.buffer(j).", varName, e);
                    readSuccess = false;
                }

                if (!readSuccess) {
                    log.error("LOAD_INLINE [{}]: Failed to read bytes manually from FlatBuffer.", varName);
                    return null; // Deserialization failed
                }
                log.debug("LOAD_INLINE [{}]: Manual byte reading complete.", varName);

                // Wrap the manually read bytes and create the INDArray
                ByteBuffer bbManual = ByteBuffer.wrap(readBytes).order(ByteOrder.nativeOrder());

                INDArray result = Nd4j.create(dataType, shape, order);
                DataBuffer targetBuffer = result.data();
                if (targetBuffer == null) { log.error("LOAD_INLINE [{}]: Target DataBuffer is null after creating array shape {}.", varName, shapeForLog); return null; }

                // Copy data from bbManual to targetBuffer
                ByteBuffer targetNio = targetBuffer.asNio();
                if(targetNio != null && result.offset() == 0 ) {
                    log.trace("LOAD_INLINE [{}]: Using bulk NIO copy (Target Offset is 0) from manually read bytes.", varName);
                    targetNio.order(ByteOrder.nativeOrder());
                    targetNio.position(0);
                    targetNio.limit((int) expectedBytes);
                    bbManual.limit((int) expectedBytes); // Set limit on source
                    try {
                        targetNio.put(bbManual); // Bulk copy
                    } catch (Exception e) { log.error("LOAD_INLINE [{}]: Exception during bulk NIO copy from manual bytes!", varName, e); return null; }
                    log.trace("LOAD_INLINE [{}]: Bulk NIO copy finished from manual bytes.", varName);
                } else {
                    // Fallback to element-wise copy from manually read buffer
                    log.warn("LOAD_INLINE [{}]: Using element-wise copy for shape {} from manually read bytes. Reason: Target NIO buffer null? {}, Target Offset = {}",
                            varName, shapeForLog, (targetNio == null), result.offset());
                    try {
                        for (long i = 0; i < totalElements; i++) {
                            if (bbManual.remaining() < elementSize) { log.error("LOAD_INLINE [{}]: Manual ByteBuffer ran out of data unexpectedly...", varName); break; }
                            switch (dataType) {
                                case FLOAT: result.putScalar(i, bbManual.getFloat()); break;
                                case DOUBLE: result.putScalar(i, bbManual.getDouble()); break;
                                // ... other cases as before, reading from bbManual ...
                                case INT: result.putScalar(i, bbManual.getInt()); break;
                                case LONG: result.putScalar(i, bbManual.getLong()); break;
                                case SHORT: result.putScalar(i, bbManual.getShort()); break;
                                case BYTE: result.putScalar(i, bbManual.get()); break;
                                case UBYTE: result.putScalar(i, bbManual.get() & 0xFF); break;
                                case UINT16: result.putScalar(i, bbManual.getShort() & 0xFFFF); break;
                                case UINT32: result.putScalar(i, bbManual.getInt() & 0xFFFFFFFFL); break;
                                case UINT64: result.putScalar(i, bbManual.getLong()); break;
                                case BOOL: result.putScalar(i, bbManual.get() != 0); break;
                                case BFLOAT16: case HALF:
                                    if (bbManual.remaining() >= 2) { result.putScalar(i, HalfPrecisionUtil.toFloat(bbManual.getShort())); }
                                    else { log.error("LOAD_INLINE [{}]: Insufficient bytes for HALF/BFLOAT16...", varName); i = totalElements; }
                                    break;
                                default: log.warn("LOAD_INLINE [{}]: Skipping unsupported type {}...", varName, dataType); bbManual.position(bbManual.position() + elementSize);
                            }
                        }
                    } catch (Exception e) { log.warn("LOAD_INLINE [{}]: Error during element-wise copy from manual bytes...", varName, e); }
                }

                return result; // Return the populated array

            } catch (Exception e) {
                log.error("LOAD_INLINE [{}]: Failed to deserialize inline FlatArray (Shape {}): {}", varName, shapeForLog, e.getMessage(), e);
                return null;
            }
        }

        // If FlatArray doesn't match expected inline structure
        log.warn("LOAD_INLINE [{}]: FlatArray structure did not match expected inline format. BufferLength={}, BufferChunks={}, IsExternal={}. Returning null.",
                varName, fa.bufferLength(), fa.bufferChunksLength(), fa.isExternal());
        return null;
    } // end deserializeSmallNdArrayFromInlineBuffer

    // --- Other Helpers ---
    private static Map<String, String> enrichMetadata(Map<String, String> userMetadata) {
        Map<String, String> enriched = new HashMap<>();
        enriched.put(META_ND4J_FORMAT_VERSION, String.valueOf(FILE_VERSION));
        enriched.put(META_ND4J_FORMAT_TYPE, FORMAT_TYPE_APPENDED);
        enriched.put(META_CREATION_TIME, String.valueOf(System.currentTimeMillis()));
        if (frameworkVersion == null) {
            synchronized(fwVersionLock) {
                if(frameworkVersion == null)
                    frameworkVersion = getFrameworkVersionInternal();
            }
        }
        if (!"unknown".equals(frameworkVersion))
            enriched.put(META_FRAMEWORK_VERSION, frameworkVersion);
        if (userMetadata != null)
            enriched.putAll(userMetadata);
        return enriched;
    }

    private static String getFrameworkVersionInternal() {
        try {
            Package p = Nd4j.class.getPackage();
            return p != null ? p.getImplementationVersion() : "unknown";
        } catch (Exception e) {
            return "unknown";
        }
    }
    /**
     * Helper to get variables with data, sorted descending by approximate byte size.
     */
    private static List<SDVariable> getVariablesWithDataSorted(SameDiff sameDiff) {
        List<SDVariable> varsWithData = new ArrayList<>();
        if (sameDiff == null || sameDiff.variables() == null) return varsWithData;

        for (SDVariable var : sameDiff.variables()) {
            // Include only VARIABLE and CONSTANT types with actual, non-empty data
            if ((var.getVariableType() == VariableType.VARIABLE || var.getVariableType() == VariableType.CONSTANT)
                    && var.getArr() != null && !var.getArr().isEmpty()) {
                varsWithData.add(var);
            }
        }
        // Sort descending by approximate size in bytes
        varsWithData.sort((a, b) -> {
            long sizeA = a.getArr().length() * a.dataType().width();
            long sizeB = b.getArr().length() * b.dataType().width();
            return Long.compare(sizeB, sizeA); // Descending
        });
        return varsWithData;
    }

    /**
     * Creates a SameDiff instance containing only the graph structure (ops) and variable
     * definitions (stubs, including shape/type/control deps), but no large data arrays.
     * Uses shallow copy for Ops - WARNING: This might break ops with internal state if
     * the original SameDiff instance is modified after creating the shard.
     *
     * @param sameDiff The original SameDiff instance.
     * @return A new SameDiff instance representing the graph structure.
     */
    private static SameDiff createGraphShard(@NonNull SameDiff sameDiff) {
        log.debug("Creating graph shard structure from original SameDiff instance...");
        SameDiff graphShard = SameDiff.create();
        graphShard.setLogExecution(false); // Don't log ops during this internal build

        // --- 1. Shallow Copy Ops ---
        log.debug("Shallow copying {} operations for graph shard...", sameDiff.getOps().size());
        int opsCopiedCount = 0;
        int opsFailedCount = 0;
        for (Map.Entry<String, SameDiffOp> opEntry : sameDiff.getOps().entrySet()) {
            String opOwnName = opEntry.getKey();
            SameDiffOp originalOp = opEntry.getValue();
            DifferentialFunction originalDf = originalOp.getOp();

            if (originalDf == null) {
                log.warn("Original SameDiffOp metadata for '{}' has a null DifferentialFunction. Skipping copy.", opOwnName);
                opsFailedCount++;
                continue;
            }

            try {
                // Associate original op instance with the new graphShard
                // WARNING: Re-parenting the original op!
                originalDf.setSameDiff(graphShard);

                // Add to graphShard's op map, reusing the original DF instance
                graphShard.getOps().put(opOwnName, SameDiffOp.builder()
                        .name(opOwnName)
                        .op(originalDf) // Use the original function instance
                        .inputsToOp(copyList(originalOp.getInputsToOp())) // Copy metadata lists
                        .outputsOfOp(copyList(originalOp.getOutputsOfOp()))
                        .controlDeps(copyList(originalOp.getControlDeps()))
                        .controlDepFor(copyList(originalOp.getControlDepFor()))
                        .varControlDeps(copyList(originalOp.getVarControlDeps()))
                        .build());
                opsCopiedCount++;
                log.trace("Shallow copied op '{}' into graph shard.", opOwnName);
            } catch (Exception e) {
                // Catch potential errors during metadata copying or map insertion
                log.error("Failed during shallow copy setup for op '{}' ({}). Graph shard might be incomplete.", opOwnName, originalDf.opName(), e);
                opsFailedCount++;
            }
        }
        log.debug("Finished shallow copying operations. Copied: {}, Failed/Skipped: {}", opsCopiedCount, opsFailedCount);
        // --- End Op Copying ---

        // --- 2. Create Stubs for ALL Variables ---
        log.debug("Creating variable stubs for graph shard ({} total variables)...", sameDiff.variables().size());
        int stubsCreated = 0;
        for (SDVariable var : sameDiff.variables()) {
            String name = var.name();
            if (name == null) {
                log.warn("Original variable has null name. Skipping stub creation.");
                continue;
            }
            // Skip if already added (shouldn't happen if ops don't add vars)
            if (graphShard.hasVariable(name)) {
                log.trace("Variable stub '{}' already exists, skipping.", name);
                continue;
            }

            DataType dtype = var.dataType(); if (dtype == null || dtype == DataType.UNKNOWN) dtype = DataType.FLOAT; // Default
            long[] shape = var.getShape() != null ? var.getShape().clone() : null; // Clone shape
            VariableType type = var.getVariableType();

            SDVariable stub;
            switch (type) {
                case PLACEHOLDER: graphShard.placeHolder(name, dtype, shape); break;
                case CONSTANT: case VARIABLE: case ARRAY:
                    stub = new SDVariable(name, type, graphShard, shape, dtype); graphShard.addVariable(stub); break;
                case SEQUENCE: log.trace("Skipping SEQUENCE var '{}'", name); continue;
                default: log.warn("Unhandled VariableType '{}' for var '{}'.", type, name); continue;
            }
            stubsCreated++;

            // Copy structural metadata (control dependencies)
            Variable originalVarMeta = sameDiff.getVariables().get(name);
            Variable stubVarMeta = graphShard.getVariables().get(name); // Get the *new* metadata obj
            if (originalVarMeta != null && stubVarMeta != null) {
                stubVarMeta.setControlDeps(copyList(originalVarMeta.getControlDeps()));
                stubVarMeta.setControlDepsForOp(copyList(originalVarMeta.getControlDepsForOp()));
                stubVarMeta.setControlDepsForVar(copyList(originalVarMeta.getControlDepsForVar()));
            } else if (originalVarMeta != null) { log.warn("Metadata mismatch for var '{}'", name); }
        }
        log.debug("Finished creating variable stubs. Created {} stubs. Graph shard variable count: {}", stubsCreated, graphShard.variables().size());
        // --- End Stub Creation ---


        // --- 3. Establish Op -> Output Variable Links within graphShard ---
        // This ensures that variable stubs know which *copied* op produces them.
        log.debug("Establishing op -> output variable links within graphShard...");
        int linksEstablished = 0; int linksFailed = 0;
        for (Map.Entry<String, Variable> entry : sameDiff.getVariables().entrySet()) { // Iterate original map for links
            String varName = entry.getKey(); Variable oMeta = entry.getValue(); String prodOpName = oMeta.getOutputOfOp();
            if (prodOpName != null) { // If it was an output of an op
                Variable sMeta = graphShard.getVariables().get(varName); // Find the corresponding stub
                boolean opExists = graphShard.getOps().containsKey(prodOpName); // Check if the op was copied
                if (sMeta != null && opExists) {
                    sMeta.setOutputOfOp(prodOpName); // Set the link on the stub's metadata
                    linksEstablished++;
                    log.trace("Linked graphShard var '{}' as output of op '{}'", varName, prodOpName);
                } else {
                    linksFailed++;
                    // Log which part failed (stub or op)
                    log.warn("Could not establish link for variable '{}' as output of op '{}' in graphShard: Stub exists? {}, Copied Op exists? {}",
                            varName, prodOpName, (sMeta != null), opExists);
                }
            }
        }
        log.debug("Finished establishing links. Established: {}, Failed: {}", linksEstablished, linksFailed);
        // --- End Link Establishment ---


        // --- 4. Set Training Config and Loss Variables ---
        if (sameDiff.getTrainingConfig() != null) {
            try {
                graphShard.setTrainingConfig(sameDiff.getTrainingConfig());
            }
            catch (Exception e) {
                graphShard.setTrainingConfig(sameDiff.getTrainingConfig());
                log.warn("Could not clone TrainingConfig", e);
            }
        }
        if (sameDiff.getLossVariables() != null) {
            for (String lossVar : sameDiff.getLossVariables()) {
                if(graphShard.hasVariable(lossVar)) {
                    graphShard.addLossVariable(lossVar);
                }
                else {
                    log.warn("Loss variable '{}' missing in graph shard stubs. Cannot mark as loss.", lossVar);
                }
            }
        }

        log.debug("Graph shard creation complete. Variables: {}, Ops: {}", graphShard.variables().size(), graphShard.getOps().size());
        // Sanity check counts
        if(graphShard.variables().size() != sameDiff.variables().size())
            log.warn("Variable count mismatch! Original: {}, Shard: {}", sameDiff.variables().size(), graphShard.variables().size());
        if(graphShard.getOps().size() != sameDiff.getOps().size())
            log.warn("Op count mismatch! Original: {}, Shard: {}", sameDiff.getOps().size(), graphShard.getOps().size());

        return graphShard;
    }

    /**
     * Helper to save a single variable shard.
     * MODIFIED to accept the map of large arrays to append for this shard.
     */
    private static void saveVariableShardHelper(
            @NonNull SameDiff varShard, // Contains stubs + small inline arrays for this shard
            @NonNull Map<String, INDArray> arraysToAppend, // ACTUAL large arrays for this shard
            @NonNull Map<String, GradientUpdater> shardUpdaterMap,
            @NonNull String baseName, int shardIndex,
            @NonNull Map<String, String> baseMetadata, @NonNull File parentDir,
            @NonNull List<File> savedShardFilesList, boolean saveUpdaterStateGlobal) throws IOException {

        Map<String, String> shardMetadata = new HashMap<>(baseMetadata);
        shardMetadata.put(META_SHARD_INDEX, String.valueOf(shardIndex));
        shardMetadata.put(META_SHARD_TYPE, "weights");

        boolean saveUpdaterForThisShard = false;
        if (saveUpdaterStateGlobal && shardUpdaterMap != null && !shardUpdaterMap.isEmpty()) {
            // Set the filtered map on varShard using reflection before serializing metadata
            try {
                Field updaterMapField = SameDiff.class.getDeclaredField("updaterMap");
                updaterMapField.setAccessible(true);
                updaterMapField.set(varShard, shardUpdaterMap); // Set on the stub SameDiff
                Field initField = SameDiff.class.getDeclaredField("initializedTraining");
                initField.setAccessible(true);
                initField.set(varShard, true);
                saveUpdaterForThisShard = true; // Mark for metadata serialization
            } catch (Exception e) {
                log.error("Failed to set filtered updater map on variable shard {} via reflection. Updater state will not be saved in metadata for this shard.", shardIndex, e);
            }
        } else if (saveUpdaterStateGlobal && shardContainsTrainableParams(varShard)) {
            log.warn("Updater state saving was requested, but no relevant updater state found for trainable parameters in variable shard {}.", shardIndex);
        }

        String tempShardName = String.format("%s.shard%d-of-%s.sdnb", baseName, shardIndex, "N");
        File shardFile = new File(parentDir, tempShardName);

        log.info("Saving variable shard {} ({} vars defined, {} large arrays to append) to temporary name: {}",
                shardIndex, varShard.variables().size(), arraysToAppend.size(), shardFile.getName());
        try {
            // *** Call the modified saveInternal, passing the arrays map ***
            saveInternal(varShard, shardFile, saveUpdaterForThisShard, shardMetadata, arraysToAppend);

            savedShardFilesList.add(shardFile);
            log.info(">>> Successfully saved variable shard {} and added {} to tracking list. List size now: {} <<<",
                    shardIndex, shardFile.getName(), savedShardFilesList.size());
        } catch(Exception e) {
            log.error("saveInternal FAILED for variable shard {}. File {} might be incomplete or missing. NOT adding to saved list.", shardIndex, shardFile.getName(), e);
            throw new IOException("Failed saving intermediate shard " + shardIndex + " to file " + shardFile.getAbsolutePath(), e);
        }
    }

    /**
     * Helper to rename shard files with the final count.
     * Throws IOException if renaming fails for any file.
     *
     * @param baseName Base name for files.
     * @param parentDir Directory containing the files.
     * @param tempShard0File The temporarily named graph shard file (e.g., shard0-of-N.sdnb).
     * @param tempVariableShardFiles List of temporarily named variable shard files.
     * @param finalTotalShards The final determined total number of shards.
     * @throws IOException if any rename operation fails.
     */
    private static void renameShardFiles(String baseName, File parentDir, File tempShard0File, List<File> tempVariableShardFiles, int finalTotalShards) throws IOException {
        boolean overallSuccess = true;
        List<String> failedRenames = new ArrayList<>();

        // --- Rename shard 0 ---
        String finalShard0Name = String.format("%s.shard0-of-%d.sdnb", baseName, finalTotalShards);
        File finalShard0File = new File(parentDir, finalShard0Name);
        if (!tempShard0File.exists()) {
            // This shouldn't happen if saving succeeded
            log.error("Temporary graph shard file '{}' does not exist before renaming.", tempShard0File.getAbsolutePath());
            overallSuccess = false;
            failedRenames.add(tempShard0File.getName() + " (source missing)");
        } else if (!tempShard0File.equals(finalShard0File)) {
            log.debug("Renaming graph shard '{}' to '{}'", tempShard0File.getName(), finalShard0File.getName());
            // Attempt to delete target first if it exists (robustness)
            if (finalShard0File.exists() && !finalShard0File.delete()) {
                log.warn("Could not delete existing target file '{}' before renaming shard 0.", finalShard0File.getName());
                // Continue attempt, renameTo might handle it or fail
            }

            if (!tempShard0File.renameTo(finalShard0File)) {
                log.error("FAILED to rename graph shard file from '{}' to '{}'. Check permissions/locks in {}", tempShard0File.getName(), finalShard0File.getName(), parentDir.getAbsolutePath());
                overallSuccess = false;
                failedRenames.add(tempShard0File.getName() + " -> " + finalShard0File.getName());
            } else {
                log.debug("Successfully renamed graph shard to {}", finalShard0File.getName());
            }
        } else {
            log.debug("Graph shard temporary name already matches final name: {}", finalShard0File.getName());
        }

        // --- Rename variable shards ---
        for (int i = 0; i < tempVariableShardFiles.size(); i++) {
            File oldFile = tempVariableShardFiles.get(i);
            int shardIdx = i + 1; // Variable shards are 1-based index
            String finalName = String.format("%s.shard%d-of-%d.sdnb", baseName, shardIdx, finalTotalShards);
            File finalFile = new File(parentDir, finalName);

            if (!oldFile.exists()) {
                log.error("Temporary variable shard file '{}' does not exist before renaming.", oldFile.getAbsolutePath());
                overallSuccess = false;
                failedRenames.add(oldFile.getName() + " (source missing)");
                continue;
            } else if (!oldFile.equals(finalFile)) {
                log.debug("Renaming variable shard {} from '{}' to '{}'", shardIdx, oldFile.getName(), finalFile.getName());
                if (finalFile.exists() && !finalFile.delete()) {
                    log.warn("Could not delete existing target file '{}' before renaming shard {}.", finalFile.getName(), shardIdx);
                }

                if (!oldFile.renameTo(finalFile)) {
                    log.error("FAILED to rename variable shard file from '{}' to '{}'. Check permissions/locks in {}", oldFile.getName(), finalFile.getName(), parentDir.getAbsolutePath());
                    overallSuccess = false;
                    failedRenames.add(oldFile.getName() + " -> " + finalFile.getName());
                } else {
                    log.debug("Successfully renamed variable shard {} to {}", shardIdx, finalFile.getName());
                }
            } else {
                log.debug("Variable shard {} temporary name already matches final name: {}", shardIdx, finalFile.getName());
            }
        }

        // If any rename failed, throw an exception to signal that saving is incomplete/corrupt
        if (!overallSuccess) {
            throw new IOException("Failed to rename one or more shard files to reflect final count of " + finalTotalShards +
                    ". Failed renames: " + failedRenames +
                    ". Check directory permissions and file locks: " + parentDir.getAbsolutePath());
        }
    }

    private static boolean shardContainsTrainableParams(SameDiff shard) {
        for (SDVariable var : shard.variables()) { if (var.getVariableType() == VariableType.VARIABLE)
            return true;
        }
        return false;
    }
    private static long calculateBaseMetadataSizeEstimate(SameDiff sd) {
        return 5*1024*1024 + (sd != null ? sd.getOps().size() * 1024 + sd.variables().size() * 256 : 0);
    }
    private static long calculateVariableMetadataSizeEstimate(SDVariable var) {
        return 512 + (var.name() != null ? var.name().length()*2 : 0) + (var.getShape() != null ? var.getShape().length * 8 : 0);
    }
    private static long calculateInlineArrayOverheadEstimate(INDArray arr) {
        return 256 + (arr != null ? (long)arr.dataType().width() : 0);
    } // Increased estimate
    private static List<String> copyList(List<String> list) {
        return list == null ? null : new ArrayList<>(list);
    }
    private static boolean isValidSdnbFile(File file) {
        if(!file.exists() || !file.isFile() || file.length() < HEADER_SIZE)
            return false;
        try(FileInputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis)) {
            byte[] magicRead = new byte[FILE_MAGIC.length];
            dis.readFully(magicRead);
            return Arrays.equals(FILE_MAGIC, magicRead); }
        catch (IOException e) {
            return false;
        }
    }
    private static int detectShardCount(File parentDir, String baseName) throws IOException { int numShards = -1; final String filePrefix = baseName + ".shard";
        File[] matchingFiles = parentDir.listFiles((dir, name) -> name.startsWith(filePrefix) && name.endsWith(".sdnb"));
        if (matchingFiles == null || matchingFiles.length == 0)
            throw new FileNotFoundException("No shard files matching pattern '" + filePrefix + "*.sdnb' found in " + parentDir.getAbsolutePath());
        Pattern p = Pattern.compile(Pattern.quote(baseName) + "\\.shard\\d+-of-(\\d+)\\.sdnb$");
        for (File f : matchingFiles) { Matcher m = p.matcher(f.getName());
            if (m.matches()) {
                try {
                    int count = Integer.parseInt(m.group(1));
                    if (numShards == -1)
                        numShards = count;
                    else if (numShards != count)
                        throw new IOException("Inconsistent shard counts (expected " + numShards + " found " + count + " in " + f.getName() + ")");
                } catch (NumberFormatException e) {
                    log.warn("Bad shard count in {}. Skipping.", f.getName());
                }
            }
        }
        if (numShards <= 0)
            throw new IOException("No valid shard count found in " + parentDir.getAbsolutePath());
        return numShards;
    }
    private static int createPlaceholdersVector(SameDiff sameDiff, FlatBufferBuilder bufferBuilder) {
        List<String> phList = new ArrayList<>();
        // Ensure variables map is accessed correctly
        if(sameDiff.getVariables() == null) return 0; // Should not happen in valid SameDiff
        for (Variable vMeta : sameDiff.getVariables().values()) {
            SDVariable v = vMeta.getVariable();
            if (v != null && v.isPlaceHolder() && v.name() != null) {
                phList.add(v.name());
            }
        }
        if (phList.isEmpty()) return 0; // Return 0 offset for empty vector
        int[] offsets = new int[phList.size()];
        // Create strings in reverse order for FlatBuffers efficiency
        for (int i = phList.size() - 1; i >= 0; i--) {
            offsets[i] = bufferBuilder.createString(phList.get(i));
        }
        return FlatGraph.createPlaceholdersVector(bufferBuilder, offsets);
    }

    private static int createLossVariablesVector(SameDiff sameDiff, FlatBufferBuilder bufferBuilder) {
        List<String> lossVars = sameDiff.getLossVariables();
        if (lossVars == null || lossVars.isEmpty()) return 0; // Return 0 offset for empty vector

        // Filter out potential nulls just in case
        List<String> nonNullLossVars = lossVars.stream().filter(Objects::nonNull).collect(Collectors.toList());
        if (nonNullLossVars.isEmpty()) return 0;

        int[] offsets = new int[nonNullLossVars.size()];
        // Create strings in reverse order
        for (int i = nonNullLossVars.size() - 1; i >= 0; i--) {
            offsets[i] = bufferBuilder.createString(nonNullLossVars.get(i));
        }
        return FlatGraph.createLossVariablesVector(bufferBuilder, offsets);
    }

    private static int createTrainingConfigOffset(SameDiff sameDiff, FlatBufferBuilder bufferBuilder) {
        TrainingConfig tc = sameDiff.getTrainingConfig();
        if (tc != null) {
            String json = tc.toJson();
            if (json != null && !json.isEmpty()) {
                // CreateString automatically handles caching if the same JSON appears multiple times
                return bufferBuilder.createString(json);
            }
        }
        return 0; // Return 0 offset if no config or empty JSON
    }

    /**
     * Helper to create Updater State Vector Offset.
     * Assumes updater state INDArrays are small enough to be serialized inline.
     */
    @SneakyThrows
    private static int createUpdaterStateOffset(SameDiff sameDiff, FlatBufferBuilder bufferBuilder,
                                                boolean includeUpdaterState) { // Removed largeArrayNamesToExcludeData
        if (!includeUpdaterState || sameDiff.getUpdaterMap() == null || sameDiff.getUpdaterMap().isEmpty()) {
            return 0; // Return 0 offset if not saving state or map is empty/null
        }

        List<Integer> updaterOffsetsList = new ArrayList<>(); // Stores offsets of individual UpdaterState tables
        // Iterate through a sorted view for deterministic order
        List<Map.Entry<String, GradientUpdater>> sortedUpdaters = new ArrayList<>(sameDiff.getUpdaterMap().entrySet());
        sortedUpdaters.sort(Map.Entry.comparingByKey());

        for (Map.Entry<String, GradientUpdater> g : sortedUpdaters) {
            String paramName = g.getKey();
            GradientUpdater updater = g.getValue();
            if(paramName == null || updater == null) {
                log.warn("Skipping updater state for null parameter name or updater object (Param: {}).", paramName);
                continue;
            }
            // Ensure the variable this updater state belongs to actually exists
            if (!sameDiff.hasVariable(paramName)) {
                log.warn("Skipping updater state for parameter '{}' as the variable does not exist in the SameDiff instance.", paramName);
                continue;
            }


            int paramNameOffset = bufferBuilder.createString(paramName);
            int stateKeyVectorOffset = 0; // Default to 0 (empty vector)
            int stateValuesVectorOffset = 0; // Default to 0 (empty vector)
            Map<String, INDArray> state = updater.getState(); // Get the state map for this updater

            if (state != null && !state.isEmpty()) {
                List<Integer> keysOffsetsList = new ArrayList<>();
                List<Integer> valuesOffsetsList = new ArrayList<>();
                // Sort state entries for deterministic order
                List<Map.Entry<String, INDArray>> sortedState = new ArrayList<>(state.entrySet());
                sortedState.sort(Map.Entry.comparingByKey());

                for (Map.Entry<String, INDArray> e : sortedState) {
                    String key = e.getKey();
                    INDArray stateArr = e.getValue();
                    if (key == null || stateArr == null) {
                        log.warn("Skipping null key or null state array in updater for parameter '{}'. Key: {}", paramName, key);
                        continue;
                    }
                    // Updater state arrays are typically small, serialize inline using the dedicated helper
                    int arrOffset = serializeSmallNdArrayToFlatBuffer(stateArr, bufferBuilder); // Assume helper exists
                    if (arrOffset != 0) { // Only add if serialization succeeded and produced a non-empty array offset
                        keysOffsetsList.add(bufferBuilder.createString(key));
                        valuesOffsetsList.add(arrOffset);
                    } else {
                        log.warn("Failed to serialize updater state array '{}' for parameter '{}'. Skipping this state entry.", key, paramName);
                    }
                }
                // Create vectors only if we successfully serialized some state entries
                if (!keysOffsetsList.isEmpty()) {
                    // Build vectors in reverse for FlatBuffers builder internals
                    int[] keysOffsetsArray = new int[keysOffsetsList.size()];
                    for(int i=keysOffsetsList.size()-1; i>=0; i--) keysOffsetsArray[i] = keysOffsetsList.get(i);
                    stateKeyVectorOffset = UpdaterState.createUpdaterStateKeysVector(bufferBuilder, keysOffsetsArray);

                    int[] valuesOffsetsArray = new int[valuesOffsetsList.size()];
                    for(int i=valuesOffsetsList.size()-1; i>=0; i--) valuesOffsetsArray[i] = valuesOffsetsList.get(i);
                    stateValuesVectorOffset = UpdaterState.createUpdaterStateValuesVector(bufferBuilder, valuesOffsetsArray);
                }
            }
            // Create the UpdaterState table object
            updaterOffsetsList.add(UpdaterState.createUpdaterState(bufferBuilder, paramNameOffset, stateKeyVectorOffset, stateValuesVectorOffset));
        } // End loop over updaters

        if (updaterOffsetsList.isEmpty()) return 0; // Return 0 if no updaters were actually serialized

        // Create the final vector containing offsets to all the UpdaterState tables
        // Build in reverse order
        int[] finalUpdaterOffsets = new int[updaterOffsetsList.size()];
        for(int i=updaterOffsetsList.size()-1; i>=0; i--) finalUpdaterOffsets[i] = updaterOffsetsList.get(i);
        return FlatGraph.createUpdaterStateVector(bufferBuilder, finalUpdaterOffsets);
    }


} // End of SameDiffSerializer class