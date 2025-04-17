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
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.*; // Import base package
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.io.ReflectionUtils;
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
    public static final String META_MODEL_NAME = "model.name";
    public static final String META_MODEL_VERSION = "model.version";
    public static final String META_FRAMEWORK_VERSION = "framework.version";
    public static final String META_CREATION_TIME = "creation.time";
    public static final String META_AUTHOR = "author";
    public static final String META_LICENSE = "license";
    public static final String META_DESCRIPTION = "description";
    public static final String META_TAGS = "tags";
    public static final String META_SHARD_INDEX = "shard.index";
    public static final String META_SHARD_COUNT = "shard.count";
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
     * CORRECTED to call the modified saveInternal for the single-file case.
     */
    @SneakyThrows
    public static void saveAutoShard(@NonNull SameDiff sameDiff, @NonNull File baseFile,
                                     boolean saveUpdaterState, Map<String, String> metadata) throws IOException {
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
     * Includes detailed logging for layer_0_b assignment.
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
                    // *** ADDED LOGGING FOR layer_0_b HERE ***
                    if (var.name().equals("layer_0_b")) {
                        log.info("SAVE_TRACE [Shard {}]: Assigning 'layer_0_b' (small). Inline? {}", currentShardIndex, !appendData);
                        log.info("SAVE_TRACE [Shard {}]: Adding 'layer_0_b' data inline via currentVarShard.var/constant().", currentShardIndex);
                    }
                    // Add with data
                    if (var.getVariableType() == VariableType.CONSTANT) {
                        currentVarShard.constant(var.name(), arr.dup());
                    } else { // VARIABLE or ARRAY
                        currentVarShard.var(var.name(), arr.dup());
                    }
                } else {
                    // *** ADDED LOGGING FOR layer_0_b HERE (Warning Case) ***
                    if (var.name().equals("layer_0_b")) {
                        log.warn("SAVE_TRACE [Shard {}]: 'layer_0_b' classified as large (appendData=true)? This is unexpected! SizeBytes={}", currentShardIndex, varSizeBytes);
                    }
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
            return loadInternal(modelFile, loadUpdaterState, null);
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
            result = loadInternal(shard0File, false, SameDiff.create()); // Creates the base SameDiff instance
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
                // *** CORRECT CALL for variable shards: Pass 'result' to populate ***
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
     * MODIFIED: Added detailed logging and checks around metadata buffer read/write
     * to debug the "Metadata write position error".
     * Fallback save path uses NIO with Native Order.
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
     * @throws InterruptedException If thread sleep is interrupted during write waits.
     */
    private static void saveInternal(
            @NonNull SameDiff sameDiff, @NonNull File file,
            boolean saveUpdaterState, Map<String, String> metadata,
            Map<String, INDArray> externalArraysToAppend // Map of arrays to append
    ) throws IOException, InterruptedException {

        Map<String, Pair<Long, Long>> largeArrayManifest = new LinkedHashMap<>();
        Set<String> largeArrayNamesForMetadata = new HashSet<>();
        Set<String> smallInlineArrayNamesForMetadata = new HashSet<>();

        // 1. Identify arrays for metadata serialization (Same logic as before)
        // ... (code omitted for brevity, assumed unchanged from previous correct version) ...
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
        // ... (code omitted for brevity, assumed unchanged from previous correct version) ...
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
                for(int i=0; i<checkLen; i++) { if (metadataBytes[i] != 0) { hasData = true; break; } }
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
                                            long nioChunkWriteStart = System.currentTimeMillis();

                                            while (totalWrittenThisChunk < chunkLengthBytes) {
                                                long writtenNow = channel.write(nioBufferView);
                                                if (writtenNow < 0) throw new IOException("FileChannel write error (fallback chunk) for " + name);
                                                totalWrittenThisChunk += writtenNow;
                                                if (writtenNow == 0 /* ... */) { Thread.sleep(1); /* Check timeout */ }
                                                else { nioChunkWriteStart = System.currentTimeMillis(); }
                                            }
                                            if (totalWrittenThisChunk != chunkLengthBytes) throw new IOException("Fallback NIO Write incomplete for chunk of " + name + "...");

                                            bytesWritten += totalWrittenThisChunk;
                                            log.trace("SAVE [{}]: Fallback NIO write successful for chunk ({} bytes). Total written so far: {}", name, totalWrittenThisChunk, bytesWritten);
                                        } catch (IOException | InterruptedException e) {
                                            if(e instanceof InterruptedException) Thread.currentThread().interrupt();
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
            try (ByteArrayOutputStream baos = new ByteArrayOutputStream(); ObjectOutputStream oos = new ObjectOutputStream(baos)) {
                oos.writeObject(largeArrayManifest); manifestBytes = baos.toByteArray();
            } catch (IOException e) { throw new IOException("Failed to serialize manifest", e); }
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
        log.debug("loadInternal: Loading internal format from file: {}", file.getAbsolutePath());
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
            // TODO: Check version compatibility if needed
            manifestOffset = headerBuffer.getLong();
            manifestLength = headerBuffer.getLong();
            metadataOffset = headerBuffer.getLong();
            log.debug("loadInternal: Read Header - ManifestOffset={}, ManifestLength={}, MetadataOffset={}", manifestOffset, manifestLength, metadataOffset);

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
            log.debug("loadInternal: Calculated MetadataLength={}", metadataLength);


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
                    log.debug("loadInternal: Successfully deserialized manifest with {} entries.", manifest.size());
                } catch (Exception e) {
                    throw new IOException("Failed to deserialize manifest from file: " + file.getAbsolutePath(), e);
                }
            } else {
                manifest = Collections.emptyMap(); // Handle case of no appended data
                log.debug("loadInternal: Manifest length is zero in file {}. No appended data expected.", file.getName());
            }


            // --- Read Metadata Buffer ---
            FlatGraph fg = null; // Parsed FlatGraph metadata
            if (metadataLength > 0) {
                log.debug("loadInternal: Allocating direct buffer for metadata ({} bytes)", metadataLength);
                metadataBuffer = ByteBuffer.allocateDirect((int) metadataLength).order(ByteOrder.LITTLE_ENDIAN); // FlatBuffers standard
                int metaRead = channel.read(metadataBuffer, metadataOffset);
                log.debug("loadInternal: Read {} bytes from channel into metadata buffer.", metaRead);
                if (metaRead != metadataLength)
                    throw new IOException("Failed to read complete metadata FlatBuffer from: " + file.getAbsolutePath());
                metadataBuffer.flip(); // Prepare for reading
                log.debug("loadInternal: Metadata buffer flipped. Position={}, Limit={}", metadataBuffer.position(), metadataBuffer.limit());
                // Parse FlatGraph once, used by multiple steps below
                try {
                    log.debug("loadInternal: Attempting to parse FlatGraph from metadata buffer.");
                    fg = FlatGraph.getRootAsFlatGraph(metadataBuffer.duplicate()); // Use duplicate to preserve original buffer position
                    if (fg == null) throw new IOException("Failed to get FlatGraph root from metadata ByteBuffer.");
                    log.debug("loadInternal: Successfully parsed FlatGraph metadata. VariablesLength={}, NodesLength={}", fg.variablesLength(), fg.nodesLength());
                } catch (Exception e) {
                    throw new IOException("Error parsing FlatBuffer metadata from file: " + file.getAbsolutePath(), e);
                }
            } else {
                log.warn("loadInternal: Metadata length is zero in file {}. Cannot deserialize graph structure or load inline arrays.", file.getName());
                if (existingSD == null) {
                    throw new IOException("Cannot create new SameDiff instance: metadata is empty in file " + file.getAbsolutePath());
                }
            }


            // --- Create or Use SameDiff Instance ---
            SameDiff targetSD;
            if (existingSD == null) {
                // --- Case 1: Create NEW SameDiff instance (Shard 0 or single file) ---
                log.debug("loadInternal: existingSD is null. Attempting to deserialize NEW instance from metadata.");
                if (metadataBuffer == null || metadataLength == 0) {
                    throw new IOException("Cannot create new SameDiff instance: metadata is empty or failed to read in file " + file.getAbsolutePath());
                }
                targetSD = deserializeFromFlatBuffers(metadataBuffer.duplicate(), loadUpdaterState, manifest); // Pass manifest to identify non-inline
                log.info("loadInternal: deserializeFromFlatBuffers completed for NEW instance (hashCode: {}). Variable count: {}", targetSD.hashCode(), targetSD.variables().size());

                // *** ADDED VERIFICATION LOGGING ***
                log.debug("loadInternal: Checking variables in NEWLY loaded targetSD (hashCode: {}) before loading appended data:", targetSD.hashCode());
                if (manifest != null) { // Check only vars expected in manifest
                    for (String varName : manifest.keySet()) {
                        boolean hasVar = targetSD.hasVariable(varName);
                        log.debug("loadInternal: Check: targetSD.hasVariable('{}') = {}", varName, hasVar);
                        // Add more verbose error if the critical var is missing
                        if (varName.equals("layer_0_w") && !hasVar) {
                            log.error("CRITICAL CHECK FAILED in loadInternal (New Instance Path): targetSD does NOT contain 'layer_0_w' after deserializeFromFlatBuffers!");
                            log.error("All variables present in targetSD: {}", targetSD.variableNames());
                        }
                    }
                } else {
                    log.warn("loadInternal: Manifest is null, cannot perform variable check before loading appended data.");
                }
                // *** END VERIFICATION LOGGING ***

            } else {
                // --- Case 2: Populate EXISTING SameDiff instance (Variable Shard) ---
                log.debug("loadInternal: Populating EXISTING SameDiff instance (hashCode: {}) from variable shard file {}", existingSD.hashCode(), file.getName());
                targetSD = existingSD; // Use the passed-in instance as the target

                // Load small inline arrays defined in *this shard's* metadata
                if (fg != null) { // Check if metadata was successfully parsed
                    log.debug("loadInternal: Calling loadSmallInlineArraysIntoExisting for targetSD (hashCode: {})", targetSD.hashCode());
                    loadSmallInlineArraysIntoExisting(targetSD, fg, manifest);
                    // Load and merge updater state if requested
                    if (loadUpdaterState) {
                        log.debug("loadInternal: Calling loadAndUpdateUpdaterState for targetSD (hashCode: {})", targetSD.hashCode());
                        loadAndUpdateUpdaterState(targetSD, fg);
                    }
                } else {
                    log.warn("loadInternal: Cannot load small inline arrays or updater state for existing SameDiff: metadata was empty or failed to parse in file {}", file.getName());
                }

                // *** ADDED VERIFICATION LOGGING (Existing Instance Path) ***
                log.debug("loadInternal: Checking variables in EXISTING targetSD (hashCode: {}) before loading appended data:", targetSD.hashCode());
                if (manifest != null) {
                    for(String varName : manifest.keySet()){
                        log.debug("loadInternal: Check: targetSD.hasVariable('{}') = {}", varName, targetSD.hasVariable(varName));
                    }
                } else {
                    log.warn("loadInternal: Manifest is null, cannot perform variable check before loading appended data (Existing Instance Path).");
                }
                // *** END VERIFICATION LOGGING ***
            }

            // --- Load Appended Array Data (Common to both cases) ---
            if (manifest != null && !manifest.isEmpty()) {
                log.info("loadInternal: Calling loadAppendedArrayData for {} variables into targetSD (hashCode: {})", manifest.size(), targetSD.hashCode());
                if (metadataBuffer == null) {
                    throw new IOException("Cannot load appended array data: metadata buffer is required for lookups but is missing or empty in file " + file.getAbsolutePath());
                }
                loadAppendedArrayData(targetSD, manifest, channel, metadataBuffer.duplicate()); // Pass duplicate buffer
                log.info("loadInternal: loadAppendedArrayData finished for targetSD (hashCode: {})", targetSD.hashCode());
            } else {
                log.info("loadInternal: No appended array data found in manifest for file {}. Skipping loadAppendedArrayData.", file.getName());
            }

            log.debug("loadInternal: Load process complete for file {}. Returning targetSD (hashCode: {})", file.getName(), targetSD.hashCode());
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

            // *** Specific logging for layer_0_b ***
            boolean isTargetVar = name.equals("layer_0_b");
            if (isTargetVar) log.info("LOAD_INLINE_TRACE: Processing metadata for 'layer_0_b'.");

            // Skip if this variable's data is expected to be appended (handled elsewhere)
            if (manifest.containsKey(name)) {
                skippedManifestCount++;
                if (isTargetVar) log.info("LOAD_INLINE_TRACE: 'layer_0_b' is IN MANIFEST. Skipping inline load.", name);
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
            if (targetVar == null) { // Should not happen if hasVariable passed, but check
                log.error("LOAD_INLINE: Variable '{}' exists (hasVariable=true) but getVariable is null! Skipping.", name);
                errorCount++;
                continue;
            }
            if (targetVar.getArr() != null) {
                skippedExistingCount++;
                if (isTargetVar) log.info("LOAD_INLINE_TRACE: 'layer_0_b' targetVar already has an array. Skipping inline load.", name);
                continue; // Skip inline load attempt
            }

            // Check if this FlatVariable actually contains inline array data
            FlatArray fa = fv.ndarray();
            if (isTargetVar) log.info("LOAD_INLINE_TRACE: 'layer_0_b' FlatArray metadata (fa) is null? {}", (fa == null));

            if (fa != null) {
                if (isTargetVar) { // More details for the target var
                    log.info("LOAD_INLINE_TRACE: 'layer_0_b' fa.bufferLength={}, fa.bufferChunksLength={}, fa.isExternal={}", fa.bufferLength(), fa.bufferChunksLength(), fa.isExternal());
                    // Get buffer info safely
                    ByteBuffer bb_trace = null;
                    try { bb_trace = fa.bufferAsByteBuffer(); } catch (Exception e) { log.warn("LOAD_INLINE_TRACE: Error getting bufferAsByteBuffer for layer_0_b trace: {}", e.getMessage()); }
                    log.info("LOAD_INLINE_TRACE: 'layer_0_b' bufferAsByteBuffer is null? {}, remaining={}", (bb_trace == null), (bb_trace != null ? bb_trace.remaining() : "N/A"));
                }
                log.debug("LOAD_INLINE: Found inline FlatArray metadata for '{}'. Attempting deserialization.", name);
                try {
                    // Use the potentially simplified deserializeSmallNdArrayFromInlineBuffer
                    INDArray smallArr = deserializeSmallNdArrayFromInlineBuffer(fa,name); // Call the corrected version

                    // *** Check result specifically for layer_0_b ***
                    if (isTargetVar) {
                        log.info("LOAD_INLINE_TRACE: deserializeSmallNdArrayFromInlineBuffer returned for 'layer_0_b'. Is null? {}", (smallArr == null));
                        if (smallArr != null) {
                            log.info("LOAD_INLINE_TRACE: 'layer_0_b' deserialized array shape: {}", Arrays.toString(smallArr.shape()));
                            // Maybe log first element? Be careful with types
                            try { log.info("LOAD_INLINE_TRACE: 'layer_0_b' first element value (as float): {}", smallArr.getFloat(0)); } catch (Exception e) {/*ignore*/}
                        }
                    }

                    if (smallArr != null) {
                        log.info("LOAD_INLINE: Successfully deserialized inline array for '{}'. Shape: {}", name, Arrays.toString(smallArr.shape()));

                        // Perform consistency checks (dtype, shape)
                        DataType expectedDtype = targetVar.dataType();
                        if (expectedDtype != null && expectedDtype != DataType.UNKNOWN && smallArr.dataType() != expectedDtype) {
                            log.warn("LOAD_INLINE: Data type mismatch for small inline array '{}'. Expected {}, Found {}. Attempting cast.", name, expectedDtype, smallArr.dataType());
                            try { smallArr = smallArr.castTo(expectedDtype); } catch (Exception castEx) { log.error("LOAD_INLINE: Failed to cast array '{}' to {}.", name, expectedDtype, castEx); errorCount++; continue; }
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
                if (isTargetVar) log.info("LOAD_INLINE_TRACE: No inline FlatArray metadata found for 'layer_0_b' (fa == null).", name);
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
            // TODO: Set fileMetadata on sd instance if needed (e.g., via reflection or dedicated method)
            // sd.setMetadata(fileMetadata); // Example hypothetical setter
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
     * MODIFIED: Force fallback path using memcpy instead of direct NIO read.
     *
     * @param targetSD       The SameDiff instance to populate (can be newly created or existing).
     * @param manifest       Map mapping variable names to {offset, length} in the file channel.
     * @param channel        FileChannel positioned at the start of the file (seeking will be done internally).
     * @param metadataBuffer ByteBuffer (positioned at 0, with limit=metadataLength) containing the
     * parsed FlatBuffers metadata for the entire shard. Must be Little Endian.
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
                log.error("FATAL: Manifest contains entry for variable '{}' but it was not found in the target SameDiff instance's graph structure. Cannot load data. Variables present: {}", name, targetSD.variableNames());
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
                log.warn("Variable '{}' already has an array in target SameDiff instance. Skipping append load for this entry (Offset={}, Length={}). Check save logic if this is unexpected.", name, offset, lengthBytes);
                continue;
            }

            // --- Get Metadata from SDVariable AND FlatVariable ---
            DataType dtype = var.dataType();
            long[] shape = var.getShape();
            char order;

            FlatVariable fv = findFlatVariableMeta(fg, name);
            if (fv != null) {
                // (Consistency checks for dtype/shape as before)
                DataType fbDtype = FlatBuffersMapper.getDataTypeFromByte(fv.dtype());
                if (dtype != null && dtype != DataType.UNKNOWN && fbDtype != dtype) {
                    log.warn("DataType mismatch for '{}': Target SD has {}, Shard Metadata has {}. Using Target SD type.", name, dtype, fbDtype);
                } else if (dtype == null || dtype == DataType.UNKNOWN) {
                    dtype = fbDtype;
                }
                long[] fbShape = null;
                if (fv.shapeLength() > 0) {
                    fbShape = new long[fv.shapeLength()];
                    for (int j = 0; j < fbShape.length; j++) fbShape[j] = fv.shape(j);
                }
                if (shape != null && fbShape != null && !Arrays.equals(shape, fbShape)) {
                    log.error("Shape mismatch for '{}': Target SD has {}, Shard Metadata has {}. Cannot safely load data.", name, Arrays.toString(shape), Arrays.toString(fbShape));
                    throw new IOException("Shape mismatch in metadata for variable '" + name + "'.");
                } else if (shape == null && fbShape != null) {
                    shape = fbShape;
                    log.debug("Updated shape for variable '{}' based on FlatVariable metadata: {}", name, Arrays.toString(shape));
                    var.setShape(shape);
                }

                // Get Order - NOW MANDATORY from FlatArray within FlatVariable
                FlatArray fa = fv.ndarray();
                if (fa == null) {
                    log.error("FATAL: Missing FlatArray metadata within FlatVariable for variable '{}'. Cannot determine order.", name);
                    throw new IOException("Missing FlatArray metadata within FlatVariable for variable '" + name + "'. Cannot determine order.");
                }
                order = fa.byteOrder() == 1 ? 'f' : 'c';
                log.trace("Determined order '{}' for variable '{}' from FlatArray metadata.", order, name);

            } else {
                log.error("FATAL: FlatVariable metadata missing entirely in this shard for variable '{}' which is listed in the manifest. Save process might be flawed.", name);
                throw new IOException("Missing FlatVariable metadata for manifested variable '" + name + "'.");
            }
            // --- End Metadata Determination ---

            // --- Validate Metadata and Manifest Length ---
            if (shape == null) { /* ... error ... */ }
            if (dtype == null || dtype == DataType.UNKNOWN) { /* ... error ... */ }
            // (Length validation logic remains the same...)
            if (lengthBytes <= 0) { /* ... */ }
            int elementSize = dtype.width();
            if (elementSize <= 0) { /* ... */ }
            long expectedElements = ArrayUtil.prodLong(shape);
            if (expectedElements < 0) { /* ... */ }
            else {
                long expectedLengthBytes = expectedElements * elementSize;
                if (lengthBytes != expectedLengthBytes) {
                    log.error("FATAL: Manifest length mismatch for var '{}'. Manifest: {}, Calculated from Shape*Dtype: {}. (Shape: {}, DType: {}, Order: {}). Skipping.",
                            name, lengthBytes, expectedLengthBytes, Arrays.toString(shape), dtype, order);
                    throw new IOException("Manifest length mismatch for variable '" + name + "'.");
                }
            }
            if (dtype == DataType.COMPRESSED || dtype == DataType.UTF8) { /* ... */ }


            log.debug("Preparing to load {} bytes for variable '{}' (dtype={}, shape={}, order={}) from file offset {}",
                    lengthBytes, name, dtype, Arrays.toString(shape), order, offset);

            // --- Create Target Array ---
            INDArray resultArr = null;
            DataBuffer targetBuffer = null;
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                // (Creation logic as before, using determined order)
                if (lengthBytes == 0 && expectedElements == 0) {
                    resultArr = Nd4j.create(dtype, shape, Nd4j.getStrides(shape, order), order);
                } else if (lengthBytes > 0) {
                    resultArr = Nd4j.createUninitialized(dtype, shape, order);
                }

                if (resultArr == null) throw new IOException("Nd4j.createUninitialized/create returned null for " + name);
                if (resultArr.ordering() != order && !resultArr.isEmpty()) {
                    log.error("CRITICAL: Created array for '{}' but order mismatch! Expected '{}', got '{}'.", name, order, resultArr.ordering());
                }
                targetBuffer = resultArr.data();
                if (targetBuffer == null && lengthBytes > 0) throw new IOException("Target DataBuffer is null for " + name);
                log.trace("Successfully created target INDArray for '{}'. IsEmpty={}, Length={}, Shape={}, Order={}", name, resultArr.isEmpty(), resultArr.length(), Arrays.toString(resultArr.shape()), resultArr.ordering());
            } catch (Exception e) {
                log.error("FATAL: Failed to create INDArray for {}", name, e);
                throw new IOException("Failed to create INDArray for " + name, e);
            }

            // --- Read Data From Channel into Array Buffer ---
            if (lengthBytes > 0) {
                long arrayOffsetBytes = resultArr.offset() * targetBuffer.getElementSize(); // Usually 0 for new arrays

                try {
                    long currentFilePos = channel.position();
                    log.trace("Reading data for '{}': Current channel pos before seek: {}", name, currentFilePos);
                    channel.position(offset); // Position channel to where blob starts
                    long posAfterSeek = channel.position();
                    log.debug("Reading data for '{}': Channel position after seek to {}: {}", name, offset, posAfterSeek);
                    if (posAfterSeek != offset) {
                        log.error("FATAL: Failed to seek channel to correct offset for '{}'. Expected {}, Got {}.", name, offset, posAfterSeek);
                        throw new IOException("FileChannel seek failed for variable '" + name + "'.");
                    }

                    ByteBuffer targetNio = targetBuffer.asNio(); // Still needed for reason check below

                    // *** MODIFICATION START: Force fallback path ***
                    boolean canUseDirectNio = false; // FORCE FALSE to always use fallback
                    log.warn("FORCING FALLBACK PATH for '{}' (canUseDirectNio set to false)", name);
                    // *** MODIFICATION END ***

                    if (canUseDirectNio) {
                        // --- Direct NIO Read Path ---
                        // THIS CODE WILL NOT BE EXECUTED DUE TO FORCED FALSE ABOVE
                        log.debug("Attempting Direct NIO read for '{}' into target buffer (Capacity: {}, IsDirect: {})", name, targetNio.capacity(), targetNio.isDirect());
                        // ... (Rest of direct NIO read logic) ...

                    } else {
                        // --- Fallback: Read via temporary byte[] chunks + Pointer.memcpy ---
                        log.warn("Using chunked byte[] read fallback for '{}'. (Forced Path / Reason: {} large ({} bytes) or direct buffer unavailable ({}))",
                                name, (lengthBytes > Integer.MAX_VALUE || arrayOffsetBytes > Integer.MAX_VALUE || (arrayOffsetBytes + lengthBytes) > Integer.MAX_VALUE) ? "Array > 2GB / offset issue" : "NIO buffer", lengthBytes, (targetNio==null));

                        if (tempChunk == null)
                            tempChunk = new byte[RAW_IO_CHUNK_SIZE_BYTES];
                        long bytesReadCount = 0;
                        long targetBufferWriteOffsetBytes = arrayOffsetBytes; // Start writing at the INDArray's buffer offset
                        Pointer targetPointer = targetBuffer.pointer(); // Get native pointer
                        if (targetPointer == null || targetPointer.isNull()) {
                            log.error("FATAL: Cannot get native pointer for target DataBuffer of variable '{}'. Cannot use fallback copy.", name);
                            throw new IOException("Cannot get native pointer for fallback copy for variable '" + name + "'.");
                        }
                        log.trace("Target buffer pointer obtained for fallback write for '{}'. Address: {}", name, targetPointer.address());

                        long startReadTime = System.currentTimeMillis();
                        int chunkNum = 0;
                        long endReadTime = startReadTime; // Initialize for duration calculation
                        while (bytesReadCount < lengthBytes) {
                            chunkNum++;
                            int toRead = (int) Math.min(tempChunk.length, lengthBytes - bytesReadCount);
                            log.trace("Fallback chunk {}: Reading {} bytes from channel into temp buffer...", chunkNum, toRead);
                            ByteBuffer tempNioWrapper = ByteBuffer.wrap(tempChunk, 0, toRead);
                            int actuallyRead = 0;
                            long loopStartTime = System.currentTimeMillis();
                            int zeroByteReads = 0;
                            final int MAX_ZERO_BYTE_READS = 1000;

                            // Loop to ensure the chunk buffer is filled
                            while (tempNioWrapper.hasRemaining()) {
                                int chunkReadBytes = channel.read(tempNioWrapper);
                                if (chunkReadBytes == -1) {
                                    log.error("FATAL: EOFException encountered during chunked read for '{}'. Expected {} bytes, read {} bytes total. Failed in chunk {}.", name, lengthBytes, bytesReadCount + actuallyRead, chunkNum);
                                    throw new EOFException("EOF encountered during chunked read for variable '" + name + "'.");
                                }
                                if (chunkReadBytes == 0) {
                                    zeroByteReads++;
                                    if (zeroByteReads > MAX_ZERO_BYTE_READS) {
                                        log.error("FATAL: Fallback chunk read stalled ({} consecutive 0-byte reads) for variable '{}'. Aborting.", zeroByteReads, name);
                                        throw new IOException("Fallback chunk read stalled for variable '" + name + "'.");
                                    }
                                    log.trace("Fallback chunk read returned 0 bytes, sleeping briefly (Attempt {})...", zeroByteReads);
                                    try { Thread.sleep(1); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); throw new IOException("Read interrupted", ie); }
                                    continue;
                                }
                                zeroByteReads = 0; // Reset
                                actuallyRead += chunkReadBytes;
                                if (System.currentTimeMillis() - loopStartTime > 120000) { // 2 min timeout per chunk read
                                    log.error("FATAL: Read timeout (>2min) while reading chunk {} for '{}'. Expected {} bytes in chunk, read {}.", chunkNum, name, toRead, actuallyRead);
                                    throw new IOException("Read timeout during chunk read for variable '" + name + "'.");
                                }
                            }
                            log.trace("Fallback chunk {}: Successfully read {} bytes from channel.", chunkNum, actuallyRead);
                            if (actuallyRead == 0 && bytesReadCount < lengthBytes) {
                                log.warn("Chunk read returned 0 bytes unexpectedly for '{}' (chunk {}). Retrying.", name, chunkNum);
                                continue; // Should ideally not happen with inner loop, but safety
                            }

                            int checkLen = Math.min(16, actuallyRead);
                            log.debug("First {} bytes read into tempChunk for chunk {} of '{}': [{}]", checkLen, chunkNum, name, bytesToHex(tempChunk, 0, checkLen)); // Log chunk bytes

                            // Copy using memcpy
                            try (Pointer sourcePointer = new BytePointer(tempChunk)) {
                                // Position the target pointer correctly based on INDArray offset and bytes already written
                                BytePointer targetWritePtr = new BytePointer(targetPointer).position(targetBufferWriteOffsetBytes);
                                log.trace("Fallback chunk {}: Copying {} bytes using memcpy from temp buffer to target pointer @ {}", chunkNum, actuallyRead, targetWritePtr.address());
                                Pointer.memcpy(targetWritePtr, sourcePointer, actuallyRead);
                            } catch (Exception memcpyEx) {
                                log.error("FATAL: Exception during Pointer.memcpy for chunk {} of variable '{}'", chunkNum, name, memcpyEx);
                                throw new IOException("Failed memcpy during fallback read for variable '" + name + "'", memcpyEx);
                            }
                            bytesReadCount += actuallyRead;
                            targetBufferWriteOffsetBytes += actuallyRead; // Advance target write offset
                            log.trace("Fallback chunk {}: Completed copy. Total bytes read so far: {}", chunkNum, bytesReadCount);

                            // Optional: Timeout check for overall array
                            if (System.currentTimeMillis() - startReadTime > 600000) { // 10 min overall timeout
                                log.error("FATAL: Read timeout (>10min) during chunked read fallback for '{}'. Expected {} bytes, read {} bytes.", name, lengthBytes, bytesReadCount);
                                throw new IOException("Overall read timeout during fallback for variable '" + name + "'.");
                            }
                        } // End while (bytesReadCount < lengthBytes)
                        endReadTime = System.currentTimeMillis();
                        if (bytesReadCount != lengthBytes) {
                            log.error("FATAL: Chunked read fallback incomplete for '{}'. Expected {}, Read {}.", name, lengthBytes, bytesReadCount);
                            throw new IOException("Chunked read fallback incomplete for variable '" + name + "'.");
                        }
                        log.info("Chunked fallback read successful for '{}' ({} bytes in {} ms).", name, bytesReadCount, endReadTime - startReadTime);
                        // --- End Fallback ---
                    }
                } catch (IOException e) {
                    log.error("FATAL: IOException during raw data read for variable '{}' at offset {}", name, offset, e);
                    throw e;
                } catch (Exception e) {
                    log.error("FATAL: Unexpected error during raw data read for variable '{}'", name, e);
                    throw new IOException("Failed loading raw data for variable '" + name + "'", e);
                }

                // *** Log array stats AFTER reading data ***
                try {
                    log.debug("Data load complete for '{}'. Verifying final array content (Min/Max/Sum/FirstElem):", name);
                    // Ensure array isn't closed or invalid before accessing stats
                    if (resultArr.isView() || !resultArr.closeable()) { // Simple check, might need refinement
                        double firstVal = resultArr.isScalar() ? resultArr.getDouble(0) : resultArr.getDouble(0);
                        log.debug("  -> Min: {}, Max: {}, Sum: {}, First Element [0]: {}",
                                resultArr.minNumber(), resultArr.maxNumber(), resultArr.sumNumber(), firstVal);
                        if (resultArr.sumNumber().doubleValue() == 0.0 && lengthBytes > 0 && expectedElements > 0) {
                            log.warn("WARNING: Array '{}' sum is 0.0 after loading {} bytes. Data might be incorrect or zero.", name, lengthBytes);
                        }
                    } else {
                        log.warn("Skipping post-load stats verification for '{}' as array appears to be closed or invalid.", name);
                    }
                } catch (Exception statsEx) {
                    log.warn("Could not retrieve stats for loaded array '{}'", name, statsEx);
                }

            } else {
                log.debug("Skipping data reading for empty array '{}'", name);
            }


            // --- Associate Array with SameDiff Instance & Verify ---
            log.debug("Associating loaded array with variable '{}' in target SameDiff instance.", name);
            try {
                targetSD.setArrayForVariable(name, resultArr);
                log.trace("Called targetSD.setArrayForVariable('{}', ...)", name);
                // (Verification logic as before)
                INDArray checkArr = targetSD.getArrForVarName(name);
                if(checkArr == null) {
                    log.error("!!!!!! VERIFICATION FAILED (Post-Association) !!!!!!!");
                    log.error("CRITICAL: Array is NULL immediately after setting (via getArrForVarName) for variable '{}'!", name);
                    throw new IllegalStateException("Verification failed: Array is null immediately after setting for variable '" + name + "'.");
                } else {
                    log.debug("Verification PASSED (Post-Association): Array is non-NULL via getArrForVarName for '{}'.", name);
                }
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
            @NonNull Set<String> largeArrayNamesToExcludeData, // Still used to decide *if* data goes inline
            @NonNull Set<String> smallArrayNamesToIncludeData) throws IOException { // Still used to decide *if* data goes inline


        ExecutorConfiguration configuration = ExecutorConfiguration.builder().outputMode(OutputMode.VARIABLE_SPACE)
                .executionMode(org.nd4j.autodiff.execution.conf.ExecutionMode.SEQUENTIAL)
                .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
                .build();
        Map<String, String> mergedMetadata = enrichMetadata(metadata); // Assume helper exists
        mergedMetadata.put(META_ND4J_FORMAT_TYPE, FORMAT_TYPE_APPENDED);
        mergedMetadata.put(META_ND4J_FORMAT_VERSION, String.valueOf(FILE_VERSION));


        // *** CORRECTED: Initialize FlatBufferBuilder ***
        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(2 * 1024 * 1024); // Initial size 2MB


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
        val variableListForOps = new ArrayList<>(sameDiff.variables()); // Stable list for iteration
        val reverseMap = new LinkedHashMap<String, Integer>(); // VarName -> NodeID
        val idxForOps = new IdentityHashMap<DifferentialFunction, Integer>(); // Op -> NodeID
        val idCounter = new AtomicInteger(0); // For assigning node IDs


        log.debug("Starting variable iteration for metadata FB ({} vars in this instance)", sameDiff.variables().size());
        for (SDVariable variable : variableListForOps) { // Iterate the stable list
            String varName = variable.name();
            if (varName == null || variable.getVariableType() == VariableType.SEQUENCE)
                continue; // Skip sequence types
            Variable vMeta = sameDiff.getVariables().get(varName);
            if (vMeta == null) {
                log.warn("Internal metadata missing for variable '{}'. Skipping variable serialization.", varName);
                continue;
            }


            // --- Assign Node ID ---
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
                if (outputNum < 0) outputNum = 0; // Default if not found (shouldn't happen?)
            } else {
                varIdx = idCounter.incrementAndGet(); // Independent variable (placeholder, constant, var)
                outputNum = 0;
            }
            reverseMap.put(varName, varIdx); // Map name to node ID
            int idOffset = IntPair.createIntPair(bufferBuilder, varIdx, outputNum);
            // --- End Assign Node ID ---


            // --- Basic Variable Info ---
            int nameOffset = bufferBuilder.createString(varName);
            byte varTypeByte = FlatBuffersMapper.toVarType(variable.getVariableType());
            DataType dtype = variable.dataType();
            INDArray arr = variable.getArr(); // Get array reference (might be null)
            if (dtype == DataType.UNKNOWN && arr != null)
                dtype = arr.dataType();
            if (dtype == DataType.UNKNOWN)
                dtype = DataType.FLOAT; // Default if still unknown
            byte dtypeByte = FlatBuffersMapper.getDataTypeAsByte(dtype);
            long[] shape = variable.getShape();
            int shapeOffset = 0;
            if (shape != null) {
                shapeOffset = FlatVariable.createShapeVector(bufferBuilder, shape);
            } else if (variable.getVariableType() != VariableType.PLACEHOLDER){
                log.warn("Variable '{}' (Type {}) has null shape, creating empty shape vector offset.", varName, variable.getVariableType());
                shapeOffset = FlatVariable.createShapeVector(bufferBuilder, new long[0]); // Empty vector for null shape
            }
            // --- End Basic Variable Info ---


            // *** START Order Handling Modification ***
            int arrayOffset; // Offset to the FlatArray structure
            int bufferOffset = 0; // Offset to the *inline* byte buffer within FlatArray (if applicable)
            char orderChar = (arr != null) ? arr.ordering() : 'c'; // Get order or default to 'c'
            byte orderByte = (byte) (orderChar == 'c' ? 0 : 1);


            // Decide if data should be serialized inline AND attempt it
            boolean serializeDataInline = smallArrayNamesToIncludeData.contains(varName) && arr != null && !arr.isEmpty() && !arr.isScalar();


            if (serializeDataInline) {
                // Attempt to serialize the small array data inline
                try {
                    // This call *only* creates the byte vector for inline data if successful
                    bufferOffset = serializeSmallNdArrayToFlatBuffer(arr, bufferBuilder);
                    if (bufferOffset != 0) {
                        log.trace("Serialized small array inline for '{}', bufferOffset={}", varName, bufferOffset);
                    } else {
                        log.warn("serializeSmallNdArrayToFlatBuffer returned 0 for supposedly small inline array '{}'. No inline data will be stored.", varName);
                    }
                } catch (Exception e) {
                    log.warn("Error serializing small array inline for '{}'. Buffer offset will be 0.", varName, e);
                    bufferOffset = 0; // Ensure bufferOffset is 0 on error
                }
            }
            // ELSE: for large/appended arrays, or empty/scalar arrays, bufferOffset remains 0


            // ALWAYS create the FlatArray structure to store metadata (shape, dtype, order)
            // bufferOffset will be non-zero only if data was successfully serialized inline above
            try {
                // isExternal flag doesn't seem essential here, using false.
                arrayOffset = FlatArray.createFlatArray(bufferBuilder, shapeOffset, bufferOffset, dtypeByte, orderByte, 0, 0, 0, false);
                log.trace("Created FlatArray structure for '{}'. Offset={}, BufferOffset={}, OrderByte={}", varName, arrayOffset, bufferOffset, orderByte);
            } catch (Exception e) {
                log.error("Failed to create FlatArray structure for variable '{}'. arrayOffset will be 0.", varName, e);
                arrayOffset = 0; // Ensure arrayOffset is 0 on error. This might cause issues on loading if FlatVariable expects non-zero.
            }
            // *** END Order Handling Modification ***


            // --- Control Dependencies ---
            int controlDepsOffset = 0, controlDepsForOpOffset = 0, controlDepsForVarOffset = 0;
            // Safely get internal metadata
            Variable internalVarMeta = null;
            // SDVariable sdVar = sd.getVariable(varName); // Cannot get from target sd, get from source
            SDVariable sdVar = sameDiff.getVariable(varName);
            if(sdVar != null) internalVarMeta = sameDiff.getVariables().get(varName);


            if(internalVarMeta != null) {
                int[] cds = FlatBuffersMapper.mapOrNull(internalVarMeta.getControlDeps(), bufferBuilder);
                if (cds != null) controlDepsOffset = FlatVariable.createControlDepsVector(bufferBuilder, cds);
                int[] cdsForOp = FlatBuffersMapper.mapOrNull(internalVarMeta.getControlDepsForOp(), bufferBuilder);
                if (cdsForOp != null)
                    controlDepsForOpOffset = FlatVariable.createControlDepForOpVector(bufferBuilder, cdsForOp);
                int[] cdsForVar = FlatBuffersMapper.mapOrNull(internalVarMeta.getControlDepsForVar(), bufferBuilder);
                if (cdsForVar != null)
                    controlDepsForVarOffset = FlatVariable.createControlDepsForVarVector(bufferBuilder, cdsForVar);
            } else {
                log.warn("Could not retrieve internal Variable metadata for '{}' when processing control dependencies.", varName);
            }
            // --- End Control Dependencies ---




            // Create the FlatVariable using the (potentially data-less but metadata-rich) arrayOffset
            if (arrayOffset == 0 && !variable.isPlaceHolder() && variable.getArr() != null) {
                log.warn("Creating FlatVariable for '{}' with arrayOffset = 0. Ensure loading logic handles this.", varName);
            }
            flatVariables.add(FlatVariable.createFlatVariable(
                    bufferBuilder,
                    idOffset,
                    nameOffset,
                    dtypeByte,
                    shapeOffset,
                    arrayOffset, // Use the offset to the FlatArray structure
                    -1, // constantValue
                    varTypeByte,
                    controlDepsOffset,
                    controlDepsForOpOffset,
                    controlDepsForVarOffset));
        } // End variable loop


        log.debug("serializeMetadataFlatBuffer: Finished variable iteration. flatVariables.size() = {} (Expected ~{})", flatVariables.size(), variableListForOps.size()); // Compare to iterated list size
        if (flatVariables.isEmpty() && !variableListForOps.stream().allMatch(v -> v.getVariableType() == VariableType.SEQUENCE || v.name() == null)) {
            log.warn("Variable processing loop resulted in empty flatVariables list, but original SameDiff had non-sequence variables!");
        }
        int variablesVectorOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));




        // --- Ops Vec ---
        val flatNodes = new ArrayList<Integer>();
        log.debug("Starting op iteration for metadata FB ({} ops in this instance)", sameDiff.getOps().size());
        // Initialize maps used by asFlatNode - values might be needed across calls if graph has cycles/shared nodes?
        Map<String, Integer> forwardMap = new HashMap<>();
        Map<String, Integer> framesMap = new HashMap<>();


        for (SameDiffOp op : sameDiff.getOps().values()) {
            DifferentialFunction df = op.getOp();
            if (df == null) {
                log.warn("Skipping op '{}' with null function.", op.getName());
                continue;
            }
            Integer fnId = idxForOps.get(df); // Get ID assigned via output variables during variable loop
            if (fnId == null) {
                // Re-attempt ID assignment or lookup if needed
                boolean foundOutputId = false;
                for(String outName : op.getOutputsOfOp()) {
                    if(sameDiff.hasVariable(outName) && reverseMap.containsKey(outName)) {
                        fnId = reverseMap.get(outName);
                        idxForOps.put(df, fnId); // Store for potential reuse
                        foundOutputId = true;
                        log.trace("Assigning Op '{}' ID {} based on its output '{}'", op.getName(), fnId, outName);
                        break;
                    }
                }
                if(!foundOutputId) {
                    fnId = idCounter.incrementAndGet(); // Assign new ID if truly independent
                    idxForOps.put(df, fnId);
                    log.warn("Op '{}' ({}) was not assigned an ID via its outputs (Outputs: {}). Assigning new ID {}. Check graph structure/linking.", op.getName(), df.opName(), op.getOutputsOfOp(), fnId);
                }
            }
            try {
                // Pass necessary context maps to asFlatNode
                flatNodes.add(FlatBuffersMapper.asFlatNode(sameDiff, df, bufferBuilder, variableListForOps, reverseMap, forwardMap, framesMap, idCounter, fnId));
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
                String outputName = sameDiff.outputs().get(i);
                if(outputName != null) { // Add null check
                    outputsOffsets[i] = bufferBuilder.createString(outputName);
                } else {
                    log.warn("Null output name found at index {} in SameDiff outputs list.", i);
                    outputsOffsets[i] = bufferBuilder.createString(""); // Use empty string offset
                }
            }
            outputsVectorOffset = FlatGraph.createOutputsVector(bufferBuilder, outputsOffsets);
        } else {
            outputsVectorOffset = FlatGraph.createOutputsVector(bufferBuilder, new int[0]);
        }


        int placeholdersVectorOffset = createPlaceholdersVector(sameDiff, bufferBuilder);
        int lossVariablesVectorOffset = createLossVariablesVector(sameDiff, bufferBuilder);
        int trainingConfigStringOffset = createTrainingConfigOffset(sameDiff, bufferBuilder);
        int updaterStateVectorOffset = createUpdaterStateOffset(sameDiff, bufferBuilder, saveUpdaterState);
        int configurationTableOffset = configuration.getFlatConfiguration(bufferBuilder);




        // --- Finalize FlatBuffer ---
        log.debug("serializeMetadataFlatBuffer: Finalizing FlatBuffer. VarVecOffset={}, NodeVecOffset={}, PlaceholderVecOffset={}, LossVecOffset={}, UpdaterStateVecOffset={}",
                variablesVectorOffset, nodesVectorOffset, placeholdersVectorOffset, lossVariablesVectorOffset, updaterStateVectorOffset);
        int fgOffset = FlatGraph.createFlatGraph(bufferBuilder,
                0, // Graph ID
                variablesVectorOffset, nodesVectorOffset, outputsVectorOffset, configurationTableOffset,
                placeholdersVectorOffset, lossVariablesVectorOffset, trainingConfigStringOffset, updaterStateVectorOffset,
                metadataKeysOffset, metadataValuesOffset);
        bufferBuilder.finish(fgOffset); // Use the final offset


        ByteBuffer resultBuffer = bufferBuilder.dataBuffer();
        log.debug("serializeMetadataFlatBuffer: Finished. Result buffer size: {}", resultBuffer.remaining());
        if (resultBuffer.remaining() == 0 && (sameDiff.variables().size() > 0 || sameDiff.getOps().size() > 0)) {
            boolean onlySequence = sameDiff.variables().stream().allMatch(v -> v.getVariableType() == VariableType.SEQUENCE || v.name() == null);
            if(!onlySequence)
                log.error("CRITICAL: serializeMetadataFlatBuffer produced an empty buffer for a non-empty SameDiff instance!");
        }
        return resultBuffer;
    }

    /**
     * Serializes a small INDArray (non-scalar, non-empty) to a FlatBuffer buffer vector.
     * Uses Native Endian byte order. Includes enhanced byte verification.
     * REVERTED: Use createBufferVector with byte[] again.
     */
    public static int serializeSmallNdArrayToFlatBuffer(@NonNull INDArray arr, @NonNull FlatBufferBuilder builder) throws IOException {
        // Try to get a somewhat identifiable name/string for logging
        String varNameForLog = "InlineArray"; // Default
        try {
            // Generating a unique ID or using shape string for logging
            java.util.UUID uuid = java.util.UUID.randomUUID(); // Example: Use UUID
            varNameForLog = arr.shapeInfoToString() + "_" + uuid.toString().substring(0, 8); // Example shape + partial UUID
        } catch (Exception e) {
            try { varNameForLog = arr.shapeInfoToString(); } catch (Exception e2) { /* ignore */ }
        }


        log.info("SERIALIZE_INLINE: Attempting for Var='{}', Shape={}, DType={}", varNameForLog, Arrays.toString(arr.shape()), arr.dataType());


        // Skip arrays we can't handle safely
        if (arr.dataType() == DataType.UTF8 || arr.dataType() == DataType.COMPRESSED || arr.dataType() == DataType.UNKNOWN) {
            log.warn("SERIALIZE_INLINE [{}]: Cannot serialize array of type {} inline. Skipping.", varNameForLog, arr.dataType());
            return 0;
        }


        long[] shape = arr.shape();
        int rank = arr.rank();
        // Scalars (rank 0 or length 1 with shape [1]) are often handled differently or not serialized inline.
        // Let's clarify the scalar condition based on typical INDArray usage.
        boolean isScalar = arr.isScalar(); // Use INDArray's method


        // Check dimension validity (ensure dimensions fit in int for FlatBuffer vectors if needed, though data is ubyte vector)
        if (!isScalar && shape != null) {
            for (long d : shape) {
                // FlatBuffers vectors have size limits (usually 2GB or Integer.MAX_VALUE elements).
                // While the *data* might exceed this for large arrays (handled by append),
                // the *shape dimensions* themselves should usually be reasonable. Add a practical check.
                if (d < 0 || d > Integer.MAX_VALUE) { // Check individual dimensions
                    log.error("SERIALIZE_INLINE [{}]: Invalid shape dimension: {} exceeds Integer.MAX_VALUE. Skipping serialization.", varNameForLog, Arrays.toString(shape));
                    return 0;
                }
            }
        } else if (shape == null && !isScalar && !arr.isEmpty()) {
            // Shape should not be null for non-scalar, non-empty arrays
            log.error("SERIALIZE_INLINE [{}]: Invalid null shape for non-scalar, non-empty inline array. Skipping serialization.", varNameForLog);
            return 0;
        }


        // Skip explicit scalars for inline data buffer? (Metadata like shape/dtype/order is still saved)
        // FlatBuffers might handle scalars differently (e.g., direct value fields, not vectors).
        // For simplicity here, we might skip creating a data buffer for scalars, but the FlatArray metadata will exist.
        if (isScalar) {
            log.debug("SERIALIZE_INLINE [{}]: Skipping inline data buffer creation for scalar array.", varNameForLog);
            // Return 0, indicating no data buffer offset. The FlatArray structure will still be created
            // in serializeMetadataFlatBuffer to hold shape/dtype/order.
            return 0;
        }


        // Handle empty arrays (shape only, no data buffer)
        if (arr.isEmpty()) {
            log.debug("SERIALIZE_INLINE [{}]: Array is empty. Returning 0 for buffer offset.", varNameForLog);
            // The FlatArray structure holding shape/dtype/order will be created in serializeMetadataFlatBuffer.
            return 0; // No data buffer offset needed
        }


        // --- Standard non-scalar, non-empty array handling ---
        int bufferOffset = 0; // FB offset for the data vector (default to 0)
        try {
            DataBuffer buffer = arr.data();
            ByteBuffer nioBuffer = buffer.asNio(); // Get NIO view
            long lengthBytes = arr.length() * buffer.getElementSize();


            if (nioBuffer != null) {
                nioBuffer.order(ByteOrder.nativeOrder()); // Ensure NATIVE order
                long arrOffsetBytes = arr.offset() * buffer.getElementSize(); // Use INDArray offset


                // Check if view is usable and within limits for byte[] extraction
                if (arrOffsetBytes >= 0 && arrOffsetBytes <= Integer.MAX_VALUE &&
                        lengthBytes >= 0 && lengthBytes <= Integer.MAX_VALUE && // Check lengthBytes against Integer.MAX_VALUE
                        (arrOffsetBytes + lengthBytes) >= 0 && (arrOffsetBytes + lengthBytes) <= Integer.MAX_VALUE) // Check combined offset+length
                {
                    // Create a new byte array to hold the native-ordered bytes
                    byte[] nativeBytes = new byte[(int)lengthBytes]; // Safe cast due to checks
                    // Position the NIO buffer view correctly
                    // Use duplicate to avoid modifying the original buffer's state if shared
                    ByteBuffer view = nioBuffer.duplicate().order(ByteOrder.nativeOrder());
                    view.position((int)arrOffsetBytes);
                    view.limit((int)(arrOffsetBytes + lengthBytes));


                    // Read the native-ordered bytes into the temporary array
                    try {
                        log.trace("SERIALIZE_INLINE [{}]: Reading {} bytes from NIO buffer slice into nativeBytes array.", varNameForLog, lengthBytes);
                        view.get(nativeBytes);
                    } catch (Exception e) {
                        log.error("SERIALIZE_INLINE [{}]: Exception during view.get(nativeBytes)! {} bytes requested.", varNameForLog, lengthBytes, e);
                        return 0; // Fail if extraction fails
                    }


                    // *** Optional: ENHANCED BYTE CHECK ***
                    // (Byte check code omitted for brevity, but could be re-inserted here if needed)
                    // ...


                    // Insert into FlatBuffer using the byte array
                    if (nativeBytes.length > 0) {
                        log.trace("SERIALIZE_INLINE [{}]: Creating FlatBuffer byte vector from {} native bytes.", varNameForLog, nativeBytes.length);
                        // Use the version accepting byte[]
                        bufferOffset = FlatArray.createBufferVector(builder, nativeBytes);
                        log.trace("SERIALIZE_INLINE [{}]: Prepared FlatBuffer vector from native bytes. Offset={}", varNameForLog, bufferOffset);
                    } else {
                        // This case should ideally only happen for empty arrays, handled earlier
                        log.warn("SERIALIZE_INLINE [{}]: Extracted zero bytes for non-empty inline array? Buffer offset will be 0.", varNameForLog);
                        bufferOffset = 0;
                    }


                } else {
                    log.error("SERIALIZE_INLINE [{}]: Cannot serialize inline array: Offset/length ({}/{}) or total length ({}) exceeds limits for NIO buffer view or byte array.",
                            varNameForLog, arrOffsetBytes, lengthBytes, lengthBytes);
                    return 0; // Indicate failure
                }
            } else {
                log.error("SERIALIZE_INLINE [{}]: Cannot serialize inline array: Direct NIO buffer not available (asNio=null).", varNameForLog);
                return 0; // Indicate failure
            }


            // --- Create FlatArray Structure (Moved to serializeMetadataFlatBuffer) ---
            // The FlatArray structure itself (holding shape, dtype, order, and bufferOffset)
            // is now created in serializeMetadataFlatBuffer to ensure it's always present.


            // Log success/failure based on whether data was actually prepared
            if (bufferOffset > 0) {
                log.info("SERIALIZE_INLINE: SUCCESS preparing inline data buffer for Var='{}'. Returning bufferOffset {}.", varNameForLog, bufferOffset);
            } else if (arr.length() > 0) {
                // If array wasn't empty but we failed to get a buffer offset
                log.error("SERIALIZE_INLINE: FAILURE preparing inline data buffer for Var='{}'. Failed to create buffer vector? Returning 0.", varNameForLog);
                return 0; // Return 0 to indicate failure (no data buffer offset)
            } else {
                // Array was empty or scalar, buffer offset is legitimately 0
                log.info("SERIALIZE_INLINE: SUCCESS (No Data Buffer) for Var='{}'. BufferOffset is 0.", varNameForLog);
            }
            // Return the offset to the data buffer vector *within* the FlatBuffer file
            // This offset will be stored in the FlatArray structure created later.
            return bufferOffset;


        } catch (Exception e) {
            log.error("SERIALIZE_INLINE [{}]: Unhandled error preparing inline array data: {}", varNameForLog, e.getMessage(), e);
            return 0; // Return 0 on any error
        }
    } // end serializeSmallNdArrayToFlatBuffer


    /**
     * Deserializes an inline INDArray from FlatBuffer data. Expects Native Endian bytes.
     * MODIFIED: Reads bytes manually using fa.buffer(j) instead of fa.bufferAsByteBuffer()
     * MODIFIED AGAIN: Ignores fa.bufferLength() for validation and reads expectedBytes directly,
     * checking bounds via fa.buffer(last_index).
     *
     * @param fa      FlatBuffer FlatArray object containing inline data.
     * @param varName The name of the variable being deserialized (for logging).
     * @return Deserialized INDArray or null on failure.
     */
    private static INDArray deserializeSmallNdArrayFromInlineBuffer(FlatArray fa, String varName) throws IOException {
        final boolean isTargetVar = "layer_0_b".equals(varName); // Flag for specific logging

        if (fa == null) {
            log.trace("LOAD_INLINE [{}]: FlatArray object is null. Returning null.", varName);
            return null;
        }

        // Handle empty array case first (based on shape, no data buffer expected/needed)
        if (fa.shapeLength() > 0 && fa.bufferLength() == 0 && fa.bufferChunksLength() == 0 && !fa.isExternal()) {
            try {
                long[] shape = new long[fa.shapeLength()];
                for (int i = 0; i < shape.length; i++) { shape[i] = fa.shape(i); }
                byte dtypeByte = fa.dtype();
                DataType dataType = FlatBuffersMapper.getDataTypeFromByte(dtypeByte);
                if (dataType == null || dataType == DataType.UNKNOWN) { log.debug("LOAD_INLINE [{}]: Empty FlatArray has unrecognized dtype ({}). Defaulting to FLOAT.", varName, dtypeByte); dataType = DataType.FLOAT; }
                char order = fa.byteOrder() == 0 ? 'c' : 'f';
                long numElements = ArrayUtil.prod(shape);
                if (numElements != 0) {
                    log.warn("LOAD_INLINE [{}]: Shape {} implies {} elements, but FlatBuffer data length is 0. Creating empty array.", varName, Arrays.toString(shape), numElements);
                } else {
                    log.trace("LOAD_INLINE [{}]: Creating empty array from shape-only FlatArray. Shape {}, Dtype {}, Order {}", varName, Arrays.toString(shape), dataType, order);
                }
                // Use try-with-resources for memory management if creating within a specific scope matters, otherwise direct creation is fine here.
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    return Nd4j.create(dataType, shape, order); // Create based on shape
                }
            } catch (Exception e) { log.error("LOAD_INLINE [{}]: Failed to create empty array from shape-only FlatArray: {}", varName, e.getMessage(), e); return null; }
        }


        // Handle inline buffer with data
        // We IGNORE fa.bufferLength() for validation here due to suspected issues.
        // We rely on the vector actually containing the correct number of bytes accessed via fa.buffer(j).
        if (fa.bufferLength() >= 0 && fa.bufferChunksLength() == 0 && !fa.isExternal()) { // Check >= 0 just in case, main check is below
            String shapeForLog = "N/A";
            try {
                // Step 1: Handle scalar array (rank 0) - Assuming not saved inline based on serialize logic
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
                if (dataType == null || dataType == DataType.UNKNOWN) {
                    log.warn("LOAD_INLINE [{}]: Unrecognized dtype ({}) in inline FlatArray for shape {}. Defaulting to FLOAT.", varName, dtypeByte, shapeForLog);
                    dataType = DataType.FLOAT;
                }
                char order = fa.byteOrder() == 0 ? 'c' : 'f';

                // Calculate expected size FROM SHAPE/DTYPE
                int elementSize = dataType.width();
                if(elementSize <= 0) {
                    log.error("LOAD_INLINE [{}]: Invalid element size {} for dtype {}. Cannot load.", varName, elementSize, dataType);
                    return null;
                }
                long totalElements = ArrayUtil.prod(shape);
                if (totalElements < 0) {
                    log.error("LOAD_INLINE [{}]: Shape {} results in negative total elements. Cannot load.", varName, shapeForLog);
                    return null;
                }
                long expectedBytes = -1;
                try {
                    expectedBytes = Math.multiplyExact(totalElements, elementSize);
                } catch (ArithmeticException e) {
                    log.error("LOAD_INLINE [{}]: Overflow calculating expected bytes ({} elements * {} bytes/element). Cannot load.", varName, totalElements, elementSize);
                    return null;
                }

                // Get buffer length reported by FlatBuffers metadata - FOR LOGGING ONLY
                int fbBufferLength = fa.bufferLength();
                log.debug("LOAD_INLINE [{}]: Shape={}, DType={}, Order={}, ExpectedBytes={}, FlatBufferReportedLength={}",
                        varName, shapeForLog, dataType, order, expectedBytes, fbBufferLength);

                // *** MODIFIED VALIDATION START ***
                if (expectedBytes < 0 || expectedBytes > Integer.MAX_VALUE) {
                    log.error("LOAD_INLINE [{}]: Calculated expected byte count ({}) is invalid or exceeds Integer.MAX_VALUE. Cannot load.", varName, expectedBytes);
                    return null;
                }
                // Sanity check: Can we access the last expected byte index using fa.buffer(j)?
                // This relies on fa.buffer(j) throwing if index is out of bounds based on the *actual* vector size.
                try {
                    if (expectedBytes > 0) {
                        log.trace("LOAD_INLINE [{}]: Performing bounds check by accessing index {}.", varName, (int)expectedBytes - 1);
                        // Access the last byte to trigger potential IndexOutOfBoundsException
                        @SuppressWarnings("unused") // We only care about the potential exception
                        byte ignored = (byte) fa.buffer((int)expectedBytes - 1);
                        log.trace("LOAD_INLINE [{}]: Bounds check via fa.buffer(last_index) passed.", varName);
                    } else {
                        log.trace("LOAD_INLINE [{}]: Skipping bounds check as expectedBytes is 0.", varName);
                    }
                } catch (IndexOutOfBoundsException e) {
                    log.error("LOAD_INLINE [{}]: FlatBuffer vector access failed for expected byte count {} (index {}). Reported length was {}. Underlying vector likely too small or access method failed.",
                            varName, expectedBytes, (expectedBytes > 0 ? (int)expectedBytes - 1 : 0), fbBufferLength, e);
                    return null; // Cannot proceed if bounds check fails
                } catch (Exception e) { // Catch other potential issues like NullPointerException
                    log.error("LOAD_INLINE [{}]: Unexpected error during FlatBuffer vector bounds check for expected byte count {} (index {}). Reported length was {}.",
                            varName, expectedBytes, (expectedBytes > 0 ? (int)expectedBytes - 1 : 0), fbBufferLength, e);
                    return null;
                }
                // Log if fbBufferLength was drastically different, but proceed anyway
                if (expectedBytes != fbBufferLength && fbBufferLength >= 0) { // Check >=0 to avoid logging for weird negative lengths
                    log.warn("LOAD_INLINE [{}]: FlatBuffer reported length ({}) differs from expected length ({}). Proceeding based on expected length from shape/dtype.",
                            varName, fbBufferLength, expectedBytes);
                }
                // *** MODIFIED VALIDATION END ***


                // If empty array based on shape, return empty array
                if (totalElements == 0) {
                    log.trace("LOAD_INLINE [{}]: Shape {} implies zero elements. Returning empty array.", varName, shapeForLog);
                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        return Nd4j.create(dataType, shape, order);
                    }
                }

                // Read bytes manually using fa.buffer(j) up to expectedBytes
                log.debug("LOAD_INLINE [{}]: Reading {} bytes manually using fa.buffer(j)...", varName, expectedBytes);
                byte[] readBytes = new byte[(int)expectedBytes]; // Safe cast due to earlier check
                boolean readSuccess = true;
                try {
                    for(int j=0; j<expectedBytes; j++) {
                        // Read unsigned byte and cast to signed byte
                        readBytes[j] = (byte)fa.buffer(j);
                    }
                } catch (Exception e) {
                    log.error("LOAD_INLINE [{}]: Exception during manual byte reading fa.buffer(j). Read incomplete.", varName, e);
                    readSuccess = false;
                }

                if (!readSuccess) {
                    log.error("LOAD_INLINE [{}]: Failed to read bytes manually from FlatBuffer.", varName);
                    return null; // Deserialization failed
                }
                log.debug("LOAD_INLINE [{}]: Manual byte reading complete.", varName);
                if (isTargetVar) { // Log first few bytes for target var
                    int logLen = Math.min(16, readBytes.length);
                    log.info("LOAD_INLINE_TRACE [{}]: First {} manually read bytes: [{}]", varName, logLen, bytesToHex(readBytes, 0, logLen));
                }


                // Wrap the manually read bytes and create the INDArray
                ByteBuffer bbManual = ByteBuffer.wrap(readBytes).order(ByteOrder.nativeOrder());
                INDArray result = null;
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    result = Nd4j.create(dataType, shape, order);
                }
                if (result == null) {
                    log.error("LOAD_INLINE [{}]: Nd4j.create returned null for Shape {}.", varName, shapeForLog); return null;
                }
                DataBuffer targetBuffer = result.data();
                if (targetBuffer == null) { log.error("LOAD_INLINE [{}]: Target DataBuffer is null after creating array shape {}.", varName, shapeForLog); return null; }

                // Copy data from bbManual to targetBuffer
                ByteBuffer targetNio = targetBuffer.asNio();
                if(targetNio != null && result.offset() == 0 ) {
                    log.trace("LOAD_INLINE [{}]: Using bulk NIO copy (Target Offset is 0) from manually read bytes.", varName);
                    targetNio.order(ByteOrder.nativeOrder());
                    // Ensure target position/limit are set correctly for the put operation
                    targetNio.clear(); // Reset position=0, limit=capacity
                    targetNio.limit((int) expectedBytes); // Limit to where data should end
                    // Ensure source buffer limit is set correctly
                    bbManual.clear(); // Reset position=0, limit=capacity(expectedBytes)
                    try {
                        targetNio.put(bbManual); // Bulk copy from bbManual's position 0 up to its limit
                        log.trace("LOAD_INLINE [{}]: Bulk NIO copy finished. Target buffer pos after put: {}", varName, targetNio.position());
                        // Verify position advanced correctly
                        if(targetNio.position() != expectedBytes) {
                            log.warn("LOAD_INLINE [{}]: Target buffer position ({}) after bulk put does not match expected bytes ({}).", varName, targetNio.position(), expectedBytes);
                        }
                    } catch (Exception e) { log.error("LOAD_INLINE [{}]: Exception during bulk NIO copy from manual bytes!", varName, e); return null; }

                } else {
                    // Fallback to element-wise copy from manually read buffer
                    log.warn("LOAD_INLINE [{}]: Using element-wise copy for shape {} from manually read bytes. Reason: Target NIO buffer null? {}, Target Offset = {}",
                            varName, shapeForLog, (targetNio == null), result.offset());
                    bbManual.clear(); // Reset position=0 for relative gets
                    try {
                        for (long i = 0; i < totalElements; i++) {
                            if (bbManual.remaining() < elementSize) { log.error("LOAD_INLINE [{}]: Manual ByteBuffer ran out of data unexpectedly at element {}/{}.", varName, i, totalElements); break; }
                            // Use appropriate putScalar variant based on type
                            switch (dataType) {
                                case FLOAT: result.putScalar(i, bbManual.getFloat()); break;
                                case DOUBLE: result.putScalar(i, bbManual.getDouble()); break;
                                case INT: result.putScalar(i, bbManual.getInt()); break;
                                case LONG: result.putScalar(i, bbManual.getLong()); break;
                                case SHORT: result.putScalar(i, bbManual.getShort()); break;
                                case BYTE: result.putScalar(i, bbManual.get()); break;
                                case UBYTE: result.putScalar(i, bbManual.get() & 0xFF); break;
                                case UINT16: result.putScalar(i, bbManual.getShort() & 0xFFFF); break;
                                case UINT32: result.putScalar(i, bbManual.getInt() & 0xFFFFFFFFL); break;
                                case UINT64: result.putScalar(i, bbManual.getLong()); break; // Might need BigInteger for full range if Java doesn't support unsigned long directly
                                case BOOL: result.putScalar(i, bbManual.get() != 0); break;
                                case BFLOAT16: // Nd4j might not have direct putScalar for bfloat16, cast from float
                                    if (bbManual.remaining() >= 2) { result.putScalar(i, HalfPrecisionUtil.bfloat16ToFloat(bbManual.getShort())); }
                                    else { log.error("LOAD_INLINE [{}]: Insufficient bytes for BFLOAT16 at element {}/{}.", varName, i, totalElements); i = totalElements; } // Exit loop
                                    break;
                                case HALF:
                                    if (bbManual.remaining() >= 2) { result.putScalar(i, HalfPrecisionUtil.toFloat(bbManual.getShort())); }
                                    else { log.error("LOAD_INLINE [{}]: Insufficient bytes for HALF at element {}/{}.", varName, i, totalElements); i = totalElements; } // Exit loop
                                    break;
                                default: log.warn("LOAD_INLINE [{}]: Skipping unsupported type {} for element-wise copy.", varName, dataType); bbManual.position(bbManual.position() + elementSize);
                            }
                        }
                        // Check if loop finished correctly
                        if(bbManual.hasRemaining()) {
                            log.warn("LOAD_INLINE [{}]: Fallback copy finished, but source buffer still has {} bytes remaining.", varName, bbManual.remaining());
                        }
                    } catch (Exception e) { log.warn("LOAD_INLINE [{}]: Error during element-wise copy from manual bytes.", varName, e); }
                }
                // Log first value from the final array
                if (isTargetVar && result.length() > 0) {
                    try { log.info("LOAD_INLINE_TRACE [{}]: Final first element value (as float): {}", varName, result.getFloat(0)); } catch(Exception e) {/*ignore*/}
                }
                return result; // Return the populated array

            } catch (Exception e) {
                log.error("LOAD_INLINE [{}]: Failed to deserialize inline FlatArray (Shape {}): {}", varName, shapeForLog, e.getMessage(), e);
                return null;
            }
        }

        // If FlatArray doesn't match expected inline structure (e.g., uses bufferChunks or isExternal)
        log.warn("LOAD_INLINE [{}]: FlatArray structure did not match expected inline format or buffer length issue. BufferLength={}, BufferChunks={}, IsExternal={}. Returning null.",
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
    private static int detectShardCount(File parentDir, String baseName) throws IOException { int numShards = -1; final String filePrefix = baseName + ".shard"; File[] matchingFiles = parentDir.listFiles((dir, name) -> name.startsWith(filePrefix) && name.endsWith(".sdnb")); if (matchingFiles == null || matchingFiles.length == 0) throw new FileNotFoundException("No shard files matching pattern '" + filePrefix + "*.sdnb' found in " + parentDir.getAbsolutePath()); Pattern p = Pattern.compile(Pattern.quote(baseName) + "\\.shard\\d+-of-(\\d+)\\.sdnb$"); for (File f : matchingFiles) { Matcher m = p.matcher(f.getName()); if (m.matches()) { try { int count = Integer.parseInt(m.group(1)); if (numShards == -1) numShards = count; else if (numShards != count) throw new IOException("Inconsistent shard counts (expected " + numShards + " found " + count + " in " + f.getName() + ")"); } catch (NumberFormatException e) { log.warn("Bad shard count in {}. Skipping.", f.getName()); } } } if (numShards <= 0) throw new IOException("No valid shard count found in " + parentDir.getAbsolutePath()); return numShards; }
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