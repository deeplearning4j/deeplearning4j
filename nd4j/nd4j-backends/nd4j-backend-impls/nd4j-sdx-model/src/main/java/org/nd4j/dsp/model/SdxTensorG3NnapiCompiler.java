/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.QuantizedMatmul;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * In-process Tensor G3 compiler for calibrated NNAPI weight-INT8 matrix multiplies.
 *
 * <p>The compiler loads the canonical SameDiff graph, rewrites supported dense
 * constant-weight {@code matmul}/{@code mmul} nodes to the explicit five-input
 * {@code quantized_matmul}, and annotates packed two-input Q4_K
 * {@code ggml_qmatmul} nodes with source-bound per-operation calibration.
 * Packed Q4 weights and the reference CPU/CUDA op ABI remain unchanged. The
 * derived SDZ preserves every non-model entry byte-for-byte so tokenizer, model
 * configuration, chat template, and generation assets remain bundled.</p>
 */
public final class SdxTensorG3NnapiCompiler implements SdxModelCompiler.TargetCompiler {
    public static final String COMPILER_ID = "sdx-tensor-g3-nnapi";
    public static final String COMPILER_VERSION = "6";
    public static final String TARGET_SOC = "Tensor_G3";

    private static final String TENSOR_G3 = TARGET_SOC;
    // Keep the structural ABI value local so the standalone mobile packager can
    // compile this module against its already-installed nd4j-api dependency.
    private static final long WEIGHT_LAYOUT_N_BY_K = 1L;
    private static final long ZIP_TIME = 0L;

    @Override
    public String id() {
        return COMPILER_ID;
    }

    @Override
    public String version() {
        return COMPILER_VERSION;
    }

    @Override
    public String cacheKeyMaterial(
            Path sourceModel,
            SdxTargetProfile target,
            SdxModelCompiler.CompileOptions options) throws IOException {
        validateRequest(target, options);
        SdxSourceIdentity sourceIdentity = SdxSourceIdentity.identify(sourceModel);
        SdxQuantizationContract contract =
                SdxQuantizationContract.load(requireQuantization(options));
        contract.validateForCompilation(sourceIdentity, target, options.targetSoc());
        return "graphRewrite=quantized_matmul-5-native-nk+ggml-q4-calibration-sargs-v2-mixed;"
                + "sdzStorage=stored-native-memory;quantization="
                + contract.summaryJson();
    }

    @Override
    public Path compile(SdxModelCompiler.CompilationContext context) throws Exception {
        validateRequest(context.target(), context.options());
        Path contractPath = requireQuantization(context.options());
        SdxQuantizationContract contract = SdxQuantizationContract.load(contractPath);
        contract.validateForCompilation(
                context.sourceIdentity(), context.target(), context.options().targetSoc());

        Path source = context.sourceModel().toAbsolutePath().normalize();
        Path modelOutput = context.suggestedModelOutput().toAbsolutePath().normalize();
        Path generated = context.workDirectory().resolve(
                ".tensor-g3-" + UUID.randomUUID() + ".graph.sdz");
        try {
            SameDiff graph;
            try {
                graph = SDZSerializer.load(source.toFile(), false);
            } catch (RuntimeException failure) {
                throw new IOException("Unable to load canonical SameDiff SDZ " + source, failure);
            }
            try (SameDiff closeable = graph) {
                int prepared = prepareEligibleMatmuls(closeable, contract);
                if (prepared == 0) {
                    throw new IOException(
                            "Tensor G3 NNAPI compilation found no eligible operations covered by its quantization contract");
                }
                SDZSerializer.save(closeable, generated.toFile(), false, null);
            }

            mergeDerivedSdz(source, generated, modelOutput);
            SdxSourceIdentity derivedIdentity = SdxSourceIdentity.identify(modelOutput);
            if (context.sourceIdentity().sha256().equals(derivedIdentity.sha256())) {
                throw new IOException("Tensor G3 NNAPI graph rewrite produced an unchanged SDZ");
            }

            SdxNnapiDevicePolicy.create(
                            context.target(),
                            context.options().targetSoc(),
                            context.sourceIdentity(),
                            derivedIdentity)
                    .write(context.suggestedOutput());
            return context.suggestedOutput();
        } finally {
            Files.deleteIfExists(generated);
        }
    }

    private static int prepareEligibleMatmuls(
            SameDiff graph, SdxQuantizationContract contract) throws IOException {
        List<SameDiffOp> denseCandidates = new ArrayList<>();
        List<SameDiffOp> q4Candidates = new ArrayList<>();
        for (SameDiffOp op : new ArrayList<>(graph.getOps().values())) {
            String opName = op.getOp().opName().toLowerCase(Locale.ROOT);
            if ("matmul".equals(opName) || "mmul".equals(opName)) {
                denseCandidates.add(op);
            } else if ("ggml_qmatmul".equals(opName) && isQ4K(op)) {
                q4Candidates.add(op);
            }
        }
        if (contract.isTensorG3Q4PerOperator()) {
            return annotateQ4Matmuls(graph, q4Candidates, contract);
        }
        if (!q4Candidates.isEmpty()) {
            throw new IOException(
                    "Tensor G3 Q4_K operations require a source-bound per-op calibration contract");
        }

        int prepared = 0;
        Set<String> rewrittenWeightVariables = new HashSet<>();
        for (SameDiffOp candidate : denseCandidates) {
            List<String> inputs = candidate.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                throw unsupported(candidate, "requires exactly two inputs");
            }
            List<String> originalInputs = new ArrayList<>(inputs);
            SDVariable activation = graph.getVariable(inputs.get(0));
            SDVariable weightVariable = graph.getVariable(inputs.get(1));
            if (activation == null || weightVariable == null) {
                throw unsupported(candidate, "references a missing input variable");
            }
            if (weightVariable.getVariableType() != VariableType.CONSTANT) {
                throw unsupported(candidate, "has dynamic/non-constant weights");
            }
            INDArray weights = weightVariable.getArr();
            if (weights == null) {
                throw unsupported(candidate, "constant weights have no materialized array");
            }
            if (weights.rank() != 2
                    || weights.ordering() != 'c'
                    || !Shape.strideDescendingCAscendingF(weights)) {
                throw unsupported(candidate, "requires dense rank-2 C-order weights");
            }
            DataType activationType = activation.dataType();
            if (activationType != DataType.FLOAT && activationType != DataType.HALF) {
                throw unsupported(candidate, "activation dtype must be FLOAT or HALF");
            }
            if (weights.dataType() != DataType.FLOAT && weights.dataType() != DataType.HALF) {
                throw unsupported(candidate, "weight dtype must be FLOAT or HALF");
            }
            if (!(candidate.getOp() instanceof DynamicCustomOp)) {
                throw unsupported(candidate, "does not expose a supported dynamic matmul layout");
            }
            long[] integerArgs = ((DynamicCustomOp) candidate.getOp()).iArgs();
            if (integerArgs != null) {
                for (long value : integerArgs) {
                    if (value != 0L) {
                        throw unsupported(candidate, "transpose/batched matmul layouts are unsupported");
                    }
                }
            }

            String base = candidate.getName() + "__tensor_g3";
            SDVariable quantizedWeight = graph.constant(
                    uniqueName(graph, base + "_weight_int8"),
                    quantizeSignedSymmetricNativeNByK(weights, contract.weightScale()));
            SDVariable activationScale = graph.constant(
                    uniqueName(graph, base + "_activation_scale"),
                    Nd4j.scalar(contract.activationScale()));
            SDVariable weightScale = graph.constant(
                    uniqueName(graph, base + "_weight_scale"),
                    Nd4j.scalar(contract.weightScale()));
            SDVariable outputScale = graph.constant(
                    uniqueName(graph, base + "_output_scale"),
                    Nd4j.scalar(contract.outputScale()));

            QuantizedMatmul replacement = new QuantizedMatmul();
            replacement.setSameDiff(graph);
            replacement.setOwnName(candidate.getName());
            replacement.addIArgument(WEIGHT_LAYOUT_N_BY_K);
            detachOpFromInputs(graph, candidate.getName(), originalInputs);
            candidate.setOp(replacement);
            String[] replacementInputs = {
                    activation.name(),
                    quantizedWeight.name(),
                    activationScale.name(),
                    weightScale.name(),
                    outputScale.name()
            };
            graph.addArgsFor(replacementInputs, replacement);
            rewrittenWeightVariables.add(weightVariable.name());
            prepared++;
        }
        pruneUnreferencedRewrittenWeights(graph, rewrittenWeightVariables);
        return prepared;
    }

    private static boolean isQ4K(SameDiffOp candidate) throws IOException {
        if (!(candidate.getOp() instanceof DynamicCustomOp)) {
            throw unsupported(candidate, "does not expose custom-op arguments");
        }
        long[] integerArgs = ((DynamicCustomOp) candidate.getOp()).iArgs();
        if (integerArgs == null || integerArgs.length == 0) {
            throw unsupported(candidate, "has no quantization type argument");
        }
        return integerArgs[0] == 8L;
    }

    private static int annotateQ4Matmuls(
            SameDiff graph,
            List<SameDiffOp> q4Candidates,
            SdxQuantizationContract contract) throws IOException {
        Set<String> annotatedQ4Ops = new HashSet<>();
        for (SameDiffOp candidate : q4Candidates) {
            SdxQuantizationContract.OperatorCalibration calibration =
                    contract.operatorCalibration(candidate.getName());
            if (calibration == null) {
                throw unsupported(candidate,
                        "has no source-bound per-op calibration metadata");
            }
            List<String> inputs = candidate.getInputsToOp();
            if (inputs == null || inputs.size() != 2) {
                throw unsupported(candidate, "requires exactly two Q4 inputs");
            }
            SDVariable packedWeight = graph.getVariable(inputs.get(1));
            if (packedWeight == null
                    || (packedWeight.getVariableType() != VariableType.CONSTANT
                    && packedWeight.getVariableType() != VariableType.VARIABLE)
                    || packedWeight.getArr() == null
                    || packedWeight.getArr().dataType() != DataType.INT8
                    || packedWeight.getArr().rank() != 1) {
                throw unsupported(candidate,
                        "requires materialized inference-static rank-1 INT8 packed weights");
            }
            DynamicCustomOp q4 = (DynamicCustomOp) candidate.getOp();
            long[] integerArgs = q4.iArgs();
            if (integerArgs == null || integerArgs.length != 4
                    || integerArgs[1] <= 0L
                    || integerArgs[2] <= 0L || integerArgs[2] % 256L != 0L
                    || (integerArgs[3] != 0L && integerArgs[3] != 1L)) {
                throw unsupported(candidate, "is not a valid Q4_K contract");
            }
            String[] existing = q4.sArgs();
            if (existing != null && existing.length != 0) {
                throw unsupported(candidate,
                        "already contains conflicting string metadata");
            }
            q4.addSArgument(calibration.nnapiQ4SArguments(
                    contract.calibrationSampleCount(),
                    contract.calibrationDatasetSha256()));
            annotatedQ4Ops.add(candidate.getName());
        }
        for (String calibratedOp : contract.operatorCalibrations().keySet()) {
            if (!annotatedQ4Ops.contains(calibratedOp)) {
                throw new IOException("Tensor G3 NNAPI calibration references stale or "
                        + "non-Q4 op " + calibratedOp);
            }
        }
        return annotatedQ4Ops.size();
    }

    private static INDArray quantizeSignedSymmetricNativeNByK(INDArray weights, float scale)
            throws IOException {
        if (!(scale > 0.0f) || !Float.isFinite(scale)) {
            throw new IOException("weights.scale must be finite and positive");
        }
        long length = weights.length();
        if (length > Integer.MAX_VALUE) {
            throw new IOException("Tensor G3 weight tensor is too large for in-process quantization");
        }
        long kSize = weights.size(0);
        long nSize = weights.size(1);
        byte[] quantized = new byte[(int) length];
        for (long n = 0; n < nSize; n++) {
            for (long k = 0; k < kSize; k++) {
                double value = weights.getDouble(k, n);
                if (!Double.isFinite(value)) {
                    throw new IOException("Tensor G3 weight tensor contains a non-finite value");
                }
                long rounded = Math.round(value / scale);
                int nativeIndex = (int) (n * kSize + k);
                quantized[nativeIndex] =
                        (byte) Math.max(-127L, Math.min(127L, rounded));
            }
        }
        return Nd4j.createFromArray(quantized).reshape('c', nSize, kSize);
    }

    private static void detachOpFromInputs(
            SameDiff graph, String opName, List<String> inputNames) {
        for (String inputName : inputNames) {
            Variable input = graph.getVariables().get(inputName);
            if (input == null || input.getInputsForOp() == null) {
                continue;
            }
            while (input.getInputsForOp().remove(opName)) {
                // Remove every stale occurrence before installing replacement inputs.
            }
        }
    }

    private static void pruneUnreferencedRewrittenWeights(
            SameDiff graph, Set<String> weightNames) {
        List<String> outputs = graph.outputs();
        for (String weightName : weightNames) {
            Variable weight = graph.getVariables().get(weightName);
            if (weight == null || (outputs != null && outputs.contains(weightName))) {
                continue;
            }
            List<String> consumers = weight.getInputsForOp();
            if (consumers == null || consumers.isEmpty()) {
                graph.getVariables().remove(weightName);
            }
        }
    }

    private static String uniqueName(SameDiff graph, String requested) {
        String candidate = requested;
        int suffix = 2;
        while (graph.hasVariable(candidate)) {
            candidate = requested + "_" + suffix++;
        }
        return candidate;
    }

    private static IOException unsupported(SameDiffOp op, String reason) {
        return new IOException(
                "Tensor G3 NNAPI cannot rewrite " + op.getName() + ": " + reason);
    }

    private static void validateRequest(
            SdxTargetProfile target, SdxModelCompiler.CompileOptions options)
            throws IOException {
        if (target != SdxTargetProfile.ANDROID_ARM64_NNAPI_ACCELERATOR) {
            throw new IOException(
                    "Tensor G3 NNAPI compiler cannot compile target " + target.id());
        }
        if (!TENSOR_G3.equals(options.targetSoc())) {
            throw new IOException(
                    "Tensor G3 NNAPI compiler requires targetSoc " + TENSOR_G3);
        }
    }

    private static Path requireQuantization(SdxModelCompiler.CompileOptions options)
            throws IOException {
        if (options.quantizationConfig() == null) {
            throw new IOException(
                    "Tensor G3 NNAPI compiler requires a calibrated quantization contract");
        }
        return options.quantizationConfig();
    }

    private static void mergeDerivedSdz(Path source, Path generated, Path output)
            throws IOException {
        Path parent = output.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path pending = output.resolveSibling(
                "." + output.getFileName() + "." + UUID.randomUUID() + ".pending");
        Files.deleteIfExists(pending);
        Set<String> names = new HashSet<>();
        try (ZipFile sourceZip = new ZipFile(source.toFile());
             ZipFile generatedZip = new ZipFile(generated.toFile());
             ZipOutputStream destination = new ZipOutputStream(
                     new BufferedOutputStream(Files.newOutputStream(
                             pending, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)))) {
            Enumeration<? extends ZipEntry> sourceEntries = sourceZip.entries();
            while (sourceEntries.hasMoreElements()) {
                ZipEntry entry = sourceEntries.nextElement();
                SdxSourceIdentity.requireSafeEntryName(entry.getName());
                if (!isSdnbModelEntry(sourceZip, entry)) {
                    copyEntry(sourceZip, entry, destination, names, false);
                }
            }
            Enumeration<? extends ZipEntry> generatedEntries = generatedZip.entries();
            int modelEntries = 0;
            while (generatedEntries.hasMoreElements()) {
                ZipEntry entry = generatedEntries.nextElement();
                if (!entry.isDirectory()) {
                    SdxSourceIdentity.requireSafeEntryName(entry.getName());
                    copyEntry(generatedZip, entry, destination, names, true);
                    modelEntries++;
                }
            }
            if (modelEntries == 0) {
                throw new IOException("Rewritten SameDiff SDZ contains no model entries");
            }
        } catch (Throwable failure) {
            Files.deleteIfExists(pending);
            if (failure instanceof IOException) {
                throw (IOException) failure;
            }
            throw new IOException("Unable to write derived Tensor G3 SDZ", failure);
        }
        try {
            Files.move(
                    pending,
                    output,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(pending, output, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(pending);
        }
    }

    private static boolean isSdnbModelEntry(ZipFile zip, ZipEntry entry) throws IOException {
        if (entry.isDirectory()) {
            return false;
        }
        try (InputStream input = zip.getInputStream(entry)) {
            byte[] magic = input.readNBytes(4);
            return magic.length == 4
                    && magic[0] == 'S'
                    && magic[1] == 'D'
                    && magic[2] == 'N'
                    && magic[3] == 'B';
        }
    }

    private static void copyEntry(
            ZipFile input,
            ZipEntry source,
            ZipOutputStream output,
            Set<String> names,
            boolean storeForNativeMapping) throws IOException {
        if (!names.add(source.getName())) {
            throw new IOException("Duplicate derived SDZ entry: " + source.getName());
        }
        ZipEntry destination = new ZipEntry(source.getName());
        destination.setTime(ZIP_TIME);
        if (storeForNativeMapping && !source.isDirectory()) {
            long size = source.getSize();
            long crc = source.getCrc();
            if (size < 0L || crc < 0L) {
                throw new IOException(
                        "Generated SDZ entry lacks size/CRC metadata: " + source.getName());
            }
            destination.setMethod(ZipEntry.STORED);
            destination.setSize(size);
            destination.setCompressedSize(size);
            destination.setCrc(crc);
        }
        output.putNextEntry(destination);
        if (!source.isDirectory()) {
            try (InputStream stream =
                         new BufferedInputStream(input.getInputStream(source))) {
                byte[] buffer = new byte[1024 * 1024];
                int read;
                while ((read = stream.read(buffer)) >= 0) {
                    if (read > 0) {
                        output.write(buffer, 0, read);
                    }
                }
            }
        }
        output.closeEntry();
    }
}
