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

package org.nd4j.ggml.export;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.ggml.GGMLExportException;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFHeader;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.ggml.format.GGUFWriter;
import org.nd4j.ggml.quantization.Dequantizer;
import org.nd4j.ggml.quantization.DequantizerFactory;
import org.nd4j.ggml.quantization.Quantizer;
import org.nd4j.ggml.quantization.QuantizerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Bounded-memory GGUF-to-GGUF requantization.
 *
 * <p>The source is decoded and re-encoded in block-aligned chunks. At no point is a complete
 * SameDiff graph or complete tensor materialized, which makes the conversion suitable for
 * memory-constrained import workers.</p>
 */
@Slf4j
public final class GGUFRequantizer {
    private static final int MAX_ELEMENTS_PER_CHUNK = 1024 * 1024;
    private static final int COPY_CHUNK_BYTES = 4 * 1024 * 1024;

    private GGUFRequantizer() {
    }

    public static void requantize(File inputFile, File outputFile,
                                  ExportOptions.QuantizationType targetQuantization)
            throws GGMLExportException {
        if (inputFile == null || outputFile == null || targetQuantization == null) {
            throw new NullPointerException("Input, output, and target quantization are required");
        }

        try {
            File input = inputFile.getCanonicalFile();
            File output = outputFile.getCanonicalFile();
            if (!input.isFile()) {
                throw new GGMLExportException("Input GGUF is not a regular file: " + input);
            }
            if (input.equals(output)) {
                throw new GGMLExportException("Input and output GGUF paths must be different");
            }

            boolean completed = false;
            try {
                requantizeVerified(input, output, targetQuantization);
                completed = true;
            } finally {
                if (!completed) {
                    try {
                        Files.deleteIfExists(output.toPath());
                    } catch (IOException cleanupFailure) {
                        log.warn("Could not remove incomplete requantized GGUF {}", output,
                                cleanupFailure);
                    }
                }
            }
        } catch (GGMLExportException failure) {
            throw failure;
        } catch (IOException | GGMLImportException failure) {
            throw new GGMLExportException("Could not stream GGUF requantization from "
                    + inputFile + " to " + outputFile, failure);
        }
    }

    private static void requantizeVerified(File input, File output,
                                           ExportOptions.QuantizationType targetQuantization)
            throws IOException, GGMLImportException, GGMLExportException {
        try (GGUFReader reader = new GGUFReader(input)) {
            GGUFHeader header = reader.getHeader();
            List<GGMLTensorInfo> sourceTensors = reader.getTensorInfos();
            List<GGMLDataType> targetTypes = new ArrayList<>(sourceTensors.size());
            for (GGMLTensorInfo tensor : sourceTensors) {
                targetTypes.add(selectTargetType(tensor, targetQuantization));
            }

            log.info("Streaming GGUF requantization: input={}, output={}, tensors={}, target={}",
                    input.getName(), output.getName(), sourceTensors.size(), targetQuantization);

            try (GGUFWriter writer = new GGUFWriter(output, header.getVersion())) {
                writer.setAlignment(header.getAlignment());
                for (Map.Entry<String, Object> metadata : header.getMetadata().entrySet()) {
                    if (!GGUFHeader.KEY_GENERAL_FILE_TYPE.equals(metadata.getKey())
                            && !GGUFHeader.KEY_GENERAL_ALIGNMENT.equals(metadata.getKey())) {
                        writer.addMetadata(metadata.getKey(), metadata.getValue());
                    }
                }
                if (header.getAlignment() != 32) {
                    writer.addMetadata(GGUFHeader.KEY_GENERAL_ALIGNMENT, header.getAlignment());
                }
                if (targetQuantization.toGGMLDataType().isQuantized()) {
                    writer.addMetadata(GGUFHeader.KEY_GENERAL_QUANTIZATION_VERSION, 2);
                }

                for (int i = 0; i < sourceTensors.size(); i++) {
                    GGMLTensorInfo tensor = sourceTensors.get(i);
                    writer.registerTensor(tensor.getName(), tensor.getShape(), targetTypes.get(i));
                }
                writer.writeHeader();

                for (int i = 0; i < sourceTensors.size(); i++) {
                    GGMLTensorInfo tensor = sourceTensors.get(i);
                    GGMLDataType targetType = targetTypes.get(i);
                    log.info("Requantizing tensor {}/{} {}: {} -> {}, elements={}",
                            i + 1, sourceTensors.size(), tensor.getName(), tensor.getDataType(),
                            targetType, tensor.getNumElements());
                    try {
                        writer.beginTensorData(tensor.getName());
                        if (targetType == tensor.getDataType()) {
                            copyTensor(reader, writer, tensor);
                        } else {
                            convertTensor(reader, writer, tensor, targetType);
                        }
                        writer.endTensorData();
                    } catch (IOException | RuntimeException failure) {
                        throw new GGMLExportException("Could not requantize tensor " + (i + 1)
                                + "/" + sourceTensors.size() + " " + tensor.getName()
                                + " shape=" + tensor.getShapeString()
                                + " type=" + tensor.getDataType() + " -> " + targetType, failure);
                    }
                }
                writer.finalizeFile();
            }
        }
    }

    private static GGMLDataType selectTargetType(
            GGMLTensorInfo tensor, ExportOptions.QuantizationType targetQuantization) {
        GGMLDataType sourceType = tensor.getDataType();
        GGMLDataType requestedType = targetQuantization.toGGMLDataType();

        if (!isNumeric(sourceType)) {
            return sourceType;
        }

        String name = tensor.getName();
        if (name.contains("norm")) {
            return GGMLDataType.GGML_TYPE_F32;
        }
        if (tensor.getNumDimensions() < 2 || !requestedType.isQuantized()) {
            return tensor.getNumDimensions() < 2 ? sourceType : requestedType;
        }

        GGMLDataType preferredType = requestedType;
        if (isFourBit(requestedType)
                && (name.contains("token_embd") || name.contains("output.weight"))) {
            preferredType = GGMLDataType.GGML_TYPE_Q8_0;
        }
        if (hasBlockCompatibleRows(tensor, preferredType)) {
            return preferredType;
        }

        // K-quants require each GGUF row (dimension 0) to contain complete 256-element
        // blocks. Hybrid/state-space models also contain narrow rank-2 tensors. Use Q8_0
        // where a 32-element block fits, otherwise retain the dense source tensor.
        if ((isFourBit(requestedType) || requestedType == GGMLDataType.GGML_TYPE_Q6_K)
                && hasBlockCompatibleRows(tensor, GGMLDataType.GGML_TYPE_Q8_0)) {
            return GGMLDataType.GGML_TYPE_Q8_0;
        }
        return sourceType;
    }

    private static boolean hasBlockCompatibleRows(GGMLTensorInfo tensor, GGMLDataType type) {
        long[] shape = tensor.getShape();
        return shape != null && shape.length > 0 && shape[0] > 0
                && shape[0] % type.getBlockSize() == 0;
    }

    private static boolean isNumeric(GGMLDataType type) {
        return type.isQuantized() || type.getNd4jType() != null;
    }

    private static boolean isFourBit(GGMLDataType type) {
        return type == GGMLDataType.GGML_TYPE_Q4_0
                || type == GGMLDataType.GGML_TYPE_Q4_1
                || type == GGMLDataType.GGML_TYPE_Q4_K;
    }

    private static void copyTensor(GGUFReader reader, GGUFWriter writer, GGMLTensorInfo tensor)
            throws IOException {
        long remaining = tensor.getDataSize();
        long byteOffset = 0;
        byte[] buffer = new byte[(int) Math.min(COPY_CHUNK_BYTES, remaining)];
        while (remaining > 0) {
            int length = (int) Math.min(buffer.length, remaining);
            reader.readTensorDataRange(tensor, byteOffset, buffer, 0, length);
            writer.writeTensorDataChunk(buffer, 0, length);
            byteOffset += length;
            remaining -= length;
        }
    }

    private static void convertTensor(GGUFReader reader, GGUFWriter writer, GGMLTensorInfo tensor,
                                      GGMLDataType targetType)
            throws IOException, GGMLExportException {
        GGMLDataType sourceType = tensor.getDataType();
        int sourceBlock = sourceType.getBlockSize();
        int targetBlock = targetType.getBlockSize();
        int chunkAlignment = leastCommonMultiple(sourceBlock, targetBlock);
        long remainingElements = tensor.getNumElements();
        long elementOffset = 0;

        Quantizer quantizer = null;
        if (targetType.isQuantized()) {
            if (!QuantizerFactory.hasQuantizer(targetType)) {
                throw new GGMLExportException("No quantizer available for " + targetType);
            }
            quantizer = QuantizerFactory.getQuantizer(targetType);
        }

        while (remainingElements > 0) {
            int elements = nextChunkElements(remainingElements, chunkAlignment);
            long sourceByteOffset = sourceType.calculateStorageBytes(elementOffset);
            int sourceBytes = checkedArrayLength(sourceType.calculateStorageBytes(elements),
                    "source tensor chunk");
            byte[] encodedSource = new byte[sourceBytes];
            reader.readTensorDataRange(tensor, sourceByteOffset, encodedSource, 0, sourceBytes);

            float[] floats = decode(encodedSource, elements, sourceType);
            byte[] encodedTarget = targetType.isQuantized()
                    ? quantizer.quantize(floats)
                    : encode(floats, targetType);
            writer.writeTensorDataChunk(encodedTarget, 0, encodedTarget.length);

            elementOffset += elements;
            remainingElements -= elements;
        }
    }

    private static int nextChunkElements(long remainingElements, int alignment) {
        long candidate = Math.min(MAX_ELEMENTS_PER_CHUNK, remainingElements);
        if (candidate < remainingElements) {
            candidate -= candidate % alignment;
            if (candidate == 0) {
                candidate = alignment;
            }
        }
        return Math.toIntExact(candidate);
    }

    private static int leastCommonMultiple(int left, int right) {
        return Math.multiplyExact(left / greatestCommonDivisor(left, right), right);
    }

    private static int greatestCommonDivisor(int left, int right) {
        int a = left;
        int b = right;
        while (b != 0) {
            int remainder = a % b;
            a = b;
            b = remainder;
        }
        return a;
    }

    private static int checkedArrayLength(long bytes, String description) throws GGMLExportException {
        if (bytes < 0 || bytes > Integer.MAX_VALUE) {
            throw new GGMLExportException(description + " exceeds Java array capacity: " + bytes);
        }
        return (int) bytes;
    }

    private static float[] decode(byte[] encoded, int elements, GGMLDataType sourceType)
            throws GGMLExportException {
        if (sourceType.isQuantized()) {
            if (!DequantizerFactory.hasDequantizer(sourceType)) {
                throw new GGMLExportException("No dequantizer available for " + sourceType);
            }
            Dequantizer dequantizer = DequantizerFactory.getDequantizer(sourceType);
            return dequantizer.dequantize(encoded, elements);
        }

        ByteBuffer source = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
        float[] floats = new float[elements];
        for (int i = 0; i < elements; i++) {
            switch (sourceType) {
                case GGML_TYPE_F32:
                    floats[i] = source.getFloat();
                    break;
                case GGML_TYPE_F16:
                    floats[i] = halfToFloat(source.getShort());
                    break;
                case GGML_TYPE_BF16:
                    floats[i] = Float.intBitsToFloat((source.getShort() & 0xffff) << 16);
                    break;
                case GGML_TYPE_F64:
                    floats[i] = (float) source.getDouble();
                    break;
                case GGML_TYPE_I8:
                    floats[i] = source.get();
                    break;
                case GGML_TYPE_I16:
                    floats[i] = source.getShort();
                    break;
                case GGML_TYPE_I32:
                    floats[i] = source.getInt();
                    break;
                case GGML_TYPE_I64:
                    floats[i] = source.getLong();
                    break;
                default:
                    throw new GGMLExportException("Unsupported source tensor type: " + sourceType);
            }
        }
        return floats;
    }

    private static byte[] encode(float[] floats, GGMLDataType targetType)
            throws GGMLExportException {
        int width;
        switch (targetType) {
            case GGML_TYPE_F32:
                width = Float.BYTES;
                break;
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
                width = Short.BYTES;
                break;
            case GGML_TYPE_F64:
                width = Double.BYTES;
                break;
            default:
                throw new GGMLExportException("Unsupported non-quantized target type: " + targetType);
        }

        byte[] encoded = new byte[Math.multiplyExact(floats.length, width)];
        ByteBuffer target = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : floats) {
            switch (targetType) {
                case GGML_TYPE_F32:
                    target.putFloat(value);
                    break;
                case GGML_TYPE_F16:
                    target.putShort(floatToHalf(value));
                    break;
                case GGML_TYPE_BF16:
                    target.putShort((short) (Float.floatToRawIntBits(value) >>> 16));
                    break;
                case GGML_TYPE_F64:
                    target.putDouble(value);
                    break;
                default:
                    throw new AssertionError(targetType);
            }
        }
        return encoded;
    }

    private static float halfToFloat(short half) {
        int bits = half & 0xffff;
        int sign = (bits & 0x8000) << 16;
        int exponent = (bits >>> 10) & 0x1f;
        int mantissa = bits & 0x03ff;

        if (exponent == 0) {
            if (mantissa == 0) {
                return Float.intBitsToFloat(sign);
            }
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= ~0x0400;
        } else if (exponent == 0x1f) {
            return Float.intBitsToFloat(sign | 0x7f800000 | (mantissa << 13));
        }

        int floatExponent = exponent + (127 - 15);
        return Float.intBitsToFloat(sign | (floatExponent << 23) | (mantissa << 13));
    }

    private static short floatToHalf(float value) {
        int bits = Float.floatToRawIntBits(value);
        int sign = (bits >>> 16) & 0x8000;
        int rounded = (bits & 0x7fffffff) + 0x1000;

        if (rounded >= 0x47800000) {
            if ((bits & 0x7fffffff) >= 0x47800000) {
                if (rounded < 0x7f800000) {
                    return (short) (sign | 0x7c00);
                }
                return (short) (sign | 0x7c00 | ((bits & 0x007fffff) >>> 13));
            }
            return (short) (sign | 0x7bff);
        }
        if (rounded >= 0x38800000) {
            return (short) (sign | ((rounded - 0x38000000) >>> 13));
        }
        if (rounded < 0x33000000) {
            return (short) sign;
        }

        int exponent = (bits & 0x7fffffff) >>> 23;
        int mantissa = (bits & 0x7fffff) | 0x800000;
        int shift = 126 - exponent;
        mantissa = (mantissa + (1 << (shift - 1))) >>> shift;
        return (short) (sign | mantissa);
    }
}
