/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.SeekableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Set;

/**
 * Validates a LiteRT-LM package before an artifact is admitted to the immutable
 * SDX model cache.
 *
 * <p>The fixed prefix mirrors {@code IsLiteRTLMFile} and
 * {@code ReadHeaderFromLiteRTLM} from Google LiteRT-LM commit
 * {@code 80f301ff9a3b02c2c1e7be2dd1a567752f7b51b6}. The pinned C++ reader uses
 * native-endian integer reads. SDX deliberately fixes little-endian decoding
 * for its supported x86-64 build hosts and Android arm64 targets, then validates
 * the pinned FlatBuffer metadata and section ranges without loading model data.
 */
final class LiteRtLmPackageValidator {
    private static final byte[] MAGIC =
            "LITERTLM".getBytes(StandardCharsets.US_ASCII);
    private static final int FIXED_HEADER_BYTES = 32;
    private static final int SUPPORTED_MAJOR_VERSION = 1;
    private static final int PINNED_MINOR_VERSION = 5;
    private static final int PINNED_MAX_SECTION_DATA_TYPE = 7;
    private static final int PINNED_MAX_VALUE_UNION_TYPE = 12;
    private static final long MAX_METADATA_BYTES = 64L * 1024L * 1024L;
    private static final long SECTION_BLOCK_BYTES = 16L * 1024L;
    private static final int STRING_VALUE_UNION_TYPE = 9;
    private static final Set<OpenOption> READ_NOFOLLOW =
            Set.of(StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS);

    private LiteRtLmPackageValidator() {}

    static void validate(Path artifact) throws IOException {
        if (artifact == null
                || !Files.isRegularFile(artifact, LinkOption.NOFOLLOW_LINKS)) {
            throw invalid(artifact, "artifact is not a regular no-follow file");
        }

        try (SeekableByteChannel channel =
                Files.newByteChannel(artifact, READ_NOFOLLOW)) {
            long fileSize = channel.size();
            if (fileSize < FIXED_HEADER_BYTES) {
                throw invalid(
                        artifact,
                        "fixed header is truncated: "
                                + fileSize
                                + " bytes, expected at least "
                                + FIXED_HEADER_BYTES);
            }

            ByteBuffer fixedHeader =
                    ByteBuffer.allocate(FIXED_HEADER_BYTES)
                            .order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, fixedHeader, artifact, "fixed header");
            fixedHeader.flip();

            for (byte expected : MAGIC) {
                if (fixedHeader.get() != expected) {
                    throw invalid(artifact, "magic must be LITERTLM");
                }
            }

            int majorVersion = fixedHeader.getInt();
            int minorVersion = fixedHeader.getInt();
            int patchVersion = fixedHeader.getInt();
            fixedHeader.getInt(); // Four padding bytes in the official format.
            long headerEndOffset = fixedHeader.getLong();

            if (majorVersion != SUPPORTED_MAJOR_VERSION) {
                throw invalid(
                        artifact,
                        "unsupported format version "
                                + Integer.toUnsignedString(majorVersion)
                                + "."
                                + Integer.toUnsignedString(minorVersion)
                                + "."
                                + Integer.toUnsignedString(patchVersion)
                                + "; supported major version is "
                                + SUPPORTED_MAJOR_VERSION);
            }
            if (headerEndOffset < FIXED_HEADER_BYTES
                    || headerEndOffset > fileSize) {
                throw invalid(
                        artifact,
                        "header end offset "
                                + headerEndOffset
                                + " is outside ["
                                + FIXED_HEADER_BYTES
                                + ", "
                                + fileSize
                                + "]");
            }

            long metadataSize = headerEndOffset - FIXED_HEADER_BYTES;
            if (metadataSize < Integer.BYTES) {
                throw invalid(artifact, "metadata FlatBuffer is missing");
            }
            if (metadataSize > MAX_METADATA_BYTES) {
                throw invalid(
                        artifact,
                        "metadata FlatBuffer exceeds the "
                                + MAX_METADATA_BYTES
                                + "-byte mobile safety limit");
            }

            ByteBuffer metadata =
                    ByteBuffer.allocate((int) metadataSize)
                            .order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, metadata, artifact, "metadata FlatBuffer");
            metadata.flip();
            new MetadataReader(metadata, artifact, minorVersion)
                    .validate(headerEndOffset, fileSize);
        } catch (UnsupportedOperationException failure) {
            throw new IOException(
                    "Cannot open LiteRT-LM package without following links: "
                            + artifact,
                    failure);
        }
    }

    private static void readFully(
            SeekableByteChannel channel,
            ByteBuffer target,
            Path artifact,
            String section)
            throws IOException {
        while (target.hasRemaining()) {
            int read = channel.read(target);
            if (read <= 0) {
                throw invalid(artifact, section + " is truncated");
            }
        }
    }

    private static IOException invalid(Path artifact, String reason) {
        return new IOException("Invalid LiteRT-LM package " + artifact + ": " + reason);
    }

    private static final class MetadataReader {
        private final ByteBuffer data;
        private final Path artifact;
        private final boolean enforcePinnedEnumRanges;

        private MetadataReader(ByteBuffer data, Path artifact, int minorVersion) {
            this.data = data;
            this.artifact = artifact;
            this.enforcePinnedEnumRanges =
                    Integer.compareUnsigned(minorVersion, PINNED_MINOR_VERSION) <= 0;
        }

        private void validate(long headerEndOffset, long fileSize)
                throws IOException {
            int rootPosition = offsetTarget(0, "metadata root");
            Table root = tableAt(rootPosition, "metadata root");

            Integer systemField = root.fieldLocation(
                    0, Integer.BYTES, false, "system_metadata");
            if (systemField != null) {
                Table system = tableAt(
                        offsetTarget(systemField, "system_metadata"),
                        "system_metadata");
                validateKeyValueVector(system, 0, true, "system_metadata.entries");
            }

            int sectionField = root.fieldLocation(
                    1, Integer.BYTES, true, "section_metadata");
            Table sectionMetadata = tableAt(
                    offsetTarget(sectionField, "section_metadata"),
                    "section_metadata");
            int objectsField = sectionMetadata.fieldLocation(
                    0, Integer.BYTES, true, "section_metadata.objects");
            int objectsVector =
                    offsetTarget(objectsField, "section_metadata.objects");
            int sectionCount = vectorLength(
                    objectsVector,
                    Integer.BYTES,
                    "section_metadata.objects");
            // SDX cache policy is intentionally stricter than FlatBuffers'
            // required-pointer rule: a runnable package must contain a section.
            if (sectionCount == 0) {
                throw malformed("section_metadata.objects is empty");
            }

            long previousEnd = -1L;
            for (int index = 0; index < sectionCount; index++) {
                int sectionPosition = vectorOffsetElement(
                        objectsVector,
                        index,
                        "section_metadata.objects[" + index + "]");
                Table section = tableAt(
                        sectionPosition,
                        "section_metadata.objects[" + index + "]");
                validateKeyValueVector(
                        section,
                        0,
                        false,
                        "section_metadata.objects[" + index + "].items");

                long begin = section.unsignedLong(
                        1,
                        "section_metadata.objects[" + index + "].begin_offset");
                long end = section.unsignedLong(
                        2,
                        "section_metadata.objects[" + index + "].end_offset");
                int dataType = section.unsignedByte(
                        3,
                        "section_metadata.objects[" + index + "].data_type");
                if (dataType == 0) {
                    throw malformed(
                            "section_metadata.objects["
                                    + index
                                    + "].data_type is NONE");
                }
                if (enforcePinnedEnumRanges
                        && dataType > PINNED_MAX_SECTION_DATA_TYPE) {
                    throw malformed(
                            "section_metadata.objects["
                                    + index
                                    + "].data_type "
                                    + dataType
                                    + " is outside the pinned 1."
                                    + PINNED_MINOR_VERSION
                                    + " schema");
                }
                if (begin < headerEndOffset || end <= begin || end > fileSize) {
                    throw malformed(
                            "section_metadata.objects["
                                    + index
                                    + "] range ["
                                    + begin
                                    + ", "
                                    + end
                                    + ") is outside the "
                                    + fileSize
                                    + "-byte package");
                }
                if (begin % SECTION_BLOCK_BYTES != 0L) {
                    throw malformed(
                            "section_metadata.objects["
                                    + index
                                    + "].begin_offset is not 16 KiB aligned");
                }
                if (previousEnd >= 0L
                        && begin < nextSectionBlock(previousEnd)) {
                    throw malformed(
                            "section_metadata.objects are overlapping or not block ordered");
                }
                previousEnd = end;
            }
        }

        private long nextSectionBlock(long previousEnd) throws IOException {
            long block = previousEnd / SECTION_BLOCK_BYTES;
            if (block >= Long.MAX_VALUE / SECTION_BLOCK_BYTES) {
                throw malformed("section offset overflows the supported file range");
            }
            return (block + 1L) * SECTION_BLOCK_BYTES;
        }

        private void validateKeyValueVector(
                Table owner, int fieldIndex, boolean required, String label)
                throws IOException {
            Integer field = owner.fieldLocation(
                    fieldIndex, Integer.BYTES, required, label);
            if (field == null) {
                return;
            }
            int vector = offsetTarget(field, label);
            int count = vectorLength(vector, Integer.BYTES, label);
            for (int index = 0; index < count; index++) {
                Table pair = tableAt(
                        vectorOffsetElement(vector, index, label + "[" + index + "]"),
                        label + "[" + index + "]");

                int keyField = pair.fieldLocation(
                        0,
                        Integer.BYTES,
                        true,
                        label + "[" + index + "].key");
                validateString(
                        offsetTarget(keyField, label + "[" + index + "].key"),
                        label + "[" + index + "].key");

                int typeField = pair.fieldLocation(
                        1,
                        Byte.BYTES,
                        true,
                        label + "[" + index + "].value_type");
                int valueType = Byte.toUnsignedInt(data.get(typeField));
                if (valueType == 0) {
                    throw malformed(label + "[" + index + "].value_type is NONE");
                }
                if (enforcePinnedEnumRanges
                        && valueType > PINNED_MAX_VALUE_UNION_TYPE) {
                    throw malformed(
                            label
                                    + "["
                                    + index
                                    + "].value_type "
                                    + valueType
                                    + " is outside the pinned 1."
                                    + PINNED_MINOR_VERSION
                                    + " schema");
                }

                int valueField = pair.fieldLocation(
                        2,
                        Integer.BYTES,
                        true,
                        label + "[" + index + "].value");
                Table value = tableAt(
                        offsetTarget(valueField, label + "[" + index + "].value"),
                        label + "[" + index + "].value");
                if (valueType == STRING_VALUE_UNION_TYPE) {
                    int stringField = value.fieldLocation(
                            0,
                            Integer.BYTES,
                            true,
                            label + "[" + index + "].value.string");
                    validateString(
                            offsetTarget(
                                    stringField,
                                    label + "[" + index + "].value.string"),
                            label + "[" + index + "].value.string");
                }
            }
        }

        private void validateString(int position, String label)
                throws IOException {
            range(position, Integer.BYTES, label + " length");
            long length = unsignedInt(position);
            long end = (long) position + Integer.BYTES + length;
            if (end >= data.limit()) {
                throw malformed(label + " escapes the metadata FlatBuffer");
            }
            if (data.get((int) end) != 0) {
                throw malformed(label + " is not null terminated");
            }
        }

        private int vectorLength(int position, int elementBytes, String label)
                throws IOException {
            range(position, Integer.BYTES, label + " length");
            long length = unsignedInt(position);
            long bytes = Integer.BYTES + length * elementBytes;
            range(position, bytes, label);
            if (length > Integer.MAX_VALUE) {
                throw malformed(label + " has too many elements");
            }
            return (int) length;
        }

        private int vectorOffsetElement(int vector, int index, String label)
                throws IOException {
            long location =
                    (long) vector + Integer.BYTES + (long) index * Integer.BYTES;
            range(location, Integer.BYTES, label);
            return offsetTarget((int) location, label);
        }

        private int offsetTarget(int location, String label)
                throws IOException {
            range(location, Integer.BYTES, label + " offset");
            long relative = unsignedInt(location);
            if (relative == 0L) {
                throw malformed(label + " offset is zero");
            }
            long target = (long) location + relative;
            range(target, Integer.BYTES, label + " target");
            return (int) target;
        }

        private Table tableAt(int position, String label) throws IOException {
            range(position, Integer.BYTES, label + " table");
            int vtableDistance = data.getInt(position);
            if (vtableDistance == 0) {
                throw malformed(label + " has a zero vtable offset");
            }
            long vtablePosition = (long) position - vtableDistance;
            range(vtablePosition, 4L, label + " vtable");
            int vtableLength =
                    Short.toUnsignedInt(data.getShort((int) vtablePosition));
            int objectLength =
                    Short.toUnsignedInt(data.getShort((int) vtablePosition + 2));
            if (vtableLength < 4
                    || (vtableLength & 1) != 0
                    || objectLength < Integer.BYTES) {
                throw malformed(label + " has invalid table lengths");
            }
            range(vtablePosition, vtableLength, label + " vtable");
            range(position, objectLength, label + " table");
            return new Table(
                    position,
                    (int) vtablePosition,
                    vtableLength,
                    objectLength,
                    label);
        }

        private long unsignedInt(int position) {
            return Integer.toUnsignedLong(data.getInt(position));
        }

        private void range(long position, long length, String label)
                throws IOException {
            if (position < 0L
                    || length < 0L
                    || position > data.limit()
                    || length > (long) data.limit() - position) {
                throw malformed(label + " is outside the metadata FlatBuffer");
            }
        }

        private IOException malformed(String reason) {
            return invalid(artifact, "malformed metadata: " + reason);
        }

        private final class Table {
            private final int position;
            private final int vtablePosition;
            private final int vtableLength;
            private final int objectLength;
            private final String label;

            private Table(
                    int position,
                    int vtablePosition,
                    int vtableLength,
                    int objectLength,
                    String label) {
                this.position = position;
                this.vtablePosition = vtablePosition;
                this.vtableLength = vtableLength;
                this.objectLength = objectLength;
                this.label = label;
            }

            private Integer fieldLocation(
                    int fieldIndex,
                    int width,
                    boolean required,
                    String fieldLabel)
                    throws IOException {
                long slot =
                        (long) vtablePosition
                                + 4L
                                + (long) fieldIndex * Short.BYTES;
                if (slot + Short.BYTES
                        > (long) vtablePosition + vtableLength) {
                    if (required) {
                        throw malformed(fieldLabel + " is missing");
                    }
                    return null;
                }
                int offset = Short.toUnsignedInt(data.getShort((int) slot));
                if (offset == 0) {
                    if (required) {
                        throw malformed(fieldLabel + " is missing");
                    }
                    return null;
                }
                if (offset < Integer.BYTES
                        || (long) offset + width > objectLength) {
                    throw malformed(fieldLabel + " has an invalid table offset");
                }
                int location = position + offset;
                range(location, width, fieldLabel);
                return location;
            }

            private long unsignedLong(int fieldIndex, String fieldLabel)
                    throws IOException {
                int location = fieldLocation(
                        fieldIndex,
                        Long.BYTES,
                        true,
                        fieldLabel);
                long value = data.getLong(location);
                if (value < 0L) {
                    throw malformed(
                            fieldLabel + " exceeds the supported signed file range");
                }
                return value;
            }

            private int unsignedByte(int fieldIndex, String fieldLabel)
                    throws IOException {
                int location = fieldLocation(
                        fieldIndex,
                        Byte.BYTES,
                        true,
                        fieldLabel);
                return Byte.toUnsignedInt(data.get(location));
            }
        }
    }
}
