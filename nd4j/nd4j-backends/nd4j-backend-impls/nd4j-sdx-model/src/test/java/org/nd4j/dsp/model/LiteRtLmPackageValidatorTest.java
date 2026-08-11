/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Base64;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LiteRtLmPackageValidatorTest {
    private static final String OFFICIAL_FIXTURE_GIT_BLOB_SHA1 =
            "e383ab507c82e83e96e14a391e2b620556d3822c";

    @TempDir
    Path temporary;

    @Test
    void acceptsSupportedMajorAndFutureMinorPatchVersions() throws Exception {
        Path artifact = writePackage(
                "valid.litertlm", LiteRtLmTestPackage.create(1, 99, 42));
        LiteRtLmPackageValidator.validate(artifact);
    }

    @Test
    void acceptsPinnedOfficialLiteRtLmFixture() throws Exception {
        byte[] fixture = readOfficialFixture();
        assertEquals(32768, fixture.length);

        ByteBuffer fixed =
                ByteBuffer.wrap(fixture).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(1, fixed.getInt(8));
        assertEquals(2, fixed.getInt(12));
        assertEquals(0, fixed.getInt(16));
        assertEquals(240L, fixed.getLong(24));

        Path artifact = writePackage("official.litertlm", fixture);
        LiteRtLmPackageValidator.validate(artifact);
    }

    @Test
    void rejectsInvalidMagic() throws Exception {
        byte[] bytes = LiteRtLmTestPackage.create(1, 5, 0);
        bytes[0] = 'X';
        Path artifact = writePackage("bad-magic.litertlm", bytes);

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("magic"));
    }

    @Test
    void rejectsUnsupportedMajorVersion() throws Exception {
        Path artifact = writePackage(
                "future-major.litertlm", LiteRtLmTestPackage.create(2, 0, 0));

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("unsupported format version"));
    }

    @Test
    void rejectsBigEndianFixedHeader() throws Exception {
        byte[] bytes = LiteRtLmTestPackage.create(1, 5, 0);
        ByteBuffer bigEndian = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
        bigEndian.putInt(8, 1);
        bigEndian.putInt(12, 5);
        bigEndian.putInt(16, 0);
        bigEndian.putLong(24, 120L);
        Path artifact = writePackage("big-endian.litertlm", bytes);

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("unsupported format version"));
    }

    @Test
    void rejectsHeaderOffsetBeforeFixedPrefix() throws Exception {
        Path artifact = withHeaderEnd("short-offset.litertlm", 31L);

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("outside"));
    }

    @Test
    void rejectsHeaderOffsetPastEndOfFile() throws Exception {
        Path artifact = withHeaderEnd("long-offset.litertlm", 32768L);

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("outside"));
    }

    @Test
    void rejectsUnsignedHeaderOffsetOutsideSignedFileRange() throws Exception {
        Path artifact = withHeaderEnd("unsigned-offset.litertlm", -1L);

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("outside"));
    }

    @Test
    void rejectsTruncatedFixedHeader() throws Exception {
        Path artifact =
                writePackage("truncated.litertlm", new byte[31]);

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("fixed header is truncated"));
    }

    @Test
    void rejectsMissingFlatBufferMetadata() throws Exception {
        byte[] bytes = LiteRtLmTestPackage.create(1, 5, 0);
        ByteBuffer.wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(24, 32L);
        Path artifact = writePackage("missing-metadata.litertlm", bytes);

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("metadata FlatBuffer is missing"));
    }

    @Test
    void rejectsMalformedFlatBufferRoot() throws Exception {
        byte[] bytes = LiteRtLmTestPackage.create(1, 5, 0);
        ByteBuffer.wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putInt(32, 0);
        Path artifact = writePackage("bad-root.litertlm", bytes);

        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(failure.getMessage().contains("metadata root offset is zero"));
    }

    @Test
    void rejectsCorruptSyntheticFlatBufferStructures() throws Exception {
        byte[] badVtable = LiteRtLmTestPackage.create(1, 5, 0);
        littleEndian(badVtable).putShort(
                LiteRtLmTestPackage.ROOT_VTABLE_LENGTH_OFFSET, (short) 3);
        assertRejected("bad-vtable.litertlm", badVtable, "metadata root has invalid table lengths");

        byte[] badObject = LiteRtLmTestPackage.create(1, 5, 0);
        littleEndian(badObject).putShort(
                LiteRtLmTestPackage.ROOT_VTABLE_OBJECT_LENGTH_OFFSET, (short) 4);
        assertRejected("bad-object.litertlm", badObject, "has an invalid table offset");

        byte[] badVectorLength = LiteRtLmTestPackage.create(1, 5, 0);
        littleEndian(badVectorLength).putInt(
                LiteRtLmTestPackage.SECTION_VECTOR_LENGTH_OFFSET,
                Integer.MAX_VALUE);
        assertRejected("bad-vector-length.litertlm", badVectorLength, "outside");

        byte[] badVectorOffset = LiteRtLmTestPackage.create(1, 5, 0);
        littleEndian(badVectorOffset).putInt(
                LiteRtLmTestPackage.SECTION_VECTOR_ELEMENT_OFFSET, 0);
        assertRejected("bad-vector-offset.litertlm", badVectorOffset, "offset is zero");

        byte[] badString = LiteRtLmTestPackage.create(1, 5, 0);
        littleEndian(badString).putInt(
                LiteRtLmTestPackage.KEY_STRING_LENGTH_OFFSET, -1);
        assertRejected("bad-string.litertlm", badString, "escapes");
    }

    @Test
    void rejectsInvalidSectionRangesAndPinnedEnumTags() throws Exception {
        byte[] zeroLength = LiteRtLmTestPackage.create(1, 5, 0);
        ByteBuffer zeroLengthBuffer = littleEndian(zeroLength);
        zeroLengthBuffer.putLong(
                LiteRtLmTestPackage.SECTION_END_OFFSET,
                zeroLengthBuffer.getLong(LiteRtLmTestPackage.SECTION_BEGIN_OFFSET));
        assertRejected("zero-length.litertlm", zeroLength, "range");

        byte[] misaligned = LiteRtLmTestPackage.create(1, 5, 0);
        littleEndian(misaligned)
                .putLong(LiteRtLmTestPackage.SECTION_BEGIN_OFFSET, 185L)
                .putLong(LiteRtLmTestPackage.SECTION_END_OFFSET, 186L);
        assertRejected("misaligned.litertlm", misaligned, "not 16 KiB aligned");

        byte[] pastEnd = LiteRtLmTestPackage.create(1, 5, 0);
        littleEndian(pastEnd).putLong(
                LiteRtLmTestPackage.SECTION_END_OFFSET, pastEnd.length + 1L);
        assertRejected("section-past-end.litertlm", pastEnd, "outside");

        byte[] noneDataType = LiteRtLmTestPackage.create(1, 5, 0);
        noneDataType[LiteRtLmTestPackage.SECTION_DATA_TYPE_OFFSET] = 0;
        assertRejected("none-data-type.litertlm", noneDataType, "is NONE");

        byte[] unknownDataType = LiteRtLmTestPackage.create(1, 5, 0);
        unknownDataType[LiteRtLmTestPackage.SECTION_DATA_TYPE_OFFSET] = (byte) 255;
        assertRejected("unknown-data-type.litertlm", unknownDataType, "pinned 1.5 schema");

        byte[] noneValueType = LiteRtLmTestPackage.create(1, 5, 0);
        noneValueType[LiteRtLmTestPackage.VALUE_TYPE_OFFSET] = 0;
        assertRejected("none-value-type.litertlm", noneValueType, "is NONE");

        byte[] unknownValueType = LiteRtLmTestPackage.create(1, 5, 0);
        unknownValueType[LiteRtLmTestPackage.VALUE_TYPE_OFFSET] = (byte) 255;
        assertRejected("unknown-value-type.litertlm", unknownValueType, "pinned 1.5 schema");
    }

    @Test
    void permitsFutureMinorEnumExtensions() throws Exception {
        byte[] future = LiteRtLmTestPackage.create(1, 6, 0);
        future[LiteRtLmTestPackage.SECTION_DATA_TYPE_OFFSET] = (byte) 255;
        future[LiteRtLmTestPackage.VALUE_TYPE_OFFSET] = (byte) 255;

        LiteRtLmPackageValidator.validate(
                writePackage("future-enums.litertlm", future));
    }

    @Test
    void rejectsCorruptOfficialFixtureVtable() throws Exception {
        byte[] fixture = readOfficialFixture();
        ByteBuffer file = littleEndian(fixture);
        int root = 32 + file.getInt(32);
        int vtable = root - file.getInt(root);
        file.putShort(vtable, (short) 3);

        assertRejected("official-bad-vtable.litertlm", fixture, "metadata root has invalid table lengths");
    }

    @Test
    void rejectsSymlinkAndDirectoryArtifacts() throws Exception {
        Path target = writePackage(
                "target.litertlm", LiteRtLmTestPackage.create(1, 5, 0));
        Path symlink = temporary.resolve("symlink.litertlm");
        Files.createSymbolicLink(symlink, target.getFileName());

        assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(symlink));
        assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(temporary));
    }

    private void assertRejected(String name, byte[] bytes, String expected)
            throws IOException {
        Path artifact = writePackage(name, bytes);
        IOException failure = assertThrows(
                IOException.class,
                () -> LiteRtLmPackageValidator.validate(artifact));
        assertTrue(
                failure.getMessage().contains(expected),
                "Expected '" + expected + "' in: " + failure.getMessage());
    }

    private static ByteBuffer littleEndian(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    private Path withHeaderEnd(String name, long headerEndOffset)
            throws IOException {
        byte[] bytes = LiteRtLmTestPackage.create(1, 5, 0);
        ByteBuffer.wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(24, headerEndOffset);
        return writePackage(name, bytes);
    }

    private Path writePackage(String name, byte[] bytes) throws IOException {
        return Files.write(temporary.resolve(name), bytes);
    }

    private byte[] readOfficialFixture() throws Exception {
        try (InputStream input =
                getClass().getResourceAsStream(
                        "/litertlm/test_hf_tokenizer.litertlm.b64")) {
            assertNotNull(input);
            byte[] encoded = input.readAllBytes();
            byte[] fixture = Base64.getMimeDecoder().decode(encoded);

            MessageDigest sha1 = MessageDigest.getInstance("SHA-1");
            sha1.update(
                    ("blob " + fixture.length + "\0")
                            .getBytes(StandardCharsets.US_ASCII));
            sha1.update(fixture);
            assertEquals(
                    OFFICIAL_FIXTURE_GIT_BLOB_SHA1,
                    toHex(sha1.digest()));
            return fixture;
        }
    }

    private static String toHex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            result.append(String.format("%02x", Byte.toUnsignedInt(value)));
        }
        return result.toString();
    }
}
