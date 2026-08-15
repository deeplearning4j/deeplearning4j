/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** Creates a minimal structurally valid LiteRT-LM package for unit tests. */
final class LiteRtLmTestPackage {
    private static final int METADATA_START = 32;
    private static final int METADATA_BYTES = 152;
    private static final int FIRST_SECTION = 16 * 1024;
    private static final int FILE_BYTES = FIRST_SECTION + 1;

    static final int ROOT_VTABLE_LENGTH_OFFSET = METADATA_START + 4;
    static final int ROOT_VTABLE_OBJECT_LENGTH_OFFSET = METADATA_START + 6;
    static final int SECTION_VECTOR_LENGTH_OFFSET = METADATA_START + 36;
    static final int SECTION_VECTOR_ELEMENT_OFFSET = METADATA_START + 40;
    static final int SECTION_BEGIN_OFFSET = METADATA_START + 64;
    static final int SECTION_END_OFFSET = METADATA_START + 72;
    static final int SECTION_DATA_TYPE_OFFSET = METADATA_START + 80;
    static final int VALUE_TYPE_OFFSET = METADATA_START + 116;
    static final int KEY_STRING_LENGTH_OFFSET = METADATA_START + 140;

    private LiteRtLmTestPackage() {}

    static byte[] create(int major, int minor, int patch) {
        ByteBuffer file =
                ByteBuffer.allocate(FILE_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        file.put("LITERTLM".getBytes(StandardCharsets.US_ASCII));
        file.putInt(major);
        file.putInt(minor);
        file.putInt(patch);
        file.putInt(0);
        file.putLong(METADATA_START + METADATA_BYTES);

        int base = METADATA_START;

        // LiteRTLMMetaData root: only section_metadata is present.
        file.putInt(base, 12);
        putVtable(file, base + 4, 8, 8, 0, 4);
        file.putInt(base + 12, 8);
        file.putInt(base + 16, 12);

        // SectionMetadata with one objects-vector field.
        putVtable(file, base + 20, 6, 8, 4);
        file.putInt(base + 28, 8);
        file.putInt(base + 32, 4);

        // Vector containing one SectionObject table offset.
        file.putInt(base + 36, 1);
        file.putInt(base + 40, 16);

        // SectionObject with one metadata item and an aligned one-byte TFLite section.
        putVtable(file, base + 44, 12, 32, 4, 8, 16, 24);
        file.putInt(base + 56, 12);
        file.putInt(base + 60, 28);
        file.putLong(base + 64, FIRST_SECTION);
        file.putLong(base + 72, FIRST_SECTION + 1L);
        file.put(base + 80, (byte) 3);

        // items vector containing one KeyValuePair table offset.
        file.putInt(base + 88, 1);
        file.putInt(base + 92, 16);

        // KeyValuePair key="k", value_type=UInt8, value=UInt8{7}.
        putVtable(file, base + 96, 10, 16, 4, 8, 12);
        file.putInt(base + 108, 12);
        file.putInt(base + 112, 28);
        file.put(base + 116, (byte) 1);
        file.putInt(base + 120, 12);

        putVtable(file, base + 124, 6, 8, 4);
        file.putInt(base + 132, 8);
        file.put(base + 136, (byte) 7);

        file.putInt(base + 140, 1);
        file.put(base + 144, (byte) 'k');
        file.put(base + 145, (byte) 0);
        file.put(FIRST_SECTION, (byte) 1);

        return file.array();
    }

    private static void putVtable(
            ByteBuffer buffer,
            int position,
            int length,
            int objectLength,
            int... fieldOffsets) {
        buffer.putShort(position, (short) length);
        buffer.putShort(position + 2, (short) objectLength);
        for (int index = 0; index < fieldOffsets.length; index++) {
            buffer.putShort(
                    position + 4 + index * Short.BYTES,
                    (short) fieldOffsets[index]);
        }
    }
}
