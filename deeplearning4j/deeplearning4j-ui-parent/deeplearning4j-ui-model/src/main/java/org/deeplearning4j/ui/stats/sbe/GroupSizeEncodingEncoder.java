/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.MutableDirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.GroupSizeEncodingEncoder"})
@SuppressWarnings("all")
public class GroupSizeEncodingEncoder {
    public static final int ENCODED_LENGTH = 4;
    private MutableDirectBuffer buffer;
    private int offset;

    public GroupSizeEncodingEncoder wrap(final MutableDirectBuffer buffer, final int offset) {
        this.buffer = buffer;
        this.offset = offset;

        return this;
    }

    public int encodedLength() {
        return ENCODED_LENGTH;
    }

    public static int blockLengthNullValue() {
        return 65535;
    }

    public static int blockLengthMinValue() {
        return 0;
    }

    public static int blockLengthMaxValue() {
        return 65534;
    }

    public GroupSizeEncodingEncoder blockLength(final int value) {
        buffer.putShort(offset + 0, (short) value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static int numInGroupNullValue() {
        return 65535;
    }

    public static int numInGroupMinValue() {
        return 0;
    }

    public static int numInGroupMaxValue() {
        return 65534;
    }

    public GroupSizeEncodingEncoder numInGroup(final int value) {
        buffer.putShort(offset + 2, (short) value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public String toString() {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder) {
        GroupSizeEncodingDecoder writer = new GroupSizeEncodingDecoder();
        writer.wrap(buffer, offset);

        return writer.appendTo(builder);
    }
}
