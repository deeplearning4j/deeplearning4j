/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.MutableDirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.MessageHeaderEncoder"})
@SuppressWarnings("all")
public class MessageHeaderEncoder {
    public static final int ENCODED_LENGTH = 8;
    private MutableDirectBuffer buffer;
    private int offset;

    public MessageHeaderEncoder wrap(final MutableDirectBuffer buffer, final int offset) {
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

    public MessageHeaderEncoder blockLength(final int value) {
        buffer.putShort(offset + 0, (short) value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static int templateIdNullValue() {
        return 65535;
    }

    public static int templateIdMinValue() {
        return 0;
    }

    public static int templateIdMaxValue() {
        return 65534;
    }

    public MessageHeaderEncoder templateId(final int value) {
        buffer.putShort(offset + 2, (short) value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static int schemaIdNullValue() {
        return 65535;
    }

    public static int schemaIdMinValue() {
        return 0;
    }

    public static int schemaIdMaxValue() {
        return 65534;
    }

    public MessageHeaderEncoder schemaId(final int value) {
        buffer.putShort(offset + 4, (short) value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static int versionNullValue() {
        return 65535;
    }

    public static int versionMinValue() {
        return 0;
    }

    public static int versionMaxValue() {
        return 65534;
    }

    public MessageHeaderEncoder version(final int value) {
        buffer.putShort(offset + 6, (short) value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

}
