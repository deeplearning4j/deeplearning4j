/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.MutableDirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.UpdateFieldsPresentEncoder"})
@SuppressWarnings("all")
public class UpdateFieldsPresentEncoder {
    public static final int ENCODED_LENGTH = 4;
    private MutableDirectBuffer buffer;
    private int offset;

    public UpdateFieldsPresentEncoder wrap(final MutableDirectBuffer buffer, final int offset) {
        this.buffer = buffer;
        this.offset = offset;

        return this;
    }

    public int encodedLength() {
        return ENCODED_LENGTH;
    }

    public UpdateFieldsPresentEncoder clear() {
        buffer.putInt(offset, (int) 0L, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder score(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 0) : bits & ~(1 << 0);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder memoryUse(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 1) : bits & ~(1 << 1);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder performance(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 2) : bits & ~(1 << 2);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder garbageCollection(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 3) : bits & ~(1 << 3);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder histogramParameters(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 4) : bits & ~(1 << 4);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder histogramGradients(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 5) : bits & ~(1 << 5);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder histogramUpdates(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 6) : bits & ~(1 << 6);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder histogramActivations(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 7) : bits & ~(1 << 7);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder meanParameters(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 8) : bits & ~(1 << 8);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder meanGradients(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 9) : bits & ~(1 << 9);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder meanUpdates(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 10) : bits & ~(1 << 10);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder meanActivations(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 11) : bits & ~(1 << 11);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder stdevParameters(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 12) : bits & ~(1 << 12);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder stdevGradients(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 13) : bits & ~(1 << 13);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder stdevUpdates(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 14) : bits & ~(1 << 14);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder stdevActivations(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 15) : bits & ~(1 << 15);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder meanMagnitudeParameters(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 16) : bits & ~(1 << 16);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder meanMagnitudeGradients(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 17) : bits & ~(1 << 17);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder meanMagnitudeUpdates(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 18) : bits & ~(1 << 18);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder meanMagnitudeActivations(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 19) : bits & ~(1 << 19);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder learningRatesPresent(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 20) : bits & ~(1 << 20);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }

    public UpdateFieldsPresentEncoder dataSetMetaDataPresent(final boolean value) {
        int bits = buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN);
        bits = value ? bits | (1 << 21) : bits & ~(1 << 21);
        buffer.putInt(offset, bits, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }
}
