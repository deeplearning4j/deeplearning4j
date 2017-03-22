/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.DirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.InitFieldsPresentDecoder"})
@SuppressWarnings("all")
public class InitFieldsPresentDecoder {
    public static final int ENCODED_LENGTH = 1;
    private DirectBuffer buffer;
    private int offset;

    public InitFieldsPresentDecoder wrap(final DirectBuffer buffer, final int offset) {
        this.buffer = buffer;
        this.offset = offset;

        return this;
    }

    public int encodedLength() {
        return ENCODED_LENGTH;
    }

    public boolean softwareInfo() {
        return 0 != (buffer.getByte(offset) & (1 << 0));
    }

    public boolean hardwareInfo() {
        return 0 != (buffer.getByte(offset) & (1 << 1));
    }

    public boolean modelInfo() {
        return 0 != (buffer.getByte(offset) & (1 << 2));
    }

    public String toString() {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder) {
        builder.append('{');
        boolean atLeastOne = false;
        if (softwareInfo()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("softwareInfo");
            atLeastOne = true;
        }
        if (hardwareInfo()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("hardwareInfo");
            atLeastOne = true;
        }
        if (modelInfo()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("modelInfo");
            atLeastOne = true;
        }
        builder.append('}');

        return builder;
    }
}
