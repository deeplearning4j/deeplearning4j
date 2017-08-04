/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.DirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.UpdateFieldsPresentDecoder"})
@SuppressWarnings("all")
public class UpdateFieldsPresentDecoder {
    public static final int ENCODED_LENGTH = 4;
    private DirectBuffer buffer;
    private int offset;

    public UpdateFieldsPresentDecoder wrap(final DirectBuffer buffer, final int offset) {
        this.buffer = buffer;
        this.offset = offset;

        return this;
    }

    public int encodedLength() {
        return ENCODED_LENGTH;
    }

    public boolean score() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 0));
    }

    public boolean memoryUse() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 1));
    }

    public boolean performance() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 2));
    }

    public boolean garbageCollection() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 3));
    }

    public boolean histogramParameters() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 4));
    }

    public boolean histogramGradients() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 5));
    }

    public boolean histogramUpdates() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 6));
    }

    public boolean histogramActivations() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 7));
    }

    public boolean meanParameters() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 8));
    }

    public boolean meanGradients() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 9));
    }

    public boolean meanUpdates() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 10));
    }

    public boolean meanActivations() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 11));
    }

    public boolean stdevParameters() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 12));
    }

    public boolean stdevGradients() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 13));
    }

    public boolean stdevUpdates() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 14));
    }

    public boolean stdevActivations() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 15));
    }

    public boolean meanMagnitudeParameters() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 16));
    }

    public boolean meanMagnitudeGradients() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 17));
    }

    public boolean meanMagnitudeUpdates() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 18));
    }

    public boolean meanMagnitudeActivations() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 19));
    }

    public boolean learningRatesPresent() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 20));
    }

    public boolean dataSetMetaDataPresent() {
        return 0 != (buffer.getInt(offset, java.nio.ByteOrder.LITTLE_ENDIAN) & (1 << 21));
    }

    public String toString() {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder) {
        builder.append('{');
        boolean atLeastOne = false;
        if (score()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("score");
            atLeastOne = true;
        }
        if (memoryUse()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("memoryUse");
            atLeastOne = true;
        }
        if (performance()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("performance");
            atLeastOne = true;
        }
        if (garbageCollection()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("garbageCollection");
            atLeastOne = true;
        }
        if (histogramParameters()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("histogramParameters");
            atLeastOne = true;
        }
        if (histogramGradients()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("histogramGradients");
            atLeastOne = true;
        }
        if (histogramUpdates()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("histogramUpdates");
            atLeastOne = true;
        }
        if (histogramActivations()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("histogramActivations");
            atLeastOne = true;
        }
        if (meanParameters()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("meanParameters");
            atLeastOne = true;
        }
        if (meanGradients()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("meanGradients");
            atLeastOne = true;
        }
        if (meanUpdates()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("meanUpdates");
            atLeastOne = true;
        }
        if (meanActivations()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("meanActivations");
            atLeastOne = true;
        }
        if (stdevParameters()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("stdevParameters");
            atLeastOne = true;
        }
        if (stdevGradients()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("stdevGradients");
            atLeastOne = true;
        }
        if (stdevUpdates()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("stdevUpdates");
            atLeastOne = true;
        }
        if (stdevActivations()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("stdevActivations");
            atLeastOne = true;
        }
        if (meanMagnitudeParameters()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("meanMagnitudeParameters");
            atLeastOne = true;
        }
        if (meanMagnitudeGradients()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("meanMagnitudeGradients");
            atLeastOne = true;
        }
        if (meanMagnitudeUpdates()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("meanMagnitudeUpdates");
            atLeastOne = true;
        }
        if (meanMagnitudeActivations()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("meanMagnitudeActivations");
            atLeastOne = true;
        }
        if (learningRatesPresent()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("learningRatesPresent");
            atLeastOne = true;
        }
        if (dataSetMetaDataPresent()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("dataSetMetaDataPresent");
            atLeastOne = true;
        }
        builder.append('}');

        return builder;
    }
}
