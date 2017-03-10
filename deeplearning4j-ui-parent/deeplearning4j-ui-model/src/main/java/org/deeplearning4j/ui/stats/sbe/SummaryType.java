/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.SummaryType"})
public enum SummaryType {
    Mean((short) 0), Stdev((short) 1), MeanMagnitude((short) 2), NULL_VAL((short) 255);

    private final short value;

    SummaryType(final short value) {
        this.value = value;
    }

    public short value() {
        return value;
    }

    public static SummaryType get(final short value) {
        switch (value) {
            case 0:
                return Mean;
            case 1:
                return Stdev;
            case 2:
                return MeanMagnitude;
        }

        if ((short) 255 == value) {
            return NULL_VAL;
        }

        throw new IllegalArgumentException("Unknown value: " + value);
    }
}
