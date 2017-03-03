/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

@javax.annotation.Generated(value = {"StatSource"})
public enum StatSource {
    Parameters((short) 0), Updates((short) 1), Activations((short) 2), NULL_VAL((short) 255);

    private final short value;

    StatSource(final short value) {
        this.value = value;
    }

    public short value() {
        return value;
    }

    public static StatSource get(final short value) {
        switch (value) {
            case 0:
                return Parameters;
            case 1:
                return Updates;
            case 2:
                return Activations;
        }

        if ((short) 255 == value) {
            return NULL_VAL;
        }

        throw new IllegalArgumentException("Unknown value: " + value);
    }
}
