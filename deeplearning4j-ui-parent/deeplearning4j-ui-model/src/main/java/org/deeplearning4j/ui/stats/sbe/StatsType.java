/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.StatsType"})
public enum StatsType {
    Parameters((short) 0), Gradients((short) 1), Updates((short) 2), Activations((short) 3), NULL_VAL((short) 255);

    private final short value;

    StatsType(final short value) {
        this.value = value;
    }

    public short value() {
        return value;
    }

    public static StatsType get(final short value) {
        switch (value) {
            case 0:
                return Parameters;
            case 1:
                return Gradients;
            case 2:
                return Updates;
            case 3:
                return Activations;
        }

        if ((short) 255 == value) {
            return NULL_VAL;
        }

        throw new IllegalArgumentException("Unknown value: " + value);
    }
}
