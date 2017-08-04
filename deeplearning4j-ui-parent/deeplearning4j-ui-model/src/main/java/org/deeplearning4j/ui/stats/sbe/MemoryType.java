/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.MemoryType"})
public enum MemoryType {
    JvmCurrent((short) 0), JvmMax((short) 1), OffHeapCurrent((short) 2), OffHeapMax((short) 3), DeviceCurrent(
                    (short) 4), DeviceMax((short) 5), NULL_VAL((short) 255);

    private final short value;

    MemoryType(final short value) {
        this.value = value;
    }

    public short value() {
        return value;
    }

    public static MemoryType get(final short value) {
        switch (value) {
            case 0:
                return JvmCurrent;
            case 1:
                return JvmMax;
            case 2:
                return OffHeapCurrent;
            case 3:
                return OffHeapMax;
            case 4:
                return DeviceCurrent;
            case 5:
                return DeviceMax;
        }

        if ((short) 255 == value) {
            return NULL_VAL;
        }

        throw new IllegalArgumentException("Unknown value: " + value);
    }
}
