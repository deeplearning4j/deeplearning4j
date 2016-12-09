package org.nd4j.nativeblas;

import lombok.Getter;

/**
 * @author raver119@gmail.com
 */
public class NativeOpsHolder {
    private static final NativeOpsHolder INSTANCE = new NativeOpsHolder();

    @Getter private final NativeOps deviceNativeOps;

    private NativeOpsHolder() {
        deviceNativeOps = new NativeOps();
    }

    public static NativeOpsHolder getInstance() {
        return INSTANCE;
    }

}
