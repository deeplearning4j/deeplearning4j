package org.nd4j.nativeblas;

import lombok.Getter;

/**
 * @author raver119@gmail.com
 */
public class NativeOpsHolder {
    private static final NativeOpsHolder INSTANCE = new NativeOpsHolder();

    @Getter private final NativeOps deviceNativeOps;

    @Getter private final Nd4jBlas deviceNativeBlas;

    private NativeOpsHolder() {
        deviceNativeOps = new NativeOps();
        deviceNativeBlas = new Nd4jBlas();
    }

    public static NativeOpsHolder getInstance() {
        return INSTANCE;
    }

}
