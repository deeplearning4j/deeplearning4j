open module nd4j.cpu.api {
    requires commons.io;
    requires nd4j.common;
    requires slf4j.api;
    requires nd4j.api;
    requires org.bytedeco.javacpp;
    exports org.nd4j.autodiff.execution;
    exports org.nd4j.compression.impl;
    exports org.nd4j.nativeblas;
    exports org.nd4j.rng;
    exports org.nd4j.rng.deallocator;
    exports org.nd4j.storage;
    provides org.nd4j.linalg.compression.NDArrayCompressor with org.nd4j.compression.impl.Gzip, org.nd4j.compression.impl.NoOp;
    provides org.nd4j.systeminfo.GPUInfoProvider with org.nd4j.nativeblas.NativeOpsGPUInfoProvider;
}
