package org.nd4j.nativeblas;

import lombok.Getter;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * @author raver119@gmail.com
 * @author saudet
 */
public class NativeOpsHolder {
    private static Logger log = LoggerFactory.getLogger(NativeOpsHolder.class);
    private static final NativeOpsHolder INSTANCE = new NativeOpsHolder();

    @Getter
    private final NativeOps deviceNativeOps;

    private NativeOpsHolder() {
        try {
            Properties props = Nd4jContext.getInstance().getConf();

            String name = System.getProperty(Nd4j.NATIVE_OPS, props.get(Nd4j.NATIVE_OPS).toString());
            Class<? extends NativeOps> nativeOpsClazz = Class.forName(name).asSubclass(NativeOps.class);
            deviceNativeOps = nativeOpsClazz.newInstance();

            PointerPointer functions = new PointerPointer(15);
            functions.put(0, Loader.addressof("cblas_sgemv"));
            functions.put(1, Loader.addressof("cblas_dgemv"));
            functions.put(2, Loader.addressof("cblas_sgemm"));
            functions.put(3, Loader.addressof("cblas_dgemm"));
            functions.put(4, Loader.addressof("cblas_sgemm_batch"));
            functions.put(5, Loader.addressof("cblas_dgemm_batch"));

            functions.put(6, Loader.addressof("cublasSgemv_v2"));
            functions.put(7, Loader.addressof("cublasDgemv_v2"));
            functions.put(8, Loader.addressof("cublasHgemm"));
            functions.put(9, Loader.addressof("cublasSgemm_v2"));
            functions.put(10, Loader.addressof("cublasDgemm_v2"));
            functions.put(11, Loader.addressof("cublasSgemmEx"));
            functions.put(12, Loader.addressof("cublasHgemmBatched"));
            functions.put(13, Loader.addressof("cublasSgemmBatched"));
            functions.put(14, Loader.addressof("cublasDgemmBatched"));
            deviceNativeOps.initializeDevicesAndFunctions(functions);

            int numThreads;
            String numThreadsString = System.getenv("OMP_NUM_THREADS");
            if (numThreadsString != null && !numThreadsString.isEmpty()) {
                numThreads = Integer.parseInt(numThreadsString);
                deviceNativeOps.setOmpNumThreads(numThreads);
            } else {
                int cores = Loader.totalCores();
                int chips = Loader.totalChips();
                if (chips > 0 && cores > 0) {
                    deviceNativeOps.setOmpNumThreads(Math.max(1, cores / chips));
                } else
                    deviceNativeOps.setOmpNumThreads(
                                    deviceNativeOps.getCores(Runtime.getRuntime().availableProcessors()));
            }
            //deviceNativeOps.setOmpNumThreads(4);

            log.info("Number of threads used for NativeOps: {}", deviceNativeOps.ompGetMaxThreads());
        } catch (Exception | Error e) {
            throw new RuntimeException(
                            "ND4J is probably missing dependencies. For more information, please refer to: http://nd4j.org/getstarted.html",
                            e);
        }
    }

    public static NativeOpsHolder getInstance() {
        return INSTANCE;
    }

}
