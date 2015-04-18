package org.nd4j.linalg.netlib;

import org.nd4j.linalg.factory.Nd4jBackend;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

public class NetlibBlasBackend extends Nd4jBackend {

    private final static String LINALG_PROPS = "/nd4j-netlib.properties";

    @Override
    public boolean isAvailable() {
        // netlib has built-in fallback behavior
        return true;
    }

    @Override
    public int getPriority() {
        return BACKEND_PRIORITY_CPU;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS);
    }
}
