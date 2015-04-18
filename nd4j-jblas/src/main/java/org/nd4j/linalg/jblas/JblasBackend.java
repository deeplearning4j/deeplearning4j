package org.nd4j.linalg.jblas;

import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

public class JblasBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JblasBackend.class);

    private final static String LINALG_PROPS = "/nd4j-jblas.properties";

    @Override
    public boolean isAvailable() {
        // execute JBLAS static initializer to confirm that the library is usable
        try {
            Class.forName("org.jblas.NativeBlas");
        } catch (Throwable e) {
            log.warn("unable to load Jblas backend", e);
            return false;
        }
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
