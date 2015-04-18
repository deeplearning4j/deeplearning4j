package org.nd4j.linalg.jcublas;

import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

public class JCublasBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JCublasBackend.class);

    private final static String LINALG_PROPS = "/nd4j-jcublas.properties";

    @Override
    public boolean isAvailable() {
        // execute SimpleJCublas static initializer to confirm that the library is usable
        try {
            Class.forName("org.nd4j.linalg.jcublas.SimpleJCublas");
        } catch (Throwable e) {
            log.warn("unable to load JCublas backend", e);
            return false;
        }
        return true;
    }

    @Override
    public int getPriority() {
        return BACKEND_PRIORITY_GPU;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS);
    }
}
