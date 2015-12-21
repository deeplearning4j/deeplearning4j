package org.arbiter.optimize.ui.resources;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.ws.rs.ext.ContextResolver;

public class ObjectMapperProvider implements ContextResolver<ObjectMapper> {
    @Override
    public ObjectMapper getContext(Class<?> type) {
        final ObjectMapper result = new ObjectMapper();
        result.registerModule(module());
        return result;
    }

    public static SimpleModule module() {
        return new SimpleModule("module");
    }
}
