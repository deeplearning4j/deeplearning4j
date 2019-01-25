package org.nd4j.jackson.objectmapper.holder;

import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * A simple object mapper holder for
 * using one single {@link ObjectMapper}
 * across the whole project.
 *
 */
public class ObjectMapperHolder {

    private static ObjectMapper objectMapper = getMapper();

    private ObjectMapperHolder() {}


    /**
     * Get a single object mapper for use
     * with reading and writing json
     * @return
     */
    public static ObjectMapper getJsonMapper() {
        return objectMapper;
    }

    private static ObjectMapper getMapper() {
        ObjectMapper om = new ObjectMapper();
        //Serialize fields only, not using getters
        //Not all getters are supported - for example, UserEntity
        om.setVisibilityChecker(om.getSerializationConfig()
                .getDefaultVisibilityChecker()
                .withFieldVisibility(JsonAutoDetect.Visibility.ANY)
                .withGetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withSetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withCreatorVisibility(JsonAutoDetect.Visibility.NONE));
        om.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        return om;
    }



}
