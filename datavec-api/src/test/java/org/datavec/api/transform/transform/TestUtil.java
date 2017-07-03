package org.datavec.api.transform.transform;

import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.datatype.joda.JodaModule;

/**
 * Some utilities for unit tests.
 *
 * @author dave@skymind.io
 */
public class TestUtil {

    /**
     * Clone of initMapper used for serialization and deserialization
     * of TransformProcess class, to support unit testing of
     * serialization and deserializtion of individual transforms.
     *
     * @param factory   JsonFactory
     * @return          ObjectMapper for serde of transforms
     */
    public static ObjectMapper initMapper(JsonFactory factory) {
        ObjectMapper om = new ObjectMapper(new JsonFactory());
        om.registerModule(new JodaModule());
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        return om;
    }
}
