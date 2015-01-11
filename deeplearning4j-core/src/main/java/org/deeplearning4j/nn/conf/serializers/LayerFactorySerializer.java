package org.deeplearning4j.nn.conf.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.deeplearning4j.nn.api.LayerFactory;

import java.io.IOException;

/**
 * Writes a field of:
 * layerFactory: layer factory class name, value.layerClazzName()
 *
 * @author Adam Gibson
 */
public class LayerFactorySerializer extends JsonSerializer<LayerFactory> {
    @Override
    public void serialize(LayerFactory value, JsonGenerator jgen, SerializerProvider provider) throws IOException {
        String write = value.getClass().getName() + "," + value.layerClazzName();
        jgen.writeStringField("layerFactory",write);

    }
}
