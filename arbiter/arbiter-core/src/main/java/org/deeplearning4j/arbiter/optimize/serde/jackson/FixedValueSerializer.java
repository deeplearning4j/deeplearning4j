package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.apache.commons.net.util.Base64;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.core.type.WritableTypeId;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;
import org.nd4j.shade.jackson.databind.jsontype.TypeSerializer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import static org.nd4j.shade.jackson.core.JsonToken.START_OBJECT;

/**
 * A custom serializer to handle arbitrary object types
 * Uses standard JSON where safe (number, string, enumerations) or Java object serialization (bytes -> base64)
 * The latter is not an ideal approach, but Jackson doesn't support serialization/deserialization of arbitrary
 * objects very well
 *
 * @author Alex Black
 */
public class FixedValueSerializer extends JsonSerializer<FixedValue> {
    @Override
    public void serialize(FixedValue fixedValue, JsonGenerator j, SerializerProvider serializerProvider) throws IOException {
        Object o = fixedValue.getValue();

        j.writeStringField("@valueclass", o.getClass().getName());
        if(o instanceof Number || o instanceof String || o instanceof Enum){
            j.writeObjectField("value", o);
        } else {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(o);
            baos.close();
            byte[] b = baos.toByteArray();
            String base64 = new Base64().encodeToString(b);
            j.writeStringField("data", base64);
        }
    }

    @Override
    public void serializeWithType(FixedValue value, JsonGenerator gen, SerializerProvider serializers, TypeSerializer typeSer) throws IOException {
        WritableTypeId typeId = typeSer.typeId(value, START_OBJECT);
        typeSer.writeTypePrefix(gen, typeId);
        serialize(value, gen, serializers);
        typeSer.writeTypeSuffix(gen, typeId);
    }
}
