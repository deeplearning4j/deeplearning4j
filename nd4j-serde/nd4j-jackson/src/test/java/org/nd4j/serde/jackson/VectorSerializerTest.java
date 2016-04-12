package org.nd4j.serde.jackson;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class VectorSerializerTest {
    @Test
    @Ignore
    public void testVectorSerializer() throws Exception {
        ObjectMapper mapper = getMapper();
        INDArray ones = Nd4j.ones(5);
        String json = mapper.writeValueAsString(ones);
        INDArray getBack = mapper.readValue(json, INDArray.class);
        assertEquals(ones,getBack);
        IComplexNDArray arr2 = Nd4j.complexOnes(5);
        json = mapper.writeValueAsString(arr2);
        getBack = mapper.readValue(json, INDArray.class);
        assertEquals(arr2,getBack);


    }

    public ObjectMapper getMapper() {
        ObjectMapper mapper = new ObjectMapper();
        SimpleModule nd4j = new SimpleModule("nd4j");
        nd4j.addDeserializer(INDArray.class, new VectorDeSerializer());
        nd4j.addSerializer(INDArray.class, new VectorSerializer());
        mapper.registerModule(nd4j);
        return mapper;
    }


}
