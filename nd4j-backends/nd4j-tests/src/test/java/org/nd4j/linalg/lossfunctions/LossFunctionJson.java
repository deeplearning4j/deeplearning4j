package org.nd4j.linalg.lossfunctions;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.impl.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 09/09/2016.
 */
public class LossFunctionJson extends BaseNd4jTest {

    public LossFunctionJson(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testJsonSerialization() throws Exception {

        ILossFunction[] lossFns = new ILossFunction[]{
                new LossBinaryXENT(),
                new LossHinge(),
                new LossKLD(),
                new LossMAE(),
                new LossMAPE(),
                new LossMCXENT(),
                new LossMSE(),
                new LossMSLE()
        };

        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        for(ILossFunction lf : lossFns){
            String asJson = mapper.writeValueAsString(lf);
//            System.out.println(asJson);

            ILossFunction fromJson = mapper.readValue(asJson, ILossFunction.class);
            assertEquals(lf, fromJson);
        }
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
