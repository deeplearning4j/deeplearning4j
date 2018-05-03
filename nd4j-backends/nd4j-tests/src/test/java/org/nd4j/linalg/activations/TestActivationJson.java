package org.nd4j.linalg.activations;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.shade.jackson.databind.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 30/12/2016.
 */
@RunWith(Parameterized.class)
public class TestActivationJson extends BaseNd4jTest {

    public TestActivationJson(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    private ObjectMapper mapper;

    @Before
    public void initMapper() {
        mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
    }

    @Test
    public void testJson() throws Exception {

        IActivation[] activations = new IActivation[] {new ActivationCube(), new ActivationELU(0.25),
                        new ActivationHardSigmoid(), new ActivationHardTanH(), new ActivationIdentity(),
                        new ActivationLReLU(0.25), new ActivationRationalTanh(), new ActivationReLU(),
                        new ActivationRReLU(0.25, 0.5), new ActivationSigmoid(), new ActivationSoftmax(),
                        new ActivationSoftPlus(), new ActivationSoftSign(), new ActivationTanH()};

        String[][] expectedFields = new String[][] {{"@class"}, //Cube
                        {"@class", "alpha"}, //ELU
                        {"@class"}, //Hard sigmoid
                        {"@class"}, //Hard TanH
                        {"@class"}, //Identity
                        {"@class", "alpha"}, //Leaky Relu
                        {"@class"}, //rational tanh
                        {"@class"}, //relu
                        {"@class", "l", "u"}, //rrelu
                        {"@class"}, //sigmoid
                        {"@class"}, //Softmax
                        {"@class"}, //Softplus
                        {"@class"}, //Softsign
                        {"@class"} //Tanh

        };

        for (int i = 0; i < activations.length; i++) {
            String asJson = mapper.writeValueAsString(activations[i]);
            System.out.println(asJson);

            JsonNode node = mapper.readTree(asJson);

            Iterator<String> fieldNamesIter = node.fieldNames();
            List<String> actualFieldsByName = new ArrayList<>();
            while (fieldNamesIter.hasNext()) {
                actualFieldsByName.add(fieldNamesIter.next());
            }

            String[] expFields = expectedFields[i];

            String msg = activations[i].toString() + "\tExpected fields: " + Arrays.toString(expFields)
                            + "\tActual fields: " + actualFieldsByName;
            assertEquals(msg, expFields.length, actualFieldsByName.size());

            for (String s : expFields) {
                msg = "Expected field \"" + s + "\", was not found in " + activations[i].toString();
                assertTrue(msg, actualFieldsByName.contains(s));
            }

            //Test conversion from JSON:
            IActivation act = mapper.readValue(asJson, IActivation.class);
            assertEquals(activations[i], act);
        }
    }
}
