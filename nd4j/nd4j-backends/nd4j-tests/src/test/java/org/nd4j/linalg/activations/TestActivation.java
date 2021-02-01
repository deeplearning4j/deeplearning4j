/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.activations;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.activations.impl.ActivationCube;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationGELU;
import org.nd4j.linalg.activations.impl.ActivationHardSigmoid;
import org.nd4j.linalg.activations.impl.ActivationHardTanH;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationRReLU;
import org.nd4j.linalg.activations.impl.ActivationRationalTanh;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftPlus;
import org.nd4j.linalg.activations.impl.ActivationSoftSign;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.primitives.Pair;
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
public class TestActivation extends BaseNd4jTest {

    public TestActivation(Nd4jBackend backend) {
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
    public void testRelu(){

        Double[] max = {null, 6.0, 2.5, 5.0};
        Double[] threshold = {0.0, 0.0, 0.75, 0.2};
        Double[] negativeSlope = {0.0, 0.0, 0.0, 0.3};

        INDArray in = Nd4j.linspace(-10, 10, 1000, DataType.DOUBLE);
        double[] dIn = in.data().asDouble();

        for( int i=0; i<max.length; i++ ){
            ActivationReLU r = new ActivationReLU(max[i], threshold[i], negativeSlope[i]);
            INDArray out = r.getActivation(in.dup(), true);
            double[] exp = new double[dIn.length];
            for( int j=0; j<exp.length; j++ ){
                if(max[i] != null && dIn[j] >= max[i]){
                    exp[j] = max[i];
                } else if(dIn[j] < threshold[i]){
                    exp[j] = negativeSlope[i] * (dIn[j] - threshold[i]);
                } else {
                    exp[j] = Math.min(dIn[j], max[i] == null ? Double.MAX_VALUE : max[i]);
                }
            }
            INDArray expArr = Nd4j.createFromArray(exp);
            assertEquals(expArr, out);
        }

        //Test backprop
        INDArray eps = Nd4j.arange(in.length()).castTo(DataType.DOUBLE);
        double[] dEps = eps.data().asDouble();
        for( int i=0; i<max.length; i++ ){
            ActivationReLU r = new ActivationReLU(max[i], threshold[i], negativeSlope[i]);
            Pair<INDArray,INDArray> p = r.backprop(in.dup(), eps.dup());
            INDArray grad = p.getFirst();
            double[] dGrad = grad.data().asDouble();

            for( int j=0; j<dGrad.length; j++ ){
                if(max[i] != null && dIn[j] >= max[i]){
                    //Max segment - gradient at input should be zero
                    assertEquals(0.0, dGrad[j], 0.0);
                } else if(dIn[j] < threshold[i]){
                    //Below threshold - gradient equal to dL/dOut * threshold
                    double exp = dEps[j] * negativeSlope[i];
                    assertEquals(exp, dGrad[j], 1e-6);
                } else {
                    //Linear part
                    assertEquals(dEps[j], dGrad[j], 1e-8);
                }
            }
        }
    }

    @Test
    public void testJson() throws Exception {

        IActivation[] activations = new IActivation[] {new ActivationCube(), new ActivationELU(0.25),
                        new ActivationHardSigmoid(), new ActivationHardTanH(), new ActivationIdentity(),
                        new ActivationLReLU(0.25), new ActivationRationalTanh(), new ActivationReLU(),
                        new ActivationRReLU(0.25, 0.5), new ActivationSigmoid(), new ActivationSoftmax(),
                        new ActivationSoftPlus(), new ActivationSoftSign(), new ActivationTanH(), new ActivationGELU(), new ActivationGELU(true)};

        String[][] expectedFields = new String[][] {{"@class"}, //Cube
                        {"@class", "alpha"}, //ELU
                        {"@class"}, //Hard sigmoid
                        {"@class"}, //Hard TanH
                        {"@class"}, //Identity
                        {"@class", "alpha"}, //Leaky Relu
                        {"@class"}, //rational tanh
                        {"@class", "max", "negativeSlope", "threshold"}, //relu
                        {"@class", "l", "u"}, //rrelu
                        {"@class"}, //sigmoid
                        {"@class"}, //Softmax
                        {"@class"}, //Softplus
                        {"@class"}, //Softsign
                        {"@class"}, //Tanh
                        {"@class", "precise"}, //GELU
                        {"@class", "precise"}  //GELU precise

        };

        for (int i = 0; i < activations.length; i++) {
            String asJson = mapper.writeValueAsString(activations[i]);

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
