/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.transform.transform.ndarray;

import org.datavec.api.transform.MathFunction;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.ndarray.NDArrayColumnsMathOpTransform;
import org.datavec.api.transform.ndarray.NDArrayMathFunctionTransform;
import org.datavec.api.transform.ndarray.NDArrayScalarOpTransform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.serde.JsonSerializer;
import org.datavec.api.transform.serde.YamlSerializer;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 20/07/2016.
 */
public class TestYamlJsonSerde {

    public static YamlSerializer y = new YamlSerializer();
    public static JsonSerializer j = new JsonSerializer();

    @Test
    public void testTransforms() {


        Transform[] transforms =
                        new Transform[] {new NDArrayColumnsMathOpTransform("newCol", MathOp.Divide, "in1", "in2"),
                                        new NDArrayMathFunctionTransform("inCol", MathFunction.SQRT),
                                        new NDArrayScalarOpTransform("inCol", MathOp.ScalarMax, 3.0)};

        for (Transform t : transforms) {
            String yaml = y.serialize(t);
            String json = j.serialize(t);

            //            System.out.println(yaml);
            //            System.out.println(json);
            //            System.out.println();

            Transform t2 = y.deserializeTransform(yaml);
            Transform t3 = j.deserializeTransform(json);
            assertEquals(t, t2);
            assertEquals(t, t3);
        }


        String tArrAsYaml = y.serialize(transforms);
        String tArrAsJson = j.serialize(transforms);
        String tListAsYaml = y.serializeTransformList(Arrays.asList(transforms));
        String tListAsJson = j.serializeTransformList(Arrays.asList(transforms));

        //        System.out.println("\n\n\n\n");
        //        System.out.println(tListAsYaml);

        List<Transform> lFromYaml = y.deserializeTransformList(tListAsYaml);
        List<Transform> lFromJson = j.deserializeTransformList(tListAsJson);

        assertEquals(Arrays.asList(transforms), y.deserializeTransformList(tArrAsYaml));
        assertEquals(Arrays.asList(transforms), j.deserializeTransformList(tArrAsJson));
        assertEquals(Arrays.asList(transforms), lFromYaml);
        assertEquals(Arrays.asList(transforms), lFromJson);
    }

    @Test
    public void testTransformProcessAndSchema() {

        Schema schema = new Schema.Builder().addColumnInteger("firstCol").addColumnNDArray("nd1a", new long[] {1, 10})
                        .addColumnNDArray("nd1b", new long[] {1, 10}).addColumnNDArray("nd2", new long[] {1, 100})
                        .addColumnNDArray("nd3", new long[] {-1, -1}).build();

        TransformProcess tp = new TransformProcess.Builder(schema).integerMathOp("firstCol", MathOp.Add, 1)
                        .ndArrayColumnsMathOpTransform("added", MathOp.Add, "nd1a", "nd1b")
                        .ndArrayMathFunctionTransform("nd2", MathFunction.SQRT)
                        .ndArrayScalarOpTransform("nd3", MathOp.Multiply, 2.0).build();

        String asJson = tp.toJson();
        String asYaml = tp.toYaml();

        TransformProcess fromJson = TransformProcess.fromJson(asJson);
        TransformProcess fromYaml = TransformProcess.fromYaml(asYaml);

        assertEquals(tp, fromJson);
        assertEquals(tp, fromYaml);
    }

}
