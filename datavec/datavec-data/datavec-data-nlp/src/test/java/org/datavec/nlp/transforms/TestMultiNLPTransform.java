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

package org.datavec.nlp.transforms;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class TestMultiNLPTransform {

    @Test
    public void test(){

        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "eggplant");
        GazeteerTransform t1 = new GazeteerTransform("words", "out", words);
        GazeteerTransform t2 = new GazeteerTransform("out", "out", words);


        MultiNlpTransform multi = new MultiNlpTransform("text", new BagOfWordsTransform[]{t1, t2}, "out");

        String[] corpus = {
                "hello I like apple".toLowerCase(),
                "date eggplant potato".toLowerCase()
        };

        List<List<List<Writable>>> input = new ArrayList<>();
        for(String s : corpus){
            String[] split = s.split(" ");
            List<List<Writable>> seq = new ArrayList<>();
            for(String s2 : split){
                seq.add(Collections.<Writable>singletonList(new Text(s2)));
            }
            input.add(seq);
        }

        SequenceSchema schema = (SequenceSchema) new SequenceSchema.Builder()
                .addColumnString("text").build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .transform(multi)
                .build();

        List<List<List<Writable>>> execute = LocalTransformExecutor.executeSequenceToSequence(input, tp);

        INDArray arr0 = ((NDArrayWritable)execute.get(0).get(0).get(0)).get();
        INDArray arr1 = ((NDArrayWritable)execute.get(0).get(1).get(0)).get();

        INDArray exp0 = Nd4j.create(new float[]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0});
        INDArray exp1 = Nd4j.create(new float[]{0, 0, 0, 1, 1, 0, 0, 0, 1, 1});

        assertEquals(exp0, arr0);
        assertEquals(exp1, arr1);


        String json = tp.toJson();
        TransformProcess tp2 = TransformProcess.fromJson(json);
        assertEquals(tp, tp2);

        List<List<List<Writable>>> execute2 = LocalTransformExecutor.executeSequenceToSequence(input, tp);
        INDArray arr0a = ((NDArrayWritable)execute2.get(0).get(0).get(0)).get();
        INDArray arr1a = ((NDArrayWritable)execute2.get(0).get(1).get(0)).get();

        assertEquals(exp0, arr0a);
        assertEquals(exp1, arr1a);

    }

}
