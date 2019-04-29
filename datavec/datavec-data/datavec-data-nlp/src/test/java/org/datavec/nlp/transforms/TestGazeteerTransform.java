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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TestGazeteerTransform {

    @Test
    public void testGazeteerTransform(){

        String[] corpus = {
                "hello I like apple".toLowerCase(),
                "cherry date eggplant potato".toLowerCase()
        };

        //Gazeteer transform: basically 0/1 if word is present. Assumes already tokenized input
        List<String> words = Arrays.asList("apple", "banana", "cherry", "date", "eggplant");

        GazeteerTransform t = new GazeteerTransform("words", "out", words);

        SequenceSchema schema = (SequenceSchema) new SequenceSchema.Builder()
                .addColumnString("words").build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .transform(t)
                .build();

        List<List<List<Writable>>> input = new ArrayList<>();
        for(String s : corpus){
            String[] split = s.split(" ");
            List<List<Writable>> seq = new ArrayList<>();
            for(String s2 : split){
                seq.add(Collections.<Writable>singletonList(new Text(s2)));
            }
            input.add(seq);
        }

        List<List<List<Writable>>> execute = LocalTransformExecutor.executeSequenceToSequence(input, tp);

        INDArray arr0 = ((NDArrayWritable)execute.get(0).get(0).get(0)).get();
        INDArray arr1 = ((NDArrayWritable)execute.get(0).get(1).get(0)).get();

        INDArray exp0 = Nd4j.create(new float[]{1, 0, 0, 0, 0});
        INDArray exp1 = Nd4j.create(new float[]{0, 0, 1, 1, 1});

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
