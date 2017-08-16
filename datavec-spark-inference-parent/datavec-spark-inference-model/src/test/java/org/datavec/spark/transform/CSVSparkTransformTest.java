package org.datavec.spark.transform;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.nlp.TextToCharacterIndexTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchCSVRecord;
import org.datavec.spark.transform.model.SequenceBatchCSVRecord;
import org.datavec.spark.transform.model.SingleCSVRecord;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 12/24/16.
 */
public class CSVSparkTransformTest {
    @Test
    public void testTransformer() throws Exception {
        List<Writable> input = new ArrayList<>();
        input.add(new DoubleWritable(1.0));
        input.add(new DoubleWritable(2.0));

        Schema schema = new Schema.Builder().addColumnDouble("1.0").addColumnDouble("2.0").build();
        List<Writable> output = new ArrayList<>();
        output.add(new Text("1.0"));
        output.add(new Text("2.0"));

        TransformProcess transformProcess =
                        new TransformProcess.Builder(schema).convertToString("1.0").convertToString("2.0").build();
        CSVSparkTransform csvSparkTransform = new CSVSparkTransform(transformProcess);
        String[] values = new String[] {"1.0", "2.0"};
        SingleCSVRecord record = csvSparkTransform.transform(new SingleCSVRecord(values));
        Base64NDArrayBody body = csvSparkTransform.toArray(new SingleCSVRecord(values));
        INDArray fromBase64 = Nd4jBase64.fromBase64(body.getNdarray());
        assertTrue(fromBase64.isVector());
        System.out.println("Base 64ed array " + fromBase64);
    }

    @Test
    public void testTransformerBatch() throws Exception {
        List<Writable> input = new ArrayList<>();
        input.add(new DoubleWritable(1.0));
        input.add(new DoubleWritable(2.0));

        Schema schema = new Schema.Builder().addColumnDouble("1.0").addColumnDouble("2.0").build();
        List<Writable> output = new ArrayList<>();
        output.add(new Text("1.0"));
        output.add(new Text("2.0"));

        TransformProcess transformProcess =
                        new TransformProcess.Builder(schema).convertToString("1.0").convertToString("2.0").build();
        CSVSparkTransform csvSparkTransform = new CSVSparkTransform(transformProcess);
        String[] values = new String[] {"1.0", "2.0"};
        SingleCSVRecord record = csvSparkTransform.transform(new SingleCSVRecord(values));
        BatchCSVRecord batchCSVRecord = new BatchCSVRecord();
        for (int i = 0; i < 3; i++)
            batchCSVRecord.add(record);
        BatchCSVRecord batchCSVRecord1 = csvSparkTransform.transform(batchCSVRecord);
        Base64NDArrayBody body = csvSparkTransform.toArray(batchCSVRecord1);
        INDArray fromBase64 = Nd4jBase64.fromBase64(body.getNdarray());
        assertTrue(fromBase64.isMatrix());
        System.out.println("Base 64ed array " + fromBase64);
    }



    @Test
    public void testSingleBatchSequence() throws Exception {
        Schema schema = new Schema.Builder()
                .addColumnsString("action")
                .build();

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeAllColumnsExceptFor("action")
                .convertToSequence()
                .transform(new TextToCharacterIndexTransform("action", "action_sequence", defaultCharIndex(), true))
                .integerToOneHot("action_sequence",0,29)
                .build();

        CSVSparkTransform csvSparkTransform = new CSVSparkTransform(transformProcess);
        String[] values = new String[] {"install new part"};
        BatchCSVRecord batchCSVRecord = new BatchCSVRecord(Arrays.asList(new SingleCSVRecord(values)));
        SequenceBatchCSVRecord input = new SequenceBatchCSVRecord();
        input.add(Arrays.asList(batchCSVRecord));
        SequenceBatchCSVRecord record = csvSparkTransform.transformSequence(input);
        System.out.print(record);
//        batchCSVRecord = new BatchCSVRecord();
////        for (int i = 0; i < 3; i++)
////            batchCSVRecord.add(record);
//        BatchCSVRecord batchCSVRecord1 = csvSparkTransform.transformSequenceIncremental(batchCSVRecord);
//        SequenceBatchCSVRecord sequenceBatchCSVRecord = new SequenceBatchCSVRecord();
//        sequenceBatchCSVRecord.add(Arrays.asList(batchCSVRecord));
//        Base64NDArrayBody sequenceArray = csvSparkTransform.transformSequenceArray(sequenceBatchCSVRecord);
//
//
//        Base64NDArrayBody body = csvSparkTransform.toArray(batchCSVRecord1);
//        INDArray fromBase64 = Nd4jBase64.fromBase64(body.getNdarray());
//        assertTrue(fromBase64.isMatrix());
//        assertArrayEquals(new int[]{1,3,2},Nd4jBase64.fromBase64(sequenceArray.getNdarray()).shape());
//        //ensure accumulation
//        sequenceBatchCSVRecord.add(Arrays.asList(batchCSVRecord));
//        sequenceArray = csvSparkTransform.transformSequenceArray(sequenceBatchCSVRecord);
//        assertArrayEquals(new int[]{2,3,2},Nd4jBase64.fromBase64(sequenceArray.getNdarray()).shape());
//
//        SequenceBatchCSVRecord transformed = csvSparkTransform.transformSequence(sequenceBatchCSVRecord);
//        assumeNotNull(transformed.getRecords());
//        System.out.println(transformed);


    }

    public static Map<Character,Integer> defaultCharIndex() {
        Map<Character,Integer> ret = new TreeMap<>();

        ret.put('a',0);
        ret.put('b',1);
        ret.put('c',2);
        ret.put('d',3);
        ret.put('e',4);
        ret.put('f',5);
        ret.put('g',6);
        ret.put('h',7);
        ret.put('i',8);
        ret.put('j',9);
        ret.put('k',10);
        ret.put('l',11);
        ret.put('m',12);
        ret.put('n',13);
        ret.put('o',14);
        ret.put('p',15);
        ret.put('q',16);
        ret.put('r',17);
        ret.put('s',18);
        ret.put('t',19);
        ret.put('u',20);
        ret.put('v',21);
        ret.put('w',22);
        ret.put('x',23);
        ret.put('y',24);
        ret.put('z',25);
        ret.put('/',26);
        ret.put(' ',27);
        ret.put('(',28);
        ret.put(')',29);

        return ret;
    }

}
