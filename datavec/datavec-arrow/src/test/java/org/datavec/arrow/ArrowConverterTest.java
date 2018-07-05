package org.datavec.arrow;

import lombok.val;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import org.datavec.arrow.recordreader.ArrowRecordReader;
import org.datavec.arrow.recordreader.ArrowWritableRecordBatch;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;

import static java.nio.channels.Channels.newChannel;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class ArrowConverterTest {

    private static BufferAllocator bufferAllocator = new RootAllocator(Long.MAX_VALUE);

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();




    @Test
    public void testArrowColumnINDArray() {
        Schema.Builder schema = new Schema.Builder();
        List<String> single = new ArrayList<>();
        int numCols = 2;
        INDArray arr = Nd4j.linspace(1,4,4);
        for(int i = 0; i < numCols; i++) {
            schema.addColumnNDArray(String.valueOf(i),new long[]{1,4});
            single.add(String.valueOf(i));
        }

        Schema buildSchema = schema.build();
        List<List<Writable>> list = new ArrayList<>();
        List<Writable> firstRow = new ArrayList<>();
        for(int i = 0 ; i < numCols; i++) {
            firstRow.add(new NDArrayWritable(arr));
        }

        list.add(firstRow);

        List<FieldVector> fieldVectors = ArrowConverter.toArrowColumns(bufferAllocator, buildSchema, list);
        assertEquals(numCols,fieldVectors.size());
        assertEquals(1,fieldVectors.get(0).getValueCount());
        assertFalse(fieldVectors.get(0).isNull(0));

        ArrowWritableRecordBatch arrowWritableRecordBatch = ArrowConverter.toArrowWritables(fieldVectors, buildSchema);
        assertEquals(1,arrowWritableRecordBatch.size());

        Writable writable = arrowWritableRecordBatch.get(0).get(0);
        assertTrue(writable instanceof NDArrayWritable);
        NDArrayWritable ndArrayWritable = (NDArrayWritable) writable;
        assertEquals(arr,ndArrayWritable.get());

        Writable writable1 = ArrowConverter.fromEntry(0, fieldVectors.get(0), ColumnType.NDArray);
        NDArrayWritable ndArrayWritablewritable1 = (NDArrayWritable) writable1;
        System.out.println(ndArrayWritablewritable1.get());

    }

    @Test
    public void testArrowColumnString() {
        Schema.Builder schema = new Schema.Builder();
        List<String> single = new ArrayList<>();
        for(int i = 0; i < 2; i++) {
            schema.addColumnInteger(String.valueOf(i));
            single.add(String.valueOf(i));
        }


        List<FieldVector> fieldVectors = ArrowConverter.toArrowColumnsStringSingle(bufferAllocator, schema.build(), single);
        List<List<Writable>> records = ArrowConverter.toArrowWritables(fieldVectors, schema.build());
        List<List<Writable>> assertion = new ArrayList<>();
        assertion.add(Arrays.<Writable>asList(new IntWritable(0),new IntWritable(1)));
        assertEquals(assertion,records);

        List<List<String>> batch = new ArrayList<>();
        for(int i = 0; i < 2; i++) {
            batch.add(Arrays.asList(String.valueOf(i),String.valueOf(i)));
        }

        List<FieldVector> fieldVectorsBatch = ArrowConverter.toArrowColumnsString(bufferAllocator, schema.build(), batch);
        List<List<Writable>> batchRecords = ArrowConverter.toArrowWritables(fieldVectorsBatch, schema.build());

        List<List<Writable>> assertionBatch = new ArrayList<>();
        assertionBatch.add(Arrays.<Writable>asList(new IntWritable(0),new IntWritable(0)));
        assertionBatch.add(Arrays.<Writable>asList(new IntWritable(1),new IntWritable(1)));
        assertEquals(assertionBatch,batchRecords);


    }



    @Test
    public void testArrowBatchSetTime() {
        Schema.Builder schema = new Schema.Builder();
        List<String> single = new ArrayList<>();
        for(int i = 0; i < 2; i++) {
            schema.addColumnTime(String.valueOf(i),TimeZone.getDefault());
            single.add(String.valueOf(i));
        }

        List<List<Writable>> input = Arrays.asList(
                Arrays.<Writable>asList(new LongWritable(0),new LongWritable(1)),
                Arrays.<Writable>asList(new LongWritable(2),new LongWritable(3))
        );

        List<FieldVector> fieldVector = ArrowConverter.toArrowColumns(bufferAllocator,schema.build(),input);
        ArrowWritableRecordBatch writableRecordBatch = new ArrowWritableRecordBatch(fieldVector,schema.build());
        List<Writable> assertion = Arrays.<Writable>asList(new LongWritable(4), new LongWritable(5));
        writableRecordBatch.set(1, Arrays.<Writable>asList(new LongWritable(4),new LongWritable(5)));
        List<Writable> recordTest = writableRecordBatch.get(1);
        assertEquals(assertion,recordTest);
    }

    @Test
    public void testArrowBatchSet() {
        Schema.Builder schema = new Schema.Builder();
        List<String> single = new ArrayList<>();
        for(int i = 0; i < 2; i++) {
            schema.addColumnInteger(String.valueOf(i));
            single.add(String.valueOf(i));
        }

        List<List<Writable>> input = Arrays.asList(
                Arrays.<Writable>asList(new IntWritable(0),new IntWritable(1)),
                Arrays.<Writable>asList(new IntWritable(2),new IntWritable(3))
        );

        List<FieldVector> fieldVector = ArrowConverter.toArrowColumns(bufferAllocator,schema.build(),input);
        ArrowWritableRecordBatch writableRecordBatch = new ArrowWritableRecordBatch(fieldVector,schema.build());
        List<Writable> assertion = Arrays.<Writable>asList(new IntWritable(4), new IntWritable(5));
        writableRecordBatch.set(1, Arrays.<Writable>asList(new IntWritable(4),new IntWritable(5)));
        List<Writable> recordTest = writableRecordBatch.get(1);
        assertEquals(assertion,recordTest);
    }

    @Test
    public void testArrowColumnsStringTimeSeries() {
        Schema.Builder schema = new Schema.Builder();
        List<List<List<String>>> entries = new ArrayList<>();
        for(int i = 0; i < 3; i++) {
            schema.addColumnInteger(String.valueOf(i));
        }

        for(int i = 0; i < 5; i++) {
            List<List<String>> arr = Arrays.asList(Arrays.asList(String.valueOf(i), String.valueOf(i), String.valueOf(i)));
            entries.add(arr);
        }

        List<FieldVector> fieldVectors = ArrowConverter.toArrowColumnsStringTimeSeries(bufferAllocator, schema.build(), entries);
        assertEquals(3,fieldVectors.size());
        assertEquals(5,fieldVectors.get(0).getValueCount());


        INDArray exp = Nd4j.create(5, 3);
        for( int i = 0; i < 5; i++) {
            exp.getRow(i).assign(i);
        }
        //Convert to ArrowWritableRecordBatch - note we can't do this in general with time series...
        ArrowWritableRecordBatch wri = ArrowConverter.toArrowWritables(fieldVectors, schema.build());
        INDArray arr = ArrowConverter.toArray(wri);
        assertArrayEquals(new long[] {5,3}, arr.shape());


        assertEquals(exp, arr);
    }

    @Test
    public void testConvertVector() {
        Schema.Builder schema = new Schema.Builder();
        List<List<List<String>>> entries = new ArrayList<>();
        for(int i = 0; i < 3; i++) {
            schema.addColumnInteger(String.valueOf(i));
        }

        for(int i = 0; i < 5; i++) {
            List<List<String>> arr = Arrays.asList(Arrays.asList(String.valueOf(i), String.valueOf(i), String.valueOf(i)));
            entries.add(arr);
        }

        List<FieldVector> fieldVectors = ArrowConverter.toArrowColumnsStringTimeSeries(bufferAllocator, schema.build(), entries);
        INDArray arr = ArrowConverter.convertArrowVector(fieldVectors.get(0),schema.build().getType(0));
        assertEquals(5,arr.length());
    }

    @Test
    public void testCreateNDArray() throws Exception {
        val recordsToWrite = recordToWrite();
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ArrowConverter.writeRecordBatchTo(recordsToWrite.getRight(),recordsToWrite.getFirst(),byteArrayOutputStream);

        File f = testDir.newFolder();

        File tmpFile = new File(f, "tmp-arrow-file-" + UUID.randomUUID().toString() + ".arrorw");
        FileOutputStream outputStream = new FileOutputStream(tmpFile);
        tmpFile.deleteOnExit();
        ArrowConverter.writeRecordBatchTo(recordsToWrite.getRight(),recordsToWrite.getFirst(),outputStream);
        outputStream.flush();
        outputStream.close();

        Pair<Schema, ArrowWritableRecordBatch> schemaArrowWritableRecordBatchPair = ArrowConverter.readFromFile(tmpFile);
        assertEquals(recordsToWrite.getFirst(),schemaArrowWritableRecordBatchPair.getFirst());
        assertEquals(recordsToWrite.getRight(),schemaArrowWritableRecordBatchPair.getRight().toArrayList());

        byte[] arr = byteArrayOutputStream.toByteArray();
        val read = ArrowConverter.readFromBytes(arr);
        assertEquals(recordsToWrite,read);

        //send file
        File tmp =  tmpDataFile(recordsToWrite);
        ArrowRecordReader recordReader = new ArrowRecordReader();

        recordReader.initialize(new FileSplit(tmp));

        recordReader.next();
        ArrowWritableRecordBatch currentBatch = recordReader.getCurrentBatch();
        INDArray arr2 = ArrowConverter.toArray(currentBatch);
        assertEquals(2,arr2.rows());
        assertEquals(2,arr2.columns());
    }


    @Test
    public void testConvertToArrowVectors() {
        INDArray matrix = Nd4j.linspace(1,4,4).reshape(2,2);
        val vectors = ArrowConverter.convertToArrowVector(matrix,Arrays.asList("test","test2"), ColumnType.Double,bufferAllocator);
        assertEquals(matrix.rows(),vectors.size());

        INDArray vector = Nd4j.linspace(1,4,4);
        val vectors2 = ArrowConverter.convertToArrowVector(vector,Arrays.asList("test"), ColumnType.Double,bufferAllocator);
        assertEquals(1,vectors2.size());
        assertEquals(matrix.length(),vectors2.get(0).getValueCount());

    }

    @Test
    public void testSchemaConversionBasic() {
        Schema.Builder schemaBuilder = new Schema.Builder();
        for(int i = 0; i < 2; i++) {
            schemaBuilder.addColumnDouble("test-" + i);
            schemaBuilder.addColumnInteger("testi-" + i);
            schemaBuilder.addColumnLong("testl-" + i);
            schemaBuilder.addColumnFloat("testf-" + i);
        }


        Schema schema = schemaBuilder.build();
        val schema2 = ArrowConverter.toArrowSchema(schema);
        assertEquals(8,schema2.getFields().size());
        val convertedSchema = ArrowConverter.toDatavecSchema(schema2);
        assertEquals(schema,convertedSchema);
    }

    @Test
    public void testReadSchemaAndRecordsFromByteArray() throws Exception {
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);

        int valueCount = 3;
        List<Field> fields = new ArrayList<>();
        fields.add(ArrowConverter.field("field1",new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)));
        fields.add(ArrowConverter.intField("field2"));

        List<FieldVector> fieldVectors = new ArrayList<>();
        fieldVectors.add(ArrowConverter.vectorFor(allocator,"field1",new float[] {1,2,3}));
        fieldVectors.add(ArrowConverter.vectorFor(allocator,"field2",new int[] {1,2,3}));


        org.apache.arrow.vector.types.pojo.Schema schema = new org.apache.arrow.vector.types.pojo.Schema(fields);

        VectorSchemaRoot schemaRoot1 = new VectorSchemaRoot(schema, fieldVectors, valueCount);
        VectorUnloader vectorUnloader = new VectorUnloader(schemaRoot1);
        vectorUnloader.getRecordBatch();
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        try(ArrowFileWriter arrowFileWriter = new ArrowFileWriter(schemaRoot1,null,newChannel(byteArrayOutputStream))) {
            arrowFileWriter.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }

        byte[] arr = byteArrayOutputStream.toByteArray();
        val arr2 = ArrowConverter.readFromBytes(arr);
        assertEquals(2,arr2.getFirst().numColumns());
        assertEquals(3,arr2.getRight().size());

        val arrowCols = ArrowConverter.toArrowColumns(allocator,arr2.getFirst(),arr2.getRight());
        assertEquals(2,arrowCols.size());
        assertEquals(valueCount,arrowCols.get(0).getValueCount());
    }


    @Test
    public void testVectorForEdgeCases() {
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        val vector = ArrowConverter.vectorFor(allocator,"field1",new float[]{Float.MIN_VALUE,Float.MAX_VALUE});
        assertEquals(Float.MIN_VALUE,vector.get(0),1e-2);
        assertEquals(Float.MAX_VALUE,vector.get(1),1e-2);

        val vectorInt = ArrowConverter.vectorFor(allocator,"field1",new int[]{Integer.MIN_VALUE,Integer.MAX_VALUE});
        assertEquals(Integer.MIN_VALUE,vectorInt.get(0),1e-2);
        assertEquals(Integer.MAX_VALUE,vectorInt.get(1),1e-2);

    }

    @Test
    public void testVectorFor() {
        BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);

        val vector = ArrowConverter.vectorFor(allocator,"field1",new float[]{1,2,3});
        assertEquals(3,vector.getValueCount());
        assertEquals(1,vector.get(0),1e-2);
        assertEquals(2,vector.get(1),1e-2);
        assertEquals(3,vector.get(2),1e-2);

        val vectorLong = ArrowConverter.vectorFor(allocator,"field1",new long[]{1,2,3});
        assertEquals(3,vectorLong.getValueCount());
        assertEquals(1,vectorLong.get(0),1e-2);
        assertEquals(2,vectorLong.get(1),1e-2);
        assertEquals(3,vectorLong.get(2),1e-2);


        val vectorInt = ArrowConverter.vectorFor(allocator,"field1",new int[]{1,2,3});
        assertEquals(3,vectorInt.getValueCount());
        assertEquals(1,vectorInt.get(0),1e-2);
        assertEquals(2,vectorInt.get(1),1e-2);
        assertEquals(3,vectorInt.get(2),1e-2);

        val vectorDouble = ArrowConverter.vectorFor(allocator,"field1",new double[]{1,2,3});
        assertEquals(3,vectorDouble.getValueCount());
        assertEquals(1,vectorDouble.get(0),1e-2);
        assertEquals(2,vectorDouble.get(1),1e-2);
        assertEquals(3,vectorDouble.get(2),1e-2);


        val vectorBool = ArrowConverter.vectorFor(allocator,"field1",new boolean[]{true,true,false});
        assertEquals(3,vectorBool.getValueCount());
        assertEquals(1,vectorBool.get(0),1e-2);
        assertEquals(1,vectorBool.get(1),1e-2);
        assertEquals(0,vectorBool.get(2),1e-2);
    }

    @Test
    public void testRecordReaderAndWriteFile() throws Exception {
        val recordsToWrite = recordToWrite();
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ArrowConverter.writeRecordBatchTo(recordsToWrite.getRight(),recordsToWrite.getFirst(),byteArrayOutputStream);
        byte[] arr = byteArrayOutputStream.toByteArray();
        val read = ArrowConverter.readFromBytes(arr);
        assertEquals(recordsToWrite,read);

        //send file
        File tmp =  tmpDataFile(recordsToWrite);
        RecordReader recordReader = new ArrowRecordReader();

        recordReader.initialize(new FileSplit(tmp));

        List<Writable> record = recordReader.next();
        assertEquals(2,record.size());

    }

    @Test
    public void testRecordReaderMetaDataList() throws Exception {
        val recordsToWrite = recordToWrite();
        //send file
        File tmp =  tmpDataFile(recordsToWrite);
        RecordReader recordReader = new ArrowRecordReader();
        RecordMetaDataIndex recordMetaDataIndex = new RecordMetaDataIndex(0,tmp.toURI(),ArrowRecordReader.class);
        recordReader.loadFromMetaData(Arrays.<RecordMetaData>asList(recordMetaDataIndex));

        Record record = recordReader.nextRecord();
        assertEquals(2,record.getRecord().size());

    }

    @Test
    public void testDates() {
        Date now = new Date();
        BufferAllocator bufferAllocator = new RootAllocator(Long.MAX_VALUE);
        TimeStampMilliVector timeStampMilliVector = ArrowConverter.vectorFor(bufferAllocator, "col1", new Date[]{now});
        assertEquals(now.getTime(),timeStampMilliVector.get(0));
    }


    @Test
    public void testRecordReaderMetaData() throws Exception {
        val recordsToWrite = recordToWrite();
        //send file
        File tmp =  tmpDataFile(recordsToWrite);
        RecordReader recordReader = new ArrowRecordReader();
        RecordMetaDataIndex recordMetaDataIndex = new RecordMetaDataIndex(0,tmp.toURI(),ArrowRecordReader.class);
        recordReader.loadFromMetaData(recordMetaDataIndex);

        Record record = recordReader.nextRecord();
        assertEquals(2,record.getRecord().size());
    }

    private File tmpDataFile(Pair<Schema,List<List<Writable>>> recordsToWrite) throws IOException {

        File f = testDir.newFolder();

        //send file
        File tmp = new File(f,"tmp-file-" + UUID.randomUUID().toString());
        tmp.mkdirs();
        File tmpFile = new File(tmp,"data.arrow");
        tmpFile.deleteOnExit();
        FileOutputStream bufferedOutputStream = new FileOutputStream(tmpFile);
        ArrowConverter.writeRecordBatchTo(recordsToWrite.getRight(),recordsToWrite.getFirst(),bufferedOutputStream);
        bufferedOutputStream.flush();
        bufferedOutputStream.close();
        return tmp;
    }

    private Pair<Schema,List<List<Writable>>> recordToWrite() {
        List<List<Writable>> records = new ArrayList<>();
        records.add(Arrays.<Writable>asList(new DoubleWritable(0.0),new DoubleWritable(0.0)));
        records.add(Arrays.<Writable>asList(new DoubleWritable(0.0),new DoubleWritable(0.0)));
        Schema.Builder schemaBuilder = new Schema.Builder();
        for(int i = 0; i < 2; i++) {
            schemaBuilder.addColumnFloat("col-" + i);
        }

        return Pair.of(schemaBuilder.build(),records);
    }




}
