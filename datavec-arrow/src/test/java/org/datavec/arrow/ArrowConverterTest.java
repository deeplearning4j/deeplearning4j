package org.datavec.arrow;

import lombok.val;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.datavec.api.transform.schema.Schema;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.nio.channels.Channels.newChannel;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class ArrowConverterTest {

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
    public void testReadSchemaAndRecordsFromByteArray() {
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


}
