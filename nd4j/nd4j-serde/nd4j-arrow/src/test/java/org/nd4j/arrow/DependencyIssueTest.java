package org.nd4j.arrow;

import com.google.flatbuffers.FlatBufferBuilder;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.dictionary.DictionaryProvider;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.*;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Method;
import java.nio.channels.WritableByteChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * See: https://github.com/deeplearning4j/deeplearning4j/issues/6372
 */
public class DependencyIssueTest {

    @BeforeClass
    public static void before(){
        Class<?> c = FlatBufferBuilder.class;
        ClassLoader cl = ArrowSerdeTest.class.getClassLoader();
        System.out.println("FlatBufferBuilder location: " + cl.getResource("com/google/flatbuffers/FlatBufferBuilder.class"));
        Method[] methods = c.getDeclaredMethods();
        System.out.println("FlatBufferBuilder Methods:");
        for(Method m : methods){
            Class<?>[] paramTypes = m.getParameterTypes();
            System.out.print("  - " + m.getName() + "(");
            boolean first = true;
            for(Class<?> p : paramTypes){
                if(!first){
                    System.out.print(",");
                }
                System.out.print(p.getSimpleName());
                first = false;
            }
            System.out.println(")");
        }
    }

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void test() throws Exception  {

        BufferAllocator bufferAllocator = new RootAllocator(Long.MAX_VALUE);


        List<Field> fields = Collections.singletonList(
                new Field("test", FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)), new ArrayList<Field>()));

        Schema s = new Schema(fields);

        Float4Vector float4Vector = new Float4Vector("test",bufferAllocator);
        float4Vector.allocateNew(10);
        float4Vector.setValueCount(10);
        for( int i=0; i<10; i++ )
            float4Vector.set(i, i);

        List<FieldVector> vectors = Collections.<FieldVector>singletonList(float4Vector);

        VectorSchemaRoot root = new VectorSchemaRoot(s,vectors,10);

        DictionaryProvider p = new DictionaryProvider.MapDictionaryProvider(new Dictionary(float4Vector, new DictionaryEncoding(0,true,null)));

        File outFile = testDir.newFile();
        try(FileOutputStream fos = new FileOutputStream(outFile)){
            WritableByteChannel channel = fos.getChannel();
            ArrowFileWriter writer = new ArrowFileWriter(root, p, channel);
            writer.start();
            writer.writeBatch();
            writer.end();
        }
    }

}
