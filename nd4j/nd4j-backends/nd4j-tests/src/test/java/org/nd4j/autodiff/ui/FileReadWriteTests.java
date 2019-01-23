package org.nd4j.autodiff.ui;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import lombok.AllArgsConstructor;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.UIGraphStructure;
import org.nd4j.graph.UIInfoType;
import org.nd4j.graph.UIStaticInfoRecord;
import org.nd4j.graph.UISystemInfo;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class FileReadWriteTests {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Before
    public void before() {
        Nd4j.create(1);
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @Test
    public void testSimple() throws IOException {

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("variable", DataType.DOUBLE, 3,4);
        SDVariable sum = v.sum();

//        File f = testDir.newFile();
        File f = new File("C:/Temp/flatbuffers.fb");
        if(f.exists())
            f.delete();
        System.out.println(f.getAbsolutePath());
        LogFileWriter w = new LogFileWriter(f);
        long bytesWritten = w.writeGraphStructure(sd);

        System.out.println("Bytes written: " + bytesWritten);
        assertTrue(bytesWritten > 0);

        Pair<UIStaticInfoRecord, Table> read = w.readStatic();
        System.out.println(read);

        UIGraphStructure s = (UIGraphStructure) read.getSecond();
        List<String> l = new ArrayList<>(s.inputsLength());
        for( int i=0; i<s.inputsLength(); i++ ){
            l.add(s.inputs(i));
        }

        assertEquals(sd.inputs(), l);
    }


    public static class LogFileWriter {
        private final File file;
        public LogFileWriter(File file) throws IOException{
//            this.file = new RandomAccessFile(file, "rw");
            this.file = file;
        }

        public long writeGraphStructure(SameDiff sd) throws IOException{

            FlatBufferBuilder fbb1 = new FlatBufferBuilder(0);

            byte type = UIInfoType.GRAPH_STRUCTURE;
            int staticInfoOffset = UIStaticInfoRecord.createUIStaticInfoRecord(fbb1, type);
            fbb1.finish(staticInfoOffset);
            int lengthHeader = fbb1.offset();       //MUST be called post finish to get real length

            FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
            int graphStructureOffset = encodeGraphStructure(fbb2, sd);
            fbb2.finish(graphStructureOffset);
            int lengthContent = fbb2.offset();

            ByteBuffer bb1 = fbb1.dataBuffer();
            ByteBuffer bb2 = fbb2.dataBuffer();

//            System.out.println("Static offset: " + staticOffInfoOffset);
//            System.out.println("Graph structure offset: " + graphStructureOffset);
            try(RandomAccessFile f = new RandomAccessFile(file, "rw"); FileChannel fc = f.getChannel(); FileLock lock = fc.lock()){
//                bb.flip();  //Reset for reading

                //Write header - length of SystemInfo header, length of content header
                //TODO make this more efficient...
                ByteBuffer header = ByteBuffer.allocate(8); //8 bytes = 2x 4 byte integers
                header.putInt(lengthHeader);
                header.putInt(lengthContent);
                header.flip();

                System.out.println("Lengths - header, content: " + lengthHeader + ", " + lengthContent);

                int l1 = fc.write(header);
                int l2 = fc.write(bb1);
                int l3 = fc.write(bb2);
//                long length = fc.write(new ByteBuffer[]{header, bb});
                return 8 + l1 + l2 + l3;
            }
        }

        private int encodeGraphStructure(FlatBufferBuilder fbb, SameDiff sd){
            //Create inputs list:
            List<String> inputs = sd.inputs();
            int[] inputListStrOffsets = new int[inputs.size()];
            for(int i=0; i<inputs.size(); i++ ){
                inputListStrOffsets[i] = fbb.createString(inputs.get(i));
            }
            int inputsOffset = UIGraphStructure.createInputsVector(fbb, inputListStrOffsets);

            //Create inputs pair list:
            int inputPairOffset = -1;

            //Create outputs list:
            int outputsOffset = -1;

            //Create variables list
            int outputsListOffset = -1;

            //Create ops list
            int opsListOffset = -1;

            return UIGraphStructure.createUIGraphStructure(fbb, inputsOffset, inputPairOffset, outputsOffset, outputsListOffset, opsListOffset);
        }

        public Pair<UIStaticInfoRecord, Table> readStatic() throws IOException {
            //For testing purposes - read only first entry...

            try(RandomAccessFile f = new RandomAccessFile(file, "r"); FileChannel fc = f.getChannel()) {
//                f.seek(0);

                //read 2 header ints - file length;
                int lengthHeader = f.readInt();
                int lengthContent = f.readInt();

//                long position = f.getChannel().position();
//                System.out.println("POSITION: " + position);

                ByteBuffer bb = ByteBuffer.allocate(lengthHeader);
                int numRead = f.getChannel().read(bb);
                System.out.println("Read header bytes: " + numRead);
                bb.flip();      //Flip for reading


                UIStaticInfoRecord r = UIStaticInfoRecord.getRootAsUIStaticInfoRecord(bb);
                bb = ByteBuffer.allocate(lengthContent);
                int contentRead = f.getChannel().read(bb);
                System.out.println("Read content bytes: " + contentRead);
                bb.flip();      //Flip for reading

                byte infoType = r.infoType();
                Table t;
                switch (infoType) {
                    case UIInfoType.GRAPH_STRUCTURE:
                        t = UIGraphStructure.getRootAsUIGraphStructure(bb);
                        break;
                    case UIInfoType.SYTEM_INFO:
                        t = UISystemInfo.getRootAsUISystemInfo(bb);
                        break;
                    case UIInfoType.START_EVENTS:
                        t = null;
                        break;
                    default:
                        throw new RuntimeException("Unknown UI static info type: " + r.infoType());
                }

                //TODO do we need to close file here?

                return new Pair<>(r, t);
            }
        }
    }
}
