package org.nd4j.autodiff.ui;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
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

@Slf4j
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
        SDVariable v = sd.var("variable", DataType.DOUBLE, 3, 4);
        SDVariable sum = v.sum();

        File f = testDir.newFile();
        if (f.exists())
            f.delete();
        System.out.println(f.getAbsolutePath());
        LogFileWriter w = new LogFileWriter(f);
//        System.out.println("Writing graph structure");
        long bytesWritten = w.writeGraphStructure(sd);
//        System.out.println("Writing static info");
        long bytesWritten2 = w.finishStatic();
//        System.out.println("Completed writing");

        assertTrue(bytesWritten > 0);
        assertTrue(bytesWritten2 > 0);

        StaticInfo read = w.readStatic();

        assertEquals(2, read.getData().size());

        long fileLength = f.length();
        assertEquals(fileLength, read.getFileOffset());

        //Check graph structure:
        UIGraphStructure s = (UIGraphStructure) read.getData().get(0).getSecond();
        List<String> l = new ArrayList<>(s.inputsLength());
        for (int i = 0; i < s.inputsLength(); i++) {
            l.add(s.inputs(i));
        }
        assertEquals(sd.inputs(), l);

        List<String> outputs = new ArrayList<>(s.outputsLength());
        for (int i = 0; i < s.outputsLength(); i++) {
            outputs.add(s.outputs(i));
        }
        assertEquals(sd.outputs(), outputs);

        assertEquals(UIInfoType.START_EVENTS, read.getData().get(1).getFirst().infoType());


        //Append a number of events
    }


    public static class LogFileWriter {
        private final File file;
        private long endStaticInfoOffset = -1;

        public LogFileWriter(File file) throws IOException {
            this.file = file;
        }

        public long writeGraphStructure(SameDiff sd) throws IOException {
            Pair<Integer, FlatBufferBuilder> h = encodeStaticHeader(UIInfoType.GRAPH_STRUCTURE);

            FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
            int graphStructureOffset = encodeGraphStructure(fbb2, sd);
            fbb2.finish(graphStructureOffset);

            long written = append(h.getSecond(), fbb2);
            return written;
        }

        public Pair<Integer, FlatBufferBuilder> encodeStaticHeader(byte type) {
            FlatBufferBuilder fbb = new FlatBufferBuilder(12);

            int staticInfoOffset = UIStaticInfoRecord.createUIStaticInfoRecord(fbb, type);
            fbb.finish(staticInfoOffset);
            int lengthHeader = fbb.offset();       //MUST be called post finish to get real length
            return new Pair<>(lengthHeader, fbb);
        }

        private int encodeGraphStructure(FlatBufferBuilder fbb, SameDiff sd) {
            //Create inputs list:
            List<String> inputs = sd.inputs();
            int[] inputListStrOffsets = new int[inputs.size()];
            for (int i = 0; i < inputs.size(); i++) {
                inputListStrOffsets[i] = fbb.createString(inputs.get(i));
            }
            int inputsOffset = UIGraphStructure.createInputsVector(fbb, inputListStrOffsets);

            //Create inputs pair list:
            int inputPairOffset = -1;

            //Create outputs list:
            List<String> outputs = sd.outputs();
            int[] outputListStrOffsets = new int[outputs.size()];
            for (int i = 0; i < outputListStrOffsets.length; i++) {
                outputListStrOffsets[i] = fbb.createString(outputs.get(i));
            }
            int outputsOffset = UIGraphStructure.createInputsVector(fbb, outputListStrOffsets);

            //Create variables list
            int outputsListOffset = -1;

            //Create ops list
            int opsListOffset = -1;

            return UIGraphStructure.createUIGraphStructure(fbb, inputsOffset, inputPairOffset, outputsOffset, outputsListOffset, opsListOffset);
        }

        private long append(FlatBufferBuilder h, FlatBufferBuilder c) throws IOException {
            ByteBuffer bb1 = h.dataBuffer();
            ByteBuffer bb2 = (c == null ? null : c.dataBuffer());

            try (RandomAccessFile f = new RandomAccessFile(file, "rw"); FileChannel fc = f.getChannel(); FileLock lock = fc.lock()) {
                //TODO can we make this more efficient - use a single byte buffer?

                //Seek to end for append
                f.seek(f.length());
                long startPos = f.getFilePointer();

                //Write header - length of SystemInfo header, length of content header
                ByteBuffer header = ByteBuffer.allocate(8); //8 bytes = 2x 4 byte integers
                int l1 = bb1.remaining();
                int l2 = bb2 == null ? 0 : bb2.remaining();
                header.putInt(l1);
                header.putInt(l2);
                header.flip();

                System.out.println("Lengths - header, content: " + l1 + ", " + l2);

                int w1 = fc.write(header);
                int w2 = fc.write(bb1);
                int w3 = bb2 == null ? 0 : fc.write(bb2);
                long total = w1 + w2 + w3;
                System.out.println("Wrote " + total + " bytes starting at position " + startPos);
                System.out.println("Post writing file length: " + file.length());
                return total;
            }
        }

        public long finishStatic() throws IOException {
            Preconditions.checkState(endStaticInfoOffset < 0, "Wrote final static already information already");
            Pair<Integer, FlatBufferBuilder> encoded = encodeStaticHeader(UIInfoType.START_EVENTS);
            long out = append(encoded.getSecond(), null);
            endStaticInfoOffset = file.length();
            return out;
        }

        /**
         * Read all static information at the start of the file
         * @return
         * @throws IOException
         */
        public StaticInfo readStatic() throws IOException {
            //For testing purposes - read only first entry...

            List<Pair<UIStaticInfoRecord, Table>> out = new ArrayList<>();
            boolean allStaticRead = false;
            try (RandomAccessFile f = new RandomAccessFile(file, "r"); FileChannel fc = f.getChannel()) {
                f.seek(0);
                while (!allStaticRead) {
//                    System.out.println("---------- Reading ----------");

                    //read 2 header ints - file length;
                    int lengthHeader = f.readInt();
                    int lengthContent = f.readInt();

                    ByteBuffer bb = ByteBuffer.allocate(lengthHeader);
                    int numRead = f.getChannel().read(bb);
//                    System.out.println("Read header bytes: " + numRead);
                    bb.flip();      //Flip for reading


                    UIStaticInfoRecord r = UIStaticInfoRecord.getRootAsUIStaticInfoRecord(bb);
                    bb = ByteBuffer.allocate(lengthContent);
                    int contentRead = f.getChannel().read(bb);
//                    System.out.println("Read content bytes: " + contentRead);
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

                    out.add(new Pair<>(r, t));
                    long pointer = f.getFilePointer();
                    long length = f.length();
                    {
                        log.trace("File pointer = {}, file length = {}", pointer, length);
                        if (infoType == UIInfoType.START_EVENTS || pointer >= length) {
                            allStaticRead = true;
                        }
                    }
                }
                StaticInfo s = new StaticInfo(out, f.getFilePointer());
                return s;
            }
        }
    }

    @AllArgsConstructor
    @Data
    public static class StaticInfo {
        private final List<Pair<UIStaticInfoRecord, Table>> data;
        private final long fileOffset;
    }
}
