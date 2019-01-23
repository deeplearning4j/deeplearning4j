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
import org.nd4j.graph.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

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
        long bytesWritten2 = w.writeFinishStaticMarker();
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
        w.registerName("accuracy");
        for( int iter=0; iter<3; iter++) {
            long t = System.currentTimeMillis();
            w.writeScalar("accuracy", t, iter, 0, 0.5 + 0.1 * iter);
        }

        //Read events back in...
        List<Pair<UIEvent,Table>> events = w.readEvents();
        assertEquals(4, events.size());             //add name + 3 scalars

        UIAddName addName = (UIAddName) events.get(0).getRight();
        assertEquals("accuracy", addName.name());

        for( int i=1; i<4; i++ ){
            FlatArray fa = (FlatArray) events.get(i).getRight();
            INDArray arr = Nd4j.createFromFlatArray(fa);

            INDArray exp = Nd4j.scalar(0.5 + (i-1) * 0.1);
            assertEquals(exp, arr);
        }
    }


    public static class LogFileWriter {
        private final File file;
        private long endStaticInfoOffset = -1;
        private final AtomicInteger nameIndexCounter = new AtomicInteger(0);
        private final Map<Integer,String> nameIndexMap = new HashMap<>();
        private final Map<String,Integer> indexNameMap = new HashMap<>();

        public LogFileWriter(File file) throws IOException {
            this.file = file;
        }

        /**
         * Write the graph structure
         * @param sd
         * @return
         * @throws IOException
         */
        public long writeGraphStructure(SameDiff sd) throws IOException {
            Preconditions.checkState(endStaticInfoOffset < 0, "Cannot write graph structure - already wrote end of static info marker");
            Pair<Integer, FlatBufferBuilder> h = encodeStaticHeader(UIInfoType.GRAPH_STRUCTURE);

            FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
            int graphStructureOffset = encodeGraphStructure(fbb2, sd);
            fbb2.finish(graphStructureOffset);

            long written = append(h.getSecond(), fbb2);
            return written;
        }

        protected Pair<Integer, FlatBufferBuilder> encodeStaticHeader(byte type) {
            FlatBufferBuilder fbb = new FlatBufferBuilder(12);

            int staticInfoOffset = UIStaticInfoRecord.createUIStaticInfoRecord(fbb, type);
            fbb.finish(staticInfoOffset);
            int lengthHeader = fbb.offset();       //MUST be called post finish to get real length
            return new Pair<>(lengthHeader, fbb);
        }

        protected int encodeGraphStructure(FlatBufferBuilder fbb, SameDiff sd) {
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

        /**
         * Write marker for final static data
         * @return
         * @throws IOException
         */
        public long writeFinishStaticMarker() throws IOException {
            Preconditions.checkState(endStaticInfoOffset < 0, "Wrote final static already information already");
            Pair<Integer, FlatBufferBuilder> encoded = encodeStaticHeader(UIInfoType.START_EVENTS);
            long out = append(encoded.getSecond(), null);
            endStaticInfoOffset = file.length();
            return out;
        }

        /**
         * Read all static information at the start of the file
         *
         * @return
         * @throws IOException
         */
        public StaticInfo readStatic() throws IOException {

            List<Pair<UIStaticInfoRecord, Table>> out = new ArrayList<>();
            boolean allStaticRead = false;
            try (RandomAccessFile f = new RandomAccessFile(file, "r"); FileChannel fc = f.getChannel()) {
                f.seek(0);
                while (!allStaticRead) {

                    //read 2 header ints - file length;
                    int lengthHeader = f.readInt();
                    int lengthContent = f.readInt();

                    ByteBuffer bb = ByteBuffer.allocate(lengthHeader);
                    int numRead = f.getChannel().read(bb);
                    bb.flip();      //Flip for reading


                    UIStaticInfoRecord r = UIStaticInfoRecord.getRootAsUIStaticInfoRecord(bb);
                    bb = ByteBuffer.allocate(lengthContent);
                    int contentRead = f.getChannel().read(bb);
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

        public List<Pair<UIEvent, Table>> readEvents() throws IOException {
            //TODO eventually we'll support working out the offset
            Preconditions.checkState(endStaticInfoOffset >= 0, "Cannot read events - have not written end of static info marker");

            if(endStaticInfoOffset >= file.length()){
                return Collections.emptyList();
            }

            List<Pair<UIEvent, Table>> out = new ArrayList<>();
            try (RandomAccessFile f = new RandomAccessFile(file, "r"); FileChannel fc = f.getChannel()) {
                f.seek(endStaticInfoOffset);
                while (f.getFilePointer() < f.length()) {
                    //read 2 header ints
                    int lengthHeader = f.readInt();
                    int lengthContent = f.readInt();

                    //Read header
                    ByteBuffer bb = ByteBuffer.allocate(lengthHeader);
                    f.getChannel().read(bb);
                    bb.flip();      //Flip for reading
                    UIEvent e = UIEvent.getRootAsUIEvent(bb);

                    //Read Content
                    bb = ByteBuffer.allocate(lengthContent);
                    f.getChannel().read(bb);
                    bb.flip();      //Flip for reading

                    byte infoType = e.eventType();
                    Table t;
                    switch (infoType) {
                        case UIEventType.ADD_NAME:
                            t = UIAddName.getRootAsUIAddName(bb);
                            break;
                        case UIEventType.SCALAR:
                        case UIEventType.ARRAY:
                            t = FlatArray.getRootAsFlatArray(bb);
                            break;
                        case UIEventType.ARRAY_LIST:
                        case UIEventType.HISTOGRAM:
                        case UIEventType.IMAGE:
                        case UIEventType.SUMMARY_STATISTICS:
                        case UIEventType.OP_TIMING:
                        case UIEventType.HARDWARE_STATE:
                        case UIEventType.GC_EVENT:
                        default:
                            throw new RuntimeException("Unknown or not yet implemented event type: " + e.eventType());
                    }

                    //TODO do we need to close file here?

                    out.add(new Pair<>(e, t));
                }
                return out;
            }
        }


        public long registerName(String name) throws IOException {
            Preconditions.checkState(endStaticInfoOffset >= 0, "Cannot write name - have not written end of static info marker");

            FlatBufferBuilder fbb = new FlatBufferBuilder(0);
            long time = System.currentTimeMillis();
            int offset = UIEvent.createUIEvent(fbb, UIEventType.ADD_NAME, -1, time, 0, 0, (short)-1, 0, 0);
            fbb.finish(offset);

            FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
            int idx = nameIndexCounter.getAndIncrement();
            nameIndexMap.put(idx, name);
            indexNameMap.put(name, idx);
            int strOffset = fbb2.createString(name);
            int offset2 = UIAddName.createUIAddName(fbb2, idx, strOffset);
            fbb2.finish(offset2);

            long l = append(fbb, fbb2);
            return l;
        }

        //TODO add support for plugin and frame/iter
        public long writeScalar(String name, long time, int iteration, int epoch, Number scalar) throws IOException {
            Preconditions.checkState(indexNameMap.containsKey(name), "Name \"%s\" not yet registered", name);
            int idx = indexNameMap.get(name);
            FlatBufferBuilder fbb = new FlatBufferBuilder(0);
            int offset = UIEvent.createUIEvent(fbb, UIEventType.SCALAR, idx, time, iteration, epoch, (short)-1, 0, 0);
            fbb.finish(offset);

            FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
            int offset2 = Nd4j.scalar(scalar).toFlatArray(fbb2);
            fbb2.finish(offset2);

            return append(fbb, fbb2);
        }
    }

    @AllArgsConstructor
    @Data
    public static class StaticInfo {
        private final List<Pair<UIStaticInfoRecord, Table>> data;
        private final long fileOffset;
    }
}
