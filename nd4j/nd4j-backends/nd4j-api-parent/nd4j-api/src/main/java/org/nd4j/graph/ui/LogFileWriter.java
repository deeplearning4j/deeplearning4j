/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.graph.ui;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.common.base.Preconditions;
import org.nd4j.graph.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.BaseCompatOp;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Enter;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Exit;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.NextIteration;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Log file writer - for writing append-only log file for UI etc.
 *
 * File format description: There are 2 blocks in the file
 * 1. The "static information" block - containing 0 or more static info frames, followed by a UIStaticInfoRecord
 *    with UIInfoType.START_EVENTS to signify the end of the "static information" section (start of events).
 * 2. The "events" block - containing 0 or more event frames
 *
 * Note that once the UIInfoType.START_EVENTS UIStaticInfoRecord marker has been written, no more static information
 * may be written; before this point, no event records may be written. This allows us to scan only the start of the
 * file to get 'static' information such as the graph structure and hardware information.
 *
 *
 * Data (for both sections) is recorded in "frames".<br>
 * Each frame is formatted as follows:<br>
 * [header_length_integer, content_length_integer, header_bytes, content_bytes]<br>
 * These are as follows:
 * <b>header_length_integer</b>: a signed 32-bit integer representing the number of bytes of the header data (stored in the "header_bytes" section of the frame)<br>
 * <b>content_length_integer</b>: a signed 32-bit integer representing the number of bytes of the content data (stored in the "content_bytes" section of the frame).
 *   May be 0 in some cases (such as when recording UIInfoType.START_EVENTS marker<br>
 * <b>header_bytes</b>: flat-buffer encoded bytes representing either a UIStaticInfoRecord (in static information block) or a UIEvent (in the event block)
 *   In both cases, this header object encodes the typo of data that follows.
 * <b>content_bytes</b>: flat-buffer encoded bytes representing the content, of the type specified in the header<br>
 * <br>
 * <br>
 *
 * To-do list:<br>
 * * Implement recording of remaining event types<br>
 * * Handle loading already existing files<br>
 * *
 *
 * @author Alex Black
 */
@Slf4j
public class LogFileWriter {
    public enum EventSubtype {NONE, EVALUATION, LOSS, LEARNING_RATE, TUNING_METRIC, PERFORMANCE, PROFILING, FEATURE_LABEL, PREDICTION, USER_CUSTOM;

        public byte asUIEventSubtype(){
            switch (this){
                case NONE:
                    return UIEventSubtype.NONE;
                case EVALUATION:
                    return UIEventSubtype.EVALUATION;
                case LOSS:
                    return UIEventSubtype.LOSS;
                case LEARNING_RATE:
                    return UIEventSubtype.LEARNING_RATE;
                case TUNING_METRIC:
                    return UIEventSubtype.TUNING_METRIC;
                case PERFORMANCE:
                    return UIEventSubtype.PERFORMANCE;
                case PROFILING:
                    return UIEventSubtype.PROFILING;
                case FEATURE_LABEL:
                    return UIEventSubtype.FEATURE_LABEL;
                case PREDICTION:
                    return UIEventSubtype.PREDICTION;
                case USER_CUSTOM:
                    return UIEventSubtype.USER_CUSTOM;
                default:
                    throw new RuntimeException();
            }
        }
    }

    private final File file;
    private long endStaticInfoOffset = -1;
    private final AtomicInteger nameIndexCounter = new AtomicInteger(0);
    private final Map<Integer,String> nameIndexMap = new HashMap<>();
    private final Map<String,Integer> indexNameMap = new HashMap<>();

    public LogFileWriter(File file) throws IOException {
        this.file = file;
        if(file.exists()){
            //Restore state
            StaticInfo si = readStatic();
            List<Pair<UIStaticInfoRecord, Table>> staticList = si.getData();
            long staticInfoOffset = 0;
            boolean seenEndStatic = false;
            for( int i=0; i<staticList.size(); i++ ){
                UIStaticInfoRecord r = staticList.get(i).getFirst();
                if(r.infoType() == UIInfoType.START_EVENTS){
                    seenEndStatic = true;
                }
                staticInfoOffset += r.getByteBuffer().capacity();
            }

//            if(seenEndStatic)
//                endStaticInfoOffset = staticInfoOffset;
            endStaticInfoOffset = si.getFileOffset();

            //Restore names:
            List<Pair<UIEvent, Table>> events = readEvents();
            for(Pair<UIEvent, Table> p : events){
                if(p.getFirst().eventType() == UIEventType.ADD_NAME){
                    nameIndexCounter.getAndIncrement();
                    UIAddName name = (UIAddName) p.getSecond();
                    nameIndexMap.put(name.nameIdx(), name.name());
                    indexNameMap.put(name.name(), name.nameIdx());
                }
            }
        }
    }

    /**
     * Write the graph structure
     * @param sd SameDiff instance to write the graph structure for
     * @return Number of bytes written
     * @throws IOException
     */
    public long writeGraphStructure(SameDiff sd) throws IOException {
        Preconditions.checkState(endStaticInfoOffset < 0, "Cannot write graph structure - already wrote end of static info marker");
        Pair<Integer, FlatBufferBuilder> h = encodeStaticHeader(UIInfoType.GRAPH_STRUCTURE);

        FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
        int graphStructureOffset = encodeGraphStructure(fbb2, sd);
        fbb2.finish(graphStructureOffset);

        return append(h.getSecond(), fbb2);
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

                //read 2 header ints
                int lengthHeader = f.readInt();
                int lengthContent = f.readInt();

                //Read header
                ByteBuffer bb = ByteBuffer.allocate(lengthHeader);
                f.getChannel().read(bb);
                bb.flip();      //Flip for reading
                UIStaticInfoRecord r = UIStaticInfoRecord.getRootAsUIStaticInfoRecord(bb);

                //Read content
                bb = ByteBuffer.allocate(lengthContent);
                f.getChannel().read(bb);
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

    /**
     * Read all of the events.
     *
     * @return All of the UI events
     */
    public List<Pair<UIEvent, Table>> readEvents() throws IOException {
        //TODO eventually we'll support working out the offset for files that were not written in this session
        Preconditions.checkState(endStaticInfoOffset >= 0, "Cannot read events - have not written end of static info marker");
        return readEvents(endStaticInfoOffset);
    }

    /**
     * Read all of the events starting at a specific file offset
     *
     * @return All of the UI events
     */
    public List<Pair<UIEvent, Table>> readEvents(long startOffset) throws IOException {
        if(endStaticInfoOffset >= file.length()){
            return Collections.emptyList();
        }

        List<Pair<UIEvent, Table>> out = new ArrayList<>();
        try (RandomAccessFile f = new RandomAccessFile(file, "r"); FileChannel fc = f.getChannel()) {
            f.seek(startOffset);
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

                    //TODO
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

    public boolean registeredEventName(String name){
        return indexNameMap.containsKey(name);
    }

    public long registerEventNameQuiet(String name) {
        Preconditions.checkState(!registeredEventName(name), "Event name \"%s\" has already been registered", name);
        try {
            return registerEventName(name);
        } catch (IOException e){
            throw new RuntimeException("Error writing to log file", e);
        }
    }

    /**
     * Register the event name - "accuracy", "loss", etc for later use in recording events.
     * @param name Name to register
     * @return Number of bytes written
     */
    public long registerEventName(String name) throws IOException {
        Preconditions.checkState(endStaticInfoOffset >= 0, "Cannot write name - have not written end of static info marker");

        FlatBufferBuilder fbb = new FlatBufferBuilder(0);
        long time = System.currentTimeMillis();
        int offset = UIEvent.createUIEvent(fbb, UIEventType.ADD_NAME, UIEventSubtype.NONE, -1, time, 0, 0, (short)-1, 0, 0);
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

    /**
     * Write a single scalar event to the file
     * @param name      Name of the event. Must be registered by {@link #registerEventName(String)} first!
     * @param time      Timestamp
     * @param iteration Iteration for the event
     * @param epoch     Epoch for the event
     * @param scalar    Scalar value to write
     * @return          Number of bytes written
     */
    public long writeScalarEvent(String name, EventSubtype subtype, long time, int iteration, int epoch, Number scalar) throws IOException {
        //TODO add support for plugin, variable and frame/iter
        Preconditions.checkState(indexNameMap.containsKey(name), "Name \"%s\" not yet registered", name);
        int idx = indexNameMap.get(name);
        FlatBufferBuilder fbb = new FlatBufferBuilder(0);
        int offset = UIEvent.createUIEvent(fbb, UIEventType.SCALAR, subtype.asUIEventSubtype(), idx, time, iteration, epoch, (short)-1, 0, 0);
        fbb.finish(offset);

        FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
        int offset2 = Nd4j.scalar(scalar).toFlatArray(fbb2);
        fbb2.finish(offset2);

        return append(fbb, fbb2);
    }

    public long writeHistogramEventDiscrete(@NonNull String name, EventSubtype subtype, long time, int iteration, int epoch, List<String> binLabels, @NonNull INDArray y) throws IOException {
        Preconditions.checkState(binLabels == null || binLabels.size() == y.length(), "Number of bin labels (if present) must " +
                "be same as Y array length - got %s bins, array shape %ndShape", (binLabels == null ? 0L : binLabels.size()), y.length());
        Preconditions.checkState(y.rank() == 1, "Y array must be rank 1, got Y array with shape %ndShape", y);

        //TODO add support for plugin, variable and frame/iter
        Preconditions.checkState(indexNameMap.containsKey(name), "Name \"%s\" not yet registered", name);
        int idx = indexNameMap.get(name);

        FlatBufferBuilder fbb = new FlatBufferBuilder(0);
        int offset = UIEvent.createUIEvent(fbb, UIEventType.HISTOGRAM, subtype.asUIEventSubtype(), idx, time, iteration, epoch, (short)-1, 0, 0);
        fbb.finish(offset);

        FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
        int yOffset = y.toFlatArray(fbb2);
        int binLabelsOffset = 0;
        if(binLabels != null){
            int[] str = new int[binLabels.size()];
            for( int i=0; i<binLabels.size(); i++ ){
                String s = binLabels.get(i);
                if(s == null)
                    s = "";
                str[i] = fbb2.createString(s);
            }
            binLabelsOffset = UIHistogram.createBinlabelsVector(fbb2, str);
        }
        int offset2 = UIHistogram.createUIHistogram(fbb2, UIHistogramType.DISCRETE, y.length(), 0, yOffset, binLabelsOffset);
        fbb2.finish(offset2);

        return append(fbb, fbb2);
    }

    public long writeHistogramEventEqualSpacing(String name, EventSubtype subtype, long time, int iteration, int epoch, double min, double max, INDArray y) throws IOException {
        Preconditions.checkState(y.rank() == 1, "Y array must be rank 1, got Y array with shape %ndShape", y);
        Preconditions.checkState(max > min, "Maximum histogram value must be greater than minimum - got max=%s, min=%s", max, min);

        //TODO add support for plugin, variable and frame/iter
        //TODO: Code duplication for histogram methods...
        Preconditions.checkState(indexNameMap.containsKey(name), "Name \"%s\" not yet registered", name);
        int idx = indexNameMap.get(name);

        FlatBufferBuilder fbb = new FlatBufferBuilder(0);
        int offset = UIEvent.createUIEvent(fbb, UIEventType.HISTOGRAM, subtype.asUIEventSubtype(), idx, time, iteration, epoch, (short)-1, 0, 0);
        fbb.finish(offset);

        FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
        int yOffset = y.toFlatArray(fbb2);

        INDArray binRangesArr = Nd4j.createFromArray(min, max);
        int binRangesOffset = binRangesArr.toFlatArray(fbb2);

        int offset2 = UIHistogram.createUIHistogram(fbb2, UIHistogramType.EQUAL_SPACING, y.length(), binRangesOffset, yOffset, 0);
        fbb2.finish(offset2);

        return append(fbb, fbb2);
    }

    public long writeHistogramEventCustomBins(String name, EventSubtype subtype, long time, int iteration, int epoch, INDArray bins, INDArray y) throws IOException {
        Preconditions.checkState(y.rank() == 1, "Y array must be rank 1, got Y array with shape %ndShape", y);
        Preconditions.checkState(bins.rank() == 2, "Bins array must have shape [2,numBins], got bins array with shape %ndShape", bins);
        Preconditions.checkState(y.length() == bins.size(1), "Bins array must have shape [2,numBins], where numBins must match y.length()=%s, got bins array with shape %ndShape", y.length(), bins);

        //TODO add support for plugin, variable and frame/iter
        //TODO: Code duplication for histogram methods...
        Preconditions.checkState(indexNameMap.containsKey(name), "Name \"%s\" not yet registered", name);
        int idx = indexNameMap.get(name);

        FlatBufferBuilder fbb = new FlatBufferBuilder(0);
        int offset = UIEvent.createUIEvent(fbb, UIEventType.HISTOGRAM, subtype.asUIEventSubtype(), idx, time, iteration, epoch, (short)-1, 0, 0);
        fbb.finish(offset);

        FlatBufferBuilder fbb2 = new FlatBufferBuilder(0);
        int yOffset = y.toFlatArray(fbb2);

        int binRangesOffset = bins.toFlatArray(fbb2);

        int offset2 = UIHistogram.createUIHistogram(fbb2, UIHistogramType.CUSTOM, y.length(), binRangesOffset, yOffset, 0);
        fbb2.finish(offset2);

        return append(fbb, fbb2);
    }

    /**
     * Encode the header as a UIStaticInfoRecord instance for the specific {@link UIEventType}
     * @param type UIEventType
     */
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
        int outputsOffset = 0;
        if(outputs != null && !outputs.isEmpty()) {
            int[] outputListStrOffsets = new int[outputs.size()];
            for (int i = 0; i < outputListStrOffsets.length; i++) {
                outputListStrOffsets[i] = fbb.createString(outputs.get(i));
            }
            outputsOffset = UIGraphStructure.createInputsVector(fbb, outputListStrOffsets);
        }


        //Create variables list
        Map<String,Variable> varMap = sd.getVariables();
        int[] varListOffsets = new int[varMap.size()];
        int count = 0;
        for(Map.Entry<String,Variable> e : varMap.entrySet()){
            int intPair = 0;
            int name = fbb.createString(e.getKey());

            String outputOfOp = e.getValue().getOutputOfOp();
            int outputOfOpIdx = 0;
            if(outputOfOp != null)
                outputOfOpIdx = fbb.createString(outputOfOp);

            List<String> inputsForOps = e.getValue().getInputsForOp();
            int inputsForOpIdx = 0;
            if(inputsForOps != null && !inputsForOps.isEmpty()){
                int[] idx = encodeStrings(fbb, inputsForOps);
                inputsForOpIdx = UIVariable.createInputsForOpVector(fbb, idx);
            }

            List<String> controlDepsForOp = e.getValue().getControlDepsForOp();
            int controlDepsForOpIdx = 0;
            if(controlDepsForOp != null && !controlDepsForOp.isEmpty()){
                int[] idx = encodeStrings(fbb, controlDepsForOp);
                controlDepsForOpIdx = UIVariable.createInputsForOpVector(fbb, idx);
            }

            List<String> controlDepsForVar = e.getValue().getControlDepsForVar();
            int controlDepsForVarIdx = 0;
            if(controlDepsForVar != null && !controlDepsForVar.isEmpty()){
                int[] idx = encodeStrings(fbb, controlDepsForVar);
                controlDepsForVarIdx = UIVariable.createInputsForOpVector(fbb, idx);
            }

            DataType dt = e.getValue().getVariable().dataType();
            byte dtVal = FlatBuffersMapper.getDataTypeAsByte(dt);

            long[] shape = e.getValue().getVariable().getShape();
            int shapeOffset = 0;
            if(shape != null){
                shapeOffset = UIVariable.createShapeVector(fbb, shape);
            }

            int controlDepsIdx = 0;
            if(e.getValue().getControlDeps() != null ){
                List<String> cds = e.getValue().getControlDeps();
                if(!cds.isEmpty()){
                    int[] cdIdxs = new int[cds.size()];
                    for( int i=0; i<cdIdxs.length; i++ ){
                        cdIdxs[i] = fbb.createString(cds.get(i));
                    }
                    controlDepsIdx = UIVariable.createControlDepsVector(fbb, cdIdxs);
                }
            }

            int uiExtraLabelOffset = 0;     //String value - "extra" information to be shown in label. Currently unused
            int constantValueOffset = 0;
            if(e.getValue().getVariable().getVariableType() == VariableType.CONSTANT){
                INDArray arr = e.getValue().getVariable().getArr();
                if(arr != null && arr.length() < 1000){
                    constantValueOffset = arr.toFlatArray(fbb);
                }
            }

            int uiVariableIdx = UIVariable.createUIVariable(fbb,
                    intPair,
                    name,
                    FlatBuffersMapper.toVarType(e.getValue().getVariable().getVariableType()),
                    dtVal,
                    shapeOffset,
                    controlDepsIdx,
                    outputOfOpIdx,
                    inputsForOpIdx,
                    controlDepsForOpIdx,
                    controlDepsForVarIdx,
                    0,       //TODO gradient variable
                    uiExtraLabelOffset,
                    constantValueOffset
            );

            varListOffsets[count++] = uiVariableIdx;
        }
        int outputsListOffset = UIGraphStructure.createVariablesVector(fbb, varListOffsets);


        //Create ops list
        Map<String,SameDiffOp> opMap = sd.getOps();
        count = 0;
        int[] opListOffsets = new int[opMap.size()];
        for(Map.Entry<String,SameDiffOp> e : opMap.entrySet()){

            int nameIdx = fbb.createString(e.getKey());
            String opName = e.getValue().getOp().opName();
            int opNameIdx = fbb.createString(opName);

            //Op input variables
            int inputsIdx = 0;
            List<String> opInputs = e.getValue().getInputsToOp();
            if(opInputs != null && !opInputs.isEmpty()){
                int[] idx = encodeStrings(fbb, opInputs);
                inputsIdx = UIOp.createInputsVector(fbb, idx);
            }

            //Op output variables
            int outputsIdx = 0;
            List<String> opOutputs = e.getValue().getOutputsOfOp();
            if(opOutputs != null && !opOutputs.isEmpty()){
                int[] idx = encodeStrings(fbb, opOutputs);
                outputsIdx = UIOp.createOutputsVector(fbb, idx);
            }

            int controlDepIdxs = 0;
            List<String> opCDeps = e.getValue().getControlDeps();
            if(opCDeps != null && !opCDeps.isEmpty()){
                int[] idx = encodeStrings(fbb, opCDeps);
                controlDepIdxs = UIOp.createControlDepsVector(fbb, idx);
            }

            int extraLabelOffset = 0;
            DifferentialFunction df = e.getValue().getOp();
            if(df instanceof Enter || df instanceof Exit || df instanceof NextIteration){ //Enter, Exit, NextIteration
                String frame = ((BaseCompatOp) df).getFrameName();
                if(frame != null) {
                    String extra = "Frame: \"" + frame + "\"";
                    extraLabelOffset = fbb.createString(extra);
                }
            }

            opListOffsets[count++] = UIOp.createUIOp(fbb,
                    nameIdx,
                    opNameIdx,
                    inputsIdx,
                    outputsIdx,
                    controlDepIdxs,
                    extraLabelOffset);

        }
        int opsListOffset = UIGraphStructure.createOpsVector(fbb, opListOffsets);

        return UIGraphStructure.createUIGraphStructure(fbb, inputsOffset, inputPairOffset, outputsOffset, outputsListOffset, opsListOffset);
    }

    private int[] encodeStrings(FlatBufferBuilder fbb, List<String> list){
        if(list == null || list.isEmpty())
            return null;
        int[] idx = new int[list.size()];
        for( int i=0; i<idx.length; i++ ){
            idx[i] = fbb.createString(list.get(i));
        }
        return idx;
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

            //System.out.println("Lengths - header, content: " + l1 + ", " + l2);

            int w1 = fc.write(header);
            int w2 = fc.write(bb1);
            int w3 = bb2 == null ? 0 : fc.write(bb2);
            long total = w1 + w2 + w3;
            //System.out.println("Wrote " + total + " bytes starting at position " + startPos);
            //System.out.println("Post writing file length: " + file.length());
            return total;
        }
    }



    @AllArgsConstructor
    @Data
    public static class StaticInfo {
        private final List<Pair<UIStaticInfoRecord, Table>> data;
        private final long fileOffset;
    }
}
