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

package org.nd4j.torchscript.ir;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.torchscript.TorchScriptImportException;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Parser for Python pickle format used in TorchScript files.
 * Handles pickle protocol 2+ which PyTorch uses.
 *
 * <p>This is a minimal implementation that handles the subset of pickle
 * operations used by PyTorch for model serialization.
 */
@Slf4j
public class PickleParser {

    // Pickle opcodes (protocol 2+)
    private static final byte PROTO = (byte) 0x80;
    private static final byte FRAME = (byte) 0x95;
    private static final byte STOP = '.';
    private static final byte MARK = '(';
    private static final byte POP = '0';
    private static final byte POP_MARK = '1';
    private static final byte DUP = '2';
    private static final byte NONE = 'N';
    private static final byte NEWTRUE = (byte) 0x88;
    private static final byte NEWFALSE = (byte) 0x89;
    private static final byte INT = 'I';
    private static final byte BININT = 'J';
    private static final byte BININT1 = 'K';
    private static final byte BININT2 = 'M';
    private static final byte LONG = 'L';
    private static final byte LONG1 = (byte) 0x8a;
    private static final byte LONG4 = (byte) 0x8b;
    private static final byte FLOAT = 'F';
    private static final byte BINFLOAT = 'G';
    private static final byte STRING = 'S';
    private static final byte SHORT_BINSTRING = 'U';
    private static final byte BINSTRING = 'T';
    private static final byte SHORT_BINBYTES = 'C';
    private static final byte BINBYTES = 'B';
    private static final byte BINBYTES8 = (byte) 0x8e;
    private static final byte SHORT_BINUNICODE = (byte) 0x8c;
    private static final byte BINUNICODE = 'X';
    private static final byte BINUNICODE8 = (byte) 0x8d;
    private static final byte UNICODE = 'V';
    private static final byte EMPTY_LIST = ']';
    private static final byte APPEND = 'a';
    private static final byte APPENDS = 'e';
    private static final byte LIST = 'l';
    private static final byte EMPTY_TUPLE = ')';
    private static final byte TUPLE = 't';
    private static final byte TUPLE1 = (byte) 0x85;
    private static final byte TUPLE2 = (byte) 0x86;
    private static final byte TUPLE3 = (byte) 0x87;
    private static final byte EMPTY_DICT = '}';
    private static final byte DICT = 'd';
    private static final byte SETITEM = 's';
    private static final byte SETITEMS = 'u';
    private static final byte GLOBAL = 'c';
    private static final byte STACK_GLOBAL = (byte) 0x93;
    private static final byte REDUCE = 'R';
    private static final byte BUILD = 'b';
    private static final byte INST = 'i';
    private static final byte OBJ = 'o';
    private static final byte NEWOBJ = (byte) 0x81;
    private static final byte NEWOBJ_EX = (byte) 0x92;
    private static final byte BINGET = 'h';
    private static final byte LONG_BINGET = 'j';
    private static final byte BINPUT = 'q';
    private static final byte LONG_BINPUT = 'r';
    private static final byte MEMOIZE = (byte) 0x94;
    private static final byte EXT1 = (byte) 0x82;
    private static final byte EXT2 = (byte) 0x83;
    private static final byte EXT4 = (byte) 0x84;
    private static final byte EMPTY_SET = (byte) 0x8f;
    private static final byte ADDITEMS = (byte) 0x90;
    private static final byte FROZENSET = (byte) 0x91;
    private static final byte PERSID = 'P';
    private static final byte BINPERSID = 'Q';

    private PickleParser() {
        // Utility class
    }

    /**
     * Parse a pickle byte array.
     *
     * @param data the pickle data
     * @return the parsed object
     * @throws TorchScriptImportException if parsing fails
     */
    public static Object parse(byte[] data) throws TorchScriptImportException {
        try {
            PickleStack stack = new PickleStack(data);
            return stack.parse();
        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to parse pickle data", e);
        }
    }

    /**
     * Parse pickle data and extract graph structure.
     *
     * @param data the pickle data
     * @return parsed graph (may be incomplete for complex models)
     * @throws TorchScriptImportException if parsing fails
     */
    public static TorchScriptGraph parseGraph(byte[] data) throws TorchScriptImportException {
        Object result = parse(data);

        TorchScriptGraph.TorchScriptGraphBuilder builder = TorchScriptGraph.builder();

        // Extract graph information from parsed structure
        if (result instanceof Map) {
            extractGraphFromMap(builder, (Map<?, ?>) result);
        } else if (result instanceof List) {
            extractGraphFromList(builder, (List<?>) result);
        }

        return builder.build();
    }

    private static void extractGraphFromMap(TorchScriptGraph.TorchScriptGraphBuilder builder, Map<?, ?> map) {
        // Look for common TorchScript keys
        if (map.containsKey("code")) {
            // Has code section
            log.debug("Found code section in pickle");
        }
        if (map.containsKey("model")) {
            Object model = map.get("model");
            if (model instanceof Map) {
                extractModelInfo(builder, (Map<?, ?>) model);
            }
        }
    }

    private static void extractGraphFromList(TorchScriptGraph.TorchScriptGraphBuilder builder, List<?> list) {
        // TorchScript may store operations as a list
        for (Object item : list) {
            if (item instanceof Map) {
                Map<?, ?> nodeMap = (Map<?, ?>) item;
                extractNodeInfo(builder, nodeMap);
            }
        }
    }

    private static void extractModelInfo(TorchScriptGraph.TorchScriptGraphBuilder builder, Map<?, ?> model) {
        if (model.containsKey("name")) {
            builder.name(String.valueOf(model.get("name")));
        }
    }

    private static void extractNodeInfo(TorchScriptGraph.TorchScriptGraphBuilder builder, Map<?, ?> nodeMap) {
        // This would extract node information from the pickle structure
        // Implementation depends on the specific pickle format
    }

    /**
     * Internal stack-based pickle parser.
     */
    private static class PickleStack {
        private final DataInputStream dis;
        private final Deque<Object> stack = new ArrayDeque<>();
        private final List<Object> memo = new ArrayList<>();
        private final Deque<Integer> marks = new ArrayDeque<>();
        private int protocol = 0;

        PickleStack(byte[] data) {
            this.dis = new DataInputStream(new ByteArrayInputStream(data));
        }

        Object parse() throws IOException, TorchScriptImportException {
            while (true) {
                int opcode = dis.read();
                if (opcode == -1) {
                    throw new TorchScriptImportException("Unexpected end of pickle data");
                }

                switch ((byte) opcode) {
                    case PROTO:
                        protocol = dis.readUnsignedByte();
                        log.debug("Pickle protocol: {}", protocol);
                        break;

                    case FRAME:
                        // Skip frame size (8 bytes)
                        dis.skipBytes(8);
                        break;

                    case STOP:
                        return stack.isEmpty() ? null : stack.pop();

                    case MARK:
                        marks.push(stack.size());
                        break;

                    case POP:
                        if (!stack.isEmpty()) stack.pop();
                        break;

                    case POP_MARK:
                        popMark();
                        break;

                    case DUP:
                        if (!stack.isEmpty()) stack.push(stack.peek());
                        break;

                    case NONE:
                        stack.push(null);
                        break;

                    case NEWTRUE:
                        stack.push(Boolean.TRUE);
                        break;

                    case NEWFALSE:
                        stack.push(Boolean.FALSE);
                        break;

                    case BININT:
                        stack.push(readInt());
                        break;

                    case BININT1:
                        stack.push(dis.readUnsignedByte());
                        break;

                    case BININT2:
                        stack.push(readShort());
                        break;

                    case LONG1:
                        int len1 = dis.readUnsignedByte();
                        stack.push(readLongBytes(len1));
                        break;

                    case LONG4:
                        int len4 = readInt();
                        stack.push(readLongBytes(len4));
                        break;

                    case BINFLOAT:
                        stack.push(readDouble());
                        break;

                    case SHORT_BINSTRING:
                    case SHORT_BINBYTES:
                        int slen = dis.readUnsignedByte();
                        byte[] sbytes = new byte[slen];
                        dis.readFully(sbytes);
                        stack.push(new String(sbytes, StandardCharsets.UTF_8));
                        break;

                    case BINSTRING:
                    case BINBYTES:
                        int blen = readInt();
                        byte[] bbytes = new byte[blen];
                        dis.readFully(bbytes);
                        stack.push(new String(bbytes, StandardCharsets.UTF_8));
                        break;

                    case BINBYTES8:
                        long blen8 = readLong();
                        byte[] bbytes8 = new byte[(int) blen8];
                        dis.readFully(bbytes8);
                        stack.push(bbytes8);
                        break;

                    case SHORT_BINUNICODE:
                        int ulen = dis.readUnsignedByte();
                        byte[] ubytes = new byte[ulen];
                        dis.readFully(ubytes);
                        stack.push(new String(ubytes, StandardCharsets.UTF_8));
                        break;

                    case BINUNICODE:
                        int ulen4 = readInt();
                        byte[] ubytes4 = new byte[ulen4];
                        dis.readFully(ubytes4);
                        stack.push(new String(ubytes4, StandardCharsets.UTF_8));
                        break;

                    case BINUNICODE8:
                        long ulen8 = readLong();
                        byte[] ubytes8 = new byte[(int) ulen8];
                        dis.readFully(ubytes8);
                        stack.push(new String(ubytes8, StandardCharsets.UTF_8));
                        break;

                    case EMPTY_LIST:
                        stack.push(new ArrayList<>());
                        break;

                    case EMPTY_TUPLE:
                        stack.push(Collections.emptyList());
                        break;

                    case EMPTY_DICT:
                        stack.push(new LinkedHashMap<>());
                        break;

                    case EMPTY_SET:
                        stack.push(new LinkedHashSet<>());
                        break;

                    case TUPLE:
                        stack.push(new ArrayList<>(popMark()));
                        break;

                    case TUPLE1:
                        stack.push(Collections.singletonList(stack.pop()));
                        break;

                    case TUPLE2:
                        Object t2b = stack.pop();
                        Object t2a = stack.pop();
                        stack.push(Arrays.asList(t2a, t2b));
                        break;

                    case TUPLE3:
                        Object t3c = stack.pop();
                        Object t3b = stack.pop();
                        Object t3a = stack.pop();
                        stack.push(Arrays.asList(t3a, t3b, t3c));
                        break;

                    case LIST:
                        stack.push(new ArrayList<>(popMark()));
                        break;

                    case DICT:
                        handleDict();
                        break;

                    case APPEND:
                        handleAppend();
                        break;

                    case APPENDS:
                        handleAppends();
                        break;

                    case SETITEM:
                        handleSetItem();
                        break;

                    case SETITEMS:
                        handleSetItems();
                        break;

                    case ADDITEMS:
                        handleAddItems();
                        break;

                    case GLOBAL:
                        String module = readLine();
                        String name = readLine();
                        stack.push(new GlobalRef(module, name));
                        break;

                    case STACK_GLOBAL:
                        Object gname = stack.pop();
                        Object gmodule = stack.pop();
                        stack.push(new GlobalRef(String.valueOf(gmodule), String.valueOf(gname)));
                        break;

                    case REDUCE:
                        Object args = stack.pop();
                        Object callable = stack.pop();
                        stack.push(new ReduceResult(callable, args));
                        break;

                    case BUILD:
                        Object state = stack.pop();
                        Object inst = stack.peek();
                        if (inst instanceof ReduceResult) {
                            ((ReduceResult) inst).setState(state);
                        }
                        break;

                    case NEWOBJ:
                        Object nargs = stack.pop();
                        Object ncls = stack.pop();
                        stack.push(new ReduceResult(ncls, nargs));
                        break;

                    case NEWOBJ_EX:
                        Object kwargs = stack.pop();
                        Object neargs = stack.pop();
                        Object necls = stack.pop();
                        stack.push(new ReduceResult(necls, Arrays.asList(neargs, kwargs)));
                        break;

                    case BINGET:
                        int idx = dis.readUnsignedByte();
                        stack.push(memo.get(idx));
                        break;

                    case LONG_BINGET:
                        int lidx = readInt();
                        stack.push(memo.get(lidx));
                        break;

                    case BINPUT:
                        int pidx = dis.readUnsignedByte();
                        ensureMemoSize(pidx + 1);
                        memo.set(pidx, stack.peek());
                        break;

                    case LONG_BINPUT:
                        int lpidx = readInt();
                        ensureMemoSize(lpidx + 1);
                        memo.set(lpidx, stack.peek());
                        break;

                    case MEMOIZE:
                        memo.add(stack.peek());
                        break;

                    case BINPERSID:
                        Object pid = stack.pop();
                        stack.push(new PersistentId(pid));
                        break;

                    default:
                        log.warn("Unknown pickle opcode: 0x{}", Integer.toHexString(opcode & 0xFF));
                        break;
                }
            }
        }

        private int readInt() throws IOException {
            byte[] bytes = new byte[4];
            dis.readFully(bytes);
            return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
        }

        private int readShort() throws IOException {
            byte[] bytes = new byte[2];
            dis.readFully(bytes);
            return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getShort() & 0xFFFF;
        }

        private long readLong() throws IOException {
            byte[] bytes = new byte[8];
            dis.readFully(bytes);
            return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
        }

        private double readDouble() throws IOException {
            byte[] bytes = new byte[8];
            dis.readFully(bytes);
            return ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).getDouble();
        }

        private long readLongBytes(int len) throws IOException {
            if (len == 0) return 0;
            byte[] bytes = new byte[len];
            dis.readFully(bytes);
            long value = 0;
            for (int i = len - 1; i >= 0; i--) {
                value = (value << 8) | (bytes[i] & 0xFF);
            }
            // Handle sign extension
            if ((bytes[len - 1] & 0x80) != 0) {
                value |= (-1L << (len * 8));
            }
            return value;
        }

        private String readLine() throws IOException {
            StringBuilder sb = new StringBuilder();
            int b;
            while ((b = dis.read()) != -1 && b != '\n') {
                sb.append((char) b);
            }
            return sb.toString().trim();
        }

        private List<Object> popMark() {
            int markIdx = marks.isEmpty() ? 0 : marks.pop();
            List<Object> items = new ArrayList<>();
            while (stack.size() > markIdx) {
                items.add(0, stack.pop());
            }
            return items;
        }

        @SuppressWarnings("unchecked")
        private void handleDict() {
            List<Object> items = popMark();
            Map<Object, Object> dict = new LinkedHashMap<>();
            for (int i = 0; i < items.size(); i += 2) {
                dict.put(items.get(i), items.get(i + 1));
            }
            stack.push(dict);
        }

        @SuppressWarnings("unchecked")
        private void handleAppend() {
            Object item = stack.pop();
            Object list = stack.peek();
            if (list instanceof List) {
                ((List<Object>) list).add(item);
            }
        }

        @SuppressWarnings("unchecked")
        private void handleAppends() {
            List<Object> items = popMark();
            Object list = stack.peek();
            if (list instanceof List) {
                ((List<Object>) list).addAll(items);
            }
        }

        @SuppressWarnings("unchecked")
        private void handleSetItem() {
            Object value = stack.pop();
            Object key = stack.pop();
            Object dict = stack.peek();
            if (dict instanceof Map) {
                ((Map<Object, Object>) dict).put(key, value);
            }
        }

        @SuppressWarnings("unchecked")
        private void handleSetItems() {
            List<Object> items = popMark();
            Object dict = stack.peek();
            if (dict instanceof Map) {
                for (int i = 0; i < items.size(); i += 2) {
                    ((Map<Object, Object>) dict).put(items.get(i), items.get(i + 1));
                }
            }
        }

        @SuppressWarnings("unchecked")
        private void handleAddItems() {
            List<Object> items = popMark();
            Object set = stack.peek();
            if (set instanceof Set) {
                ((Set<Object>) set).addAll(items);
            }
        }

        private void ensureMemoSize(int size) {
            while (memo.size() < size) {
                memo.add(null);
            }
        }
    }

    /**
     * Represents a Python global reference.
     */
    public static class GlobalRef {
        public final String module;
        public final String name;

        GlobalRef(String module, String name) {
            this.module = module;
            this.name = name;
        }

        @Override
        public String toString() {
            return module + "." + name;
        }
    }

    /**
     * Represents a REDUCE operation result.
     */
    public static class ReduceResult {
        public final Object callable;
        public final Object args;
        private Object state;

        ReduceResult(Object callable, Object args) {
            this.callable = callable;
            this.args = args;
        }

        public void setState(Object state) {
            this.state = state;
        }

        public Object getState() {
            return state;
        }

        @Override
        public String toString() {
            return callable + "(" + args + ")";
        }
    }

    /**
     * Represents a persistent ID reference.
     */
    public static class PersistentId {
        public final Object id;

        PersistentId(Object id) {
            this.id = id;
        }

        @Override
        public String toString() {
            return "PersistentId(" + id + ")";
        }
    }
}
