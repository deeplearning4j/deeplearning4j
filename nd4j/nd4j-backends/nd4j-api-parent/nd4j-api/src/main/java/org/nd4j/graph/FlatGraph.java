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

package org.nd4j.graph;

import java.nio.*;
import java.lang.*;
import java.nio.ByteOrder;

import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class FlatGraph extends Table {
  public static FlatGraph getRootAsFlatGraph(ByteBuffer _bb) { return getRootAsFlatGraph(_bb, new FlatGraph()); }
  public static FlatGraph getRootAsFlatGraph(ByteBuffer _bb, FlatGraph obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; }
  public FlatGraph __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public long id() { int o = __offset(4); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public FlatVariable variables(int j) { return variables(new FlatVariable(), j); }
  public FlatVariable variables(FlatVariable obj, int j) { int o = __offset(6); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int variablesLength() { int o = __offset(6); return o != 0 ? __vector_len(o) : 0; }
  public FlatNode nodes(int j) { return nodes(new FlatNode(), j); }
  public FlatNode nodes(FlatNode obj, int j) { int o = __offset(8); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int nodesLength() { int o = __offset(8); return o != 0 ? __vector_len(o) : 0; }
  public IntPair outputs(int j) { return outputs(new IntPair(), j); }
  public IntPair outputs(IntPair obj, int j) { int o = __offset(10); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int outputsLength() { int o = __offset(10); return o != 0 ? __vector_len(o) : 0; }
  public FlatConfiguration configuration() { return configuration(new FlatConfiguration()); }
  public FlatConfiguration configuration(FlatConfiguration obj) { int o = __offset(12); return o != 0 ? obj.__assign(__indirect(o + bb_pos), bb) : null; }
  public String placeholders(int j) { int o = __offset(14); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int placeholdersLength() { int o = __offset(14); return o != 0 ? __vector_len(o) : 0; }
  public String lossVariables(int j) { int o = __offset(16); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int lossVariablesLength() { int o = __offset(16); return o != 0 ? __vector_len(o) : 0; }
  public String trainingConfig() { int o = __offset(18); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer trainingConfigAsByteBuffer() { return __vector_as_bytebuffer(18, 1); }
  public ByteBuffer trainingConfigInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 18, 1); }
  public UpdaterState updaterState(int j) { return updaterState(new UpdaterState(), j); }
  public UpdaterState updaterState(UpdaterState obj, int j) { int o = __offset(20); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int updaterStateLength() { int o = __offset(20); return o != 0 ? __vector_len(o) : 0; }

  public static int createFlatGraph(FlatBufferBuilder builder,
      long id,
      int variablesOffset,
      int nodesOffset,
      int outputsOffset,
      int configurationOffset,
      int placeholdersOffset,
      int lossVariablesOffset,
      int trainingConfigOffset,
      int updaterStateOffset) {
    builder.startObject(9);
    FlatGraph.addId(builder, id);
    FlatGraph.addUpdaterState(builder, updaterStateOffset);
    FlatGraph.addTrainingConfig(builder, trainingConfigOffset);
    FlatGraph.addLossVariables(builder, lossVariablesOffset);
    FlatGraph.addPlaceholders(builder, placeholdersOffset);
    FlatGraph.addConfiguration(builder, configurationOffset);
    FlatGraph.addOutputs(builder, outputsOffset);
    FlatGraph.addNodes(builder, nodesOffset);
    FlatGraph.addVariables(builder, variablesOffset);
    return FlatGraph.endFlatGraph(builder);
  }

  public static void startFlatGraph(FlatBufferBuilder builder) { builder.startObject(9); }
  public static void addId(FlatBufferBuilder builder, long id) { builder.addLong(0, id, 0L); }
  public static void addVariables(FlatBufferBuilder builder, int variablesOffset) { builder.addOffset(1, variablesOffset, 0); }
  public static int createVariablesVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startVariablesVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addNodes(FlatBufferBuilder builder, int nodesOffset) { builder.addOffset(2, nodesOffset, 0); }
  public static int createNodesVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startNodesVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addOutputs(FlatBufferBuilder builder, int outputsOffset) { builder.addOffset(3, outputsOffset, 0); }
  public static int createOutputsVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startOutputsVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addConfiguration(FlatBufferBuilder builder, int configurationOffset) { builder.addOffset(4, configurationOffset, 0); }
  public static void addPlaceholders(FlatBufferBuilder builder, int placeholdersOffset) { builder.addOffset(5, placeholdersOffset, 0); }
  public static int createPlaceholdersVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startPlaceholdersVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addLossVariables(FlatBufferBuilder builder, int lossVariablesOffset) { builder.addOffset(6, lossVariablesOffset, 0); }
  public static int createLossVariablesVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startLossVariablesVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static void addTrainingConfig(FlatBufferBuilder builder, int trainingConfigOffset) { builder.addOffset(7, trainingConfigOffset, 0); }
  public static void addUpdaterState(FlatBufferBuilder builder, int updaterStateOffset) { builder.addOffset(8, updaterStateOffset, 0); }
  public static int createUpdaterStateVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startUpdaterStateVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static int endFlatGraph(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
  public static void finishFlatGraphBuffer(FlatBufferBuilder builder, int offset) { builder.finish(offset); }
  public static void finishSizePrefixedFlatGraphBuffer(FlatBufferBuilder builder, int offset) { builder.finishSizePrefixed(offset); }
}

