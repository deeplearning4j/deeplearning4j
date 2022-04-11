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
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class UIHistogram extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static UIHistogram getRootAsUIHistogram(ByteBuffer _bb) { return getRootAsUIHistogram(_bb, new UIHistogram()); }
  public static UIHistogram getRootAsUIHistogram(ByteBuffer _bb, UIHistogram obj) { _bb.order(java.nio.ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public UIHistogram __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public byte type() { int o = __offset(4); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public long numbins() { int o = __offset(6); return o != 0 ? (long)bb.getInt(o + bb_pos) & 0xFFFFFFFFL : 0L; }
  public org.nd4j.graph.FlatArray binranges() { return binranges(new org.nd4j.graph.FlatArray()); }
  public org.nd4j.graph.FlatArray binranges(org.nd4j.graph.FlatArray obj) { int o = __offset(8); return o != 0 ? obj.__assign(__indirect(o + bb_pos), bb) : null; }
  public org.nd4j.graph.FlatArray y() { return y(new org.nd4j.graph.FlatArray()); }
  public org.nd4j.graph.FlatArray y(org.nd4j.graph.FlatArray obj) { int o = __offset(10); return o != 0 ? obj.__assign(__indirect(o + bb_pos), bb) : null; }
  public String binlabels(int j) { int o = __offset(12); return o != 0 ? __string(__vector(o) + j * 4) : null; }
  public int binlabelsLength() { int o = __offset(12); return o != 0 ? __vector_len(o) : 0; }
  public StringVector binlabelsVector() { return binlabelsVector(new StringVector()); }
  public StringVector binlabelsVector(StringVector obj) { int o = __offset(12); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }

  public static int createUIHistogram(FlatBufferBuilder builder,
      byte type,
      long numbins,
      int binrangesOffset,
      int yOffset,
      int binlabelsOffset) {
    builder.startTable(5);
    UIHistogram.addBinlabels(builder, binlabelsOffset);
    UIHistogram.addY(builder, yOffset);
    UIHistogram.addBinranges(builder, binrangesOffset);
    UIHistogram.addNumbins(builder, numbins);
    UIHistogram.addType(builder, type);
    return UIHistogram.endUIHistogram(builder);
  }

  public static void startUIHistogram(FlatBufferBuilder builder) { builder.startTable(5); }
  public static void addType(FlatBufferBuilder builder, byte type) { builder.addByte(0, type, 0); }
  public static void addNumbins(FlatBufferBuilder builder, long numbins) { builder.addInt(1, (int)numbins, (int)0L); }
  public static void addBinranges(FlatBufferBuilder builder, int binrangesOffset) { builder.addOffset(2, binrangesOffset, 0); }
  public static void addY(FlatBufferBuilder builder, int yOffset) { builder.addOffset(3, yOffset, 0); }
  public static void addBinlabels(FlatBufferBuilder builder, int binlabelsOffset) { builder.addOffset(4, binlabelsOffset, 0); }
  public static int createBinlabelsVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startBinlabelsVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static int endUIHistogram(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public UIHistogram get(int j) { return get(new UIHistogram(), j); }
    public UIHistogram get(UIHistogram obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

