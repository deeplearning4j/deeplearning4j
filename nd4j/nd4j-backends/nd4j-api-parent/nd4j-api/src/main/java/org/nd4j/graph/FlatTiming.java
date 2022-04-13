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
public final class FlatTiming extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static FlatTiming getRootAsFlatTiming(ByteBuffer _bb) { return getRootAsFlatTiming(_bb, new FlatTiming()); }
  public static FlatTiming getRootAsFlatTiming(ByteBuffer _bb, FlatTiming obj) { _bb.order(java.nio.ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public FlatTiming __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int id() { int o = __offset(4); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public String name() { int o = __offset(6); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer nameAsByteBuffer() { return __vector_as_bytebuffer(6, 1); }
  public ByteBuffer nameInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 6, 1); }
  public org.nd4j.graph.LongPair timing() { return timing(new org.nd4j.graph.LongPair()); }
  public org.nd4j.graph.LongPair timing(org.nd4j.graph.LongPair obj) { int o = __offset(8); return o != 0 ? obj.__assign(__indirect(o + bb_pos), bb) : null; }

  public static int createFlatTiming(FlatBufferBuilder builder,
      int id,
      int nameOffset,
      int timingOffset) {
    builder.startTable(3);
    FlatTiming.addTiming(builder, timingOffset);
    FlatTiming.addName(builder, nameOffset);
    FlatTiming.addId(builder, id);
    return FlatTiming.endFlatTiming(builder);
  }

  public static void startFlatTiming(FlatBufferBuilder builder) { builder.startTable(3); }
  public static void addId(FlatBufferBuilder builder, int id) { builder.addInt(0, id, 0); }
  public static void addName(FlatBufferBuilder builder, int nameOffset) { builder.addOffset(1, nameOffset, 0); }
  public static void addTiming(FlatBufferBuilder builder, int timingOffset) { builder.addOffset(2, timingOffset, 0); }
  public static int endFlatTiming(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public FlatTiming get(int j) { return get(new FlatTiming(), j); }
    public FlatTiming get(FlatTiming obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

