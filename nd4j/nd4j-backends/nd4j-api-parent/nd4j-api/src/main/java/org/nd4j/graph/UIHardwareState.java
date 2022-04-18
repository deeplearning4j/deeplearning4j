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
public final class UIHardwareState extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static UIHardwareState getRootAsUIHardwareState(ByteBuffer _bb) { return getRootAsUIHardwareState(_bb, new UIHardwareState()); }
  public static UIHardwareState getRootAsUIHardwareState(ByteBuffer _bb, UIHardwareState obj) { _bb.order(java.nio.ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public UIHardwareState __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public long gpuMemory(int j) { int o = __offset(4); return o != 0 ? bb.getLong(__vector(o) + j * 8) : 0; }
  public int gpuMemoryLength() { int o = __offset(4); return o != 0 ? __vector_len(o) : 0; }
  public LongVector gpuMemoryVector() { return gpuMemoryVector(new LongVector()); }
  public LongVector gpuMemoryVector(LongVector obj) { int o = __offset(4); return o != 0 ? obj.__assign(__vector(o), bb) : null; }
  public ByteBuffer gpuMemoryAsByteBuffer() { return __vector_as_bytebuffer(4, 8); }
  public ByteBuffer gpuMemoryInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 4, 8); }
  public long hostMemory() { int o = __offset(6); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }

  public static int createUIHardwareState(FlatBufferBuilder builder,
      int gpuMemoryOffset,
      long hostMemory) {
    builder.startTable(2);
    UIHardwareState.addHostMemory(builder, hostMemory);
    UIHardwareState.addGpuMemory(builder, gpuMemoryOffset);
    return UIHardwareState.endUIHardwareState(builder);
  }

  public static void startUIHardwareState(FlatBufferBuilder builder) { builder.startTable(2); }
  public static void addGpuMemory(FlatBufferBuilder builder, int gpuMemoryOffset) { builder.addOffset(0, gpuMemoryOffset, 0); }
  public static int createGpuMemoryVector(FlatBufferBuilder builder, long[] data) { builder.startVector(8, data.length, 8); for (int i = data.length - 1; i >= 0; i--) builder.addLong(data[i]); return builder.endVector(); }
  public static void startGpuMemoryVector(FlatBufferBuilder builder, int numElems) { builder.startVector(8, numElems, 8); }
  public static void addHostMemory(FlatBufferBuilder builder, long hostMemory) { builder.addLong(1, hostMemory, 0L); }
  public static int endUIHardwareState(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public UIHardwareState get(int j) { return get(new UIHardwareState(), j); }
    public UIHardwareState get(UIHardwareState obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

