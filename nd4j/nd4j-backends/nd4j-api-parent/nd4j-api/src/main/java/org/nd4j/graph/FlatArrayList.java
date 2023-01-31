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
public final class FlatArrayList extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static FlatArrayList getRootAsFlatArrayList(ByteBuffer _bb) { return getRootAsFlatArrayList(_bb, new FlatArrayList()); }
  public static FlatArrayList getRootAsFlatArrayList(ByteBuffer _bb, FlatArrayList obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public FlatArrayList __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public org.nd4j.graph.FlatArray list(int j) { return list(new org.nd4j.graph.FlatArray(), j); }
  public org.nd4j.graph.FlatArray list(org.nd4j.graph.FlatArray obj, int j) { int o = __offset(4); return o != 0 ? obj.__assign(__indirect(__vector(o) + j * 4), bb) : null; }
  public int listLength() { int o = __offset(4); return o != 0 ? __vector_len(o) : 0; }
  public org.nd4j.graph.FlatArray.Vector listVector() { return listVector(new org.nd4j.graph.FlatArray.Vector()); }
  public org.nd4j.graph.FlatArray.Vector listVector(org.nd4j.graph.FlatArray.Vector obj) { int o = __offset(4); return o != 0 ? obj.__assign(__vector(o), 4, bb) : null; }

  public static int createFlatArrayList(FlatBufferBuilder builder,
      int listOffset) {
    builder.startTable(1);
    FlatArrayList.addList(builder, listOffset);
    return FlatArrayList.endFlatArrayList(builder);
  }

  public static void startFlatArrayList(FlatBufferBuilder builder) { builder.startTable(1); }
  public static void addList(FlatBufferBuilder builder, int listOffset) { builder.addOffset(0, listOffset, 0); }
  public static int createListVector(FlatBufferBuilder builder, int[] data) { builder.startVector(4, data.length, 4); for (int i = data.length - 1; i >= 0; i--) builder.addOffset(data[i]); return builder.endVector(); }
  public static void startListVector(FlatBufferBuilder builder, int numElems) { builder.startVector(4, numElems, 4); }
  public static int endFlatArrayList(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public FlatArrayList get(int j) { return get(new FlatArrayList(), j); }
    public FlatArrayList get(FlatArrayList obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

