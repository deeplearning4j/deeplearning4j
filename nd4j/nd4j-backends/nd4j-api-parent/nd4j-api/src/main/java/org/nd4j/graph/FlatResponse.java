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
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class FlatResponse extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static FlatResponse getRootAsFlatResponse(ByteBuffer _bb) { return getRootAsFlatResponse(_bb, new FlatResponse()); }
  public static FlatResponse getRootAsFlatResponse(ByteBuffer _bb, FlatResponse obj) { _bb.order(java.nio.ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public FlatResponse __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int status() { int o = __offset(4); return o != 0 ? bb.getInt(o + bb_pos) : 0; }

  public static int createFlatResponse(FlatBufferBuilder builder,
      int status) {
    builder.startTable(1);
    FlatResponse.addStatus(builder, status);
    return FlatResponse.endFlatResponse(builder);
  }

  public static void startFlatResponse(FlatBufferBuilder builder) { builder.startTable(1); }
  public static void addStatus(FlatBufferBuilder builder, int status) { builder.addInt(0, status, 0); }
  public static int endFlatResponse(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public FlatResponse get(int j) { return get(new FlatResponse(), j); }
    public FlatResponse get(FlatResponse obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

