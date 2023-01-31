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
public final class UISystemInfo extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static UISystemInfo getRootAsUISystemInfo(ByteBuffer _bb) { return getRootAsUISystemInfo(_bb, new UISystemInfo()); }
  public static UISystemInfo getRootAsUISystemInfo(ByteBuffer _bb, UISystemInfo obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public UISystemInfo __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int physicalCores() { int o = __offset(4); return o != 0 ? bb.getInt(o + bb_pos) : 0; }

  public static int createUISystemInfo(FlatBufferBuilder builder,
      int physicalCores) {
    builder.startTable(1);
    UISystemInfo.addPhysicalCores(builder, physicalCores);
    return UISystemInfo.endUISystemInfo(builder);
  }

  public static void startUISystemInfo(FlatBufferBuilder builder) { builder.startTable(1); }
  public static void addPhysicalCores(FlatBufferBuilder builder, int physicalCores) { builder.addInt(0, physicalCores, 0); }
  public static int endUISystemInfo(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public UISystemInfo get(int j) { return get(new UISystemInfo(), j); }
    public UISystemInfo get(UISystemInfo obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

