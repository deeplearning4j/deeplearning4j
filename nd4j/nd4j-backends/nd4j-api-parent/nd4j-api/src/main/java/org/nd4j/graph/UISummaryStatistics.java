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
public final class UISummaryStatistics extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static UISummaryStatistics getRootAsUISummaryStatistics(ByteBuffer _bb) { return getRootAsUISummaryStatistics(_bb, new UISummaryStatistics()); }
  public static UISummaryStatistics getRootAsUISummaryStatistics(ByteBuffer _bb, UISummaryStatistics obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public UISummaryStatistics __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public long bitmask() { int o = __offset(4); return o != 0 ? (long)bb.getInt(o + bb_pos) & 0xFFFFFFFFL : 0L; }
  public org.nd4j.graph.FlatArray min() { return min(new org.nd4j.graph.FlatArray()); }
  public org.nd4j.graph.FlatArray min(org.nd4j.graph.FlatArray obj) { int o = __offset(6); return o != 0 ? obj.__assign(__indirect(o + bb_pos), bb) : null; }
  public org.nd4j.graph.FlatArray max() { return max(new org.nd4j.graph.FlatArray()); }
  public org.nd4j.graph.FlatArray max(org.nd4j.graph.FlatArray obj) { int o = __offset(8); return o != 0 ? obj.__assign(__indirect(o + bb_pos), bb) : null; }
  public double mean() { int o = __offset(10); return o != 0 ? bb.getDouble(o + bb_pos) : 0.0; }
  public double stdev() { int o = __offset(12); return o != 0 ? bb.getDouble(o + bb_pos) : 0.0; }
  public long countzero() { int o = __offset(14); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public long countpositive() { int o = __offset(16); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public long countnegative() { int o = __offset(18); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public long countnan() { int o = __offset(20); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public long countinf() { int o = __offset(22); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }

  public static int createUISummaryStatistics(FlatBufferBuilder builder,
      long bitmask,
      int minOffset,
      int maxOffset,
      double mean,
      double stdev,
      long countzero,
      long countpositive,
      long countnegative,
      long countnan,
      long countinf) {
    builder.startTable(10);
    UISummaryStatistics.addCountinf(builder, countinf);
    UISummaryStatistics.addCountnan(builder, countnan);
    UISummaryStatistics.addCountnegative(builder, countnegative);
    UISummaryStatistics.addCountpositive(builder, countpositive);
    UISummaryStatistics.addCountzero(builder, countzero);
    UISummaryStatistics.addStdev(builder, stdev);
    UISummaryStatistics.addMean(builder, mean);
    UISummaryStatistics.addMax(builder, maxOffset);
    UISummaryStatistics.addMin(builder, minOffset);
    UISummaryStatistics.addBitmask(builder, bitmask);
    return UISummaryStatistics.endUISummaryStatistics(builder);
  }

  public static void startUISummaryStatistics(FlatBufferBuilder builder) { builder.startTable(10); }
  public static void addBitmask(FlatBufferBuilder builder, long bitmask) { builder.addInt(0, (int)bitmask, (int)0L); }
  public static void addMin(FlatBufferBuilder builder, int minOffset) { builder.addOffset(1, minOffset, 0); }
  public static void addMax(FlatBufferBuilder builder, int maxOffset) { builder.addOffset(2, maxOffset, 0); }
  public static void addMean(FlatBufferBuilder builder, double mean) { builder.addDouble(3, mean, 0.0); }
  public static void addStdev(FlatBufferBuilder builder, double stdev) { builder.addDouble(4, stdev, 0.0); }
  public static void addCountzero(FlatBufferBuilder builder, long countzero) { builder.addLong(5, countzero, 0L); }
  public static void addCountpositive(FlatBufferBuilder builder, long countpositive) { builder.addLong(6, countpositive, 0L); }
  public static void addCountnegative(FlatBufferBuilder builder, long countnegative) { builder.addLong(7, countnegative, 0L); }
  public static void addCountnan(FlatBufferBuilder builder, long countnan) { builder.addLong(8, countnan, 0L); }
  public static void addCountinf(FlatBufferBuilder builder, long countinf) { builder.addLong(9, countinf, 0L); }
  public static int endUISummaryStatistics(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public UISummaryStatistics get(int j) { return get(new UISummaryStatistics(), j); }
    public UISummaryStatistics get(UISummaryStatistics obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}

