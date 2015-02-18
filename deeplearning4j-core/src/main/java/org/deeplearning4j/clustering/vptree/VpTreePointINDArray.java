/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.clustering.vptree;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.base.Preconditions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.distancefunction.DistanceFunction;
import org.nd4j.linalg.distancefunction.EuclideanDistance;

/**
 * VPTree point for ndarrays
 *
 * @author Adam Gibson
 */
public class VpTreePointINDArray implements VpTreePoint<VpTreePointINDArray> {
  private INDArray data;
  private DistanceFunction distanceFunction;
  private int index = 0;


  /**
   * Takes in the data. This will throw an IllegalArgumentException if the data isn't a vector
   *
   * @param data the data
   */
  public VpTreePointINDArray(INDArray data, int index) {
    if (!data.isVector())
      throw new IllegalArgumentException("Data should only be a vector");
    this.data = data;
    distanceFunction = new EuclideanDistance(data);
    this.index = index;
  }


  /**
   * Takes in the data. This will throw an IllegalArgumentException if the data isn't a vector
   *
   * @param data the data
   */
  public VpTreePointINDArray(INDArray data) {
    if (!data.isVector())
      throw new IllegalArgumentException("Data should only be a vector");
    this.data = data;
    distanceFunction = new EuclideanDistance(data);
  }

  @Override
  public double distance(VpTreePointINDArray p) {
    Preconditions.checkArgument(p != null);
    return distanceFunction.apply(p.data);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof VpTreePointINDArray)) return false;

    VpTreePointINDArray that = (VpTreePointINDArray) o;

    return !(data != null ? !data.equals(that.data) : that.data != null);
  }


  /**
   * Builds a list of data points from a matrix
   *
   * @param matrix the matrix to build from
   * @return the data points from this matrix
   */
  public static List<VpTreePointINDArray> dataPoints(INDArray matrix) {
    final List<VpTreePointINDArray> ret = new ArrayList<>();
    final AtomicInteger count = new AtomicInteger(0);
    matrix.iterateOverAllRows(new SliceOp() {
      @Override
      public void operate(INDArray nd) {
        ret.add(new VpTreePointINDArray(nd, count.get()));
        count.incrementAndGet();
      }
    });

    return ret;
  }

  @Override
  public int hashCode() {
    int result = data != null ? data.hashCode() : 0;
    result = 31 * result + (distanceFunction != null ? distanceFunction.hashCode() : 0);
    return result;
  }

  @Override
  public String toString() {
    return data != null ? data.toString() : "[]";
  }

  public INDArray getData() {
    return data;
  }

  public void setData(INDArray data) {
    this.data = data;
  }

  public DistanceFunction getDistanceFunction() {
    return distanceFunction;
  }

  public void setDistanceFunction(DistanceFunction distanceFunction) {
    this.distanceFunction = distanceFunction;
  }

  public int getIndex() {
    return index;
  }

  public void setIndex(int index) {
    this.index = index;
  }
}
