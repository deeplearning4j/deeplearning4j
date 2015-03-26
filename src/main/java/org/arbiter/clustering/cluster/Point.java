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

package org.arbiter.clustering.cluster;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;

public class Point implements INDArray {

	private static final long	serialVersionUID	= -6658028541426027226L;

	private String				id					= UUID.randomUUID().toString();
	private String				label;
	private INDArray			array;

	public Point(INDArray array) {
		super();
		this.array = array;
	}

	public Point(String id, INDArray array) {
		super();
		this.id = id;
		this.array = array;
	}

	public Point(String id, String label, double[] data) {
		this(id, label, Nd4j.create(data));
	}
	
	public Point(String id, String label, INDArray array) {
		super();
		this.id = id;
		this.label = label;
		this.array = array;
	}
	
	

	public static List<Point> toPoints(List<INDArray> vectors) {
		List<Point> points = new ArrayList<>();
		for (INDArray vector : vectors)
			points.add(new Point(vector));
		return points;
	}

	public String getId() {
		return id;
	}

	public void setId(String id) {
		this.id = id;
	}

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

	public INDArray getArray() {
		return array;
	}

	public void setArray(INDArray array) {
		this.array = array;
	}

	public void resetLinearView() {
		array.resetLinearView();
	}

	public int secondaryStride() {
		return array.secondaryStride();
	}

	public int majorStride() {
		return array.majorStride();
	}

	public INDArray linearView() {
		return array.linearView();
	}

	public INDArray linearViewColumnOrder() {
		return array.linearViewColumnOrder();
	}

	public int vectorsAlongDimension(int dimension) {
		return array.vectorsAlongDimension(dimension);
	}

	public INDArray vectorAlongDimension(int index, int dimension) {
		return array.vectorAlongDimension(index, dimension);
	}

	public INDArray cumsumi(int dimension) {
		return array.cumsumi(dimension);
	}

	public INDArray cumsum(int dimension) {
		return array.cumsum(dimension);
	}

	public INDArray assign(INDArray arr) {
		return array.assign(arr);
	}

	public INDArray putScalar(int i, double value) {
		return array.putScalar(i, value);
	}

	public INDArray putScalar(int i, float value) {
		return array.putScalar(i, value);
	}

	public INDArray putScalar(int i, int value) {
		return array.putScalar(i, value);
	}

	public INDArray putScalar(int[] i, double value) {
		return array.putScalar(i, value);
	}

	public INDArray lt(Number other) {
		return array.lt(other);
	}

	public INDArray lti(Number other) {
		return array.lti(other);
	}

	public INDArray putScalar(int[] indexes, float value) {
		return array.putScalar(indexes, value);
	}

	public INDArray putScalar(int[] indexes, int value) {
		return array.putScalar(indexes, value);
	}

	public INDArray eps(Number other) {
		return array.eps(other);
	}

	public INDArray epsi(Number other) {
		return array.epsi(other);
	}

	public INDArray eq(Number other) {
		return array.eq(other);
	}

	public INDArray eqi(Number other) {
		return array.eqi(other);
	}

	public INDArray gt(Number other) {
		return array.gt(other);
	}

	public INDArray gti(Number other) {
		return array.gti(other);
	}

	public INDArray lt(INDArray other) {
		return array.lt(other);
	}

	public INDArray lti(INDArray other) {
		return array.lti(other);
	}

	public INDArray eps(INDArray other) {
		return array.eps(other);
	}

	public INDArray epsi(INDArray other) {
		return array.epsi(other);
	}

	public INDArray neq(Number other) {
		return array.neq(other);
	}

	public INDArray neqi(Number other) {
		return array.neqi(other);
	}

	public INDArray neq(INDArray other) {
		return array.neq(other);
	}

	public INDArray neqi(INDArray other) {
		return array.neqi(other);
	}

	public INDArray eq(INDArray other) {
		return array.eq(other);
	}

	public INDArray eqi(INDArray other) {
		return array.eqi(other);
	}

	public INDArray gt(INDArray other) {
		return array.gt(other);
	}

	public INDArray gti(INDArray other) {
		return array.gti(other);
	}

	public INDArray neg() {
		return array.neg();
	}

	public INDArray negi() {
		return array.negi();
	}

	public INDArray rdiv(Number n) {
		return array.rdiv(n);
	}

	public INDArray rdivi(Number n) {
		return array.rdivi(n);
	}

	public INDArray rsub(Number n) {
		return array.rsub(n);
	}

	public INDArray rsubi(Number n) {
		return array.rsubi(n);
	}

	public INDArray div(Number n) {
		return array.div(n);
	}

	public INDArray divi(Number n) {
		return array.divi(n);
	}

	public INDArray mul(Number n) {
		return array.mul(n);
	}

	public INDArray muli(Number n) {
		return array.muli(n);
	}

	public INDArray sub(Number n) {
		return array.sub(n);
	}

	public INDArray subi(Number n) {
		return array.subi(n);
	}

	public INDArray add(Number n) {
		return array.add(n);
	}

	public INDArray addi(Number n) {
		return array.addi(n);
	}

	public INDArray rdiv(Number n, INDArray result) {
		return array.rdiv(n, result);
	}

	public INDArray rdivi(Number n, INDArray result) {
		return array.rdivi(n, result);
	}

	public INDArray rsub(Number n, INDArray result) {
		return array.rsub(n, result);
	}

	public INDArray rsubi(Number n, INDArray result) {
		return array.rsubi(n, result);
	}

	public INDArray div(Number n, INDArray result) {
		return array.div(n, result);
	}

	public INDArray divi(Number n, INDArray result) {
		return array.divi(n, result);
	}

	public INDArray mul(Number n, INDArray result) {
		return array.mul(n, result);
	}

	public INDArray muli(Number n, INDArray result) {
		return array.muli(n, result);
	}

	public INDArray sub(Number n, INDArray result) {
		return array.sub(n, result);
	}

	public INDArray subi(Number n, INDArray result) {
		return array.subi(n, result);
	}

	public INDArray add(Number n, INDArray result) {
		return array.add(n, result);
	}

	public INDArray addi(Number n, INDArray result) {
		return array.addi(n, result);
	}

	public INDArray get(NDArrayIndex... indexes) {
		return array.get(indexes);
	}

	public INDArray getColumns(int[] columns) {
		return array.getColumns(columns);
	}

	public INDArray getRows(int[] rows) {
		return array.getRows(rows);
	}

	public INDArray rdiv(INDArray other) {
		return array.rdiv(other);
	}

	public INDArray rdivi(INDArray other) {
		return array.rdivi(other);
	}

	public INDArray rdiv(INDArray other, INDArray result) {
		return array.rdiv(other, result);
	}

	public INDArray rdivi(INDArray other, INDArray result) {
		return array.rdivi(other, result);
	}

	public INDArray rsub(INDArray other, INDArray result) {
		return array.rsub(other, result);
	}

	public INDArray rsub(INDArray other) {
		return array.rsub(other);
	}

	public INDArray rsubi(INDArray other) {
		return array.rsubi(other);
	}

	public INDArray rsubi(INDArray other, INDArray result) {
		return array.rsubi(other, result);
	}

	public INDArray assign(Number value) {
		return array.assign(value);
	}

	public int linearIndex(int i) {
		return array.linearIndex(i);
	}

	public void iterateOverAllRows(SliceOp op) {
		array.iterateOverAllRows(op);
	}

	public void iterateOverAllColumns(SliceOp op) {
		array.iterateOverAllColumns(op);
	}

	public void checkDimensions(INDArray other) {
		array.checkDimensions(other);
	}

	public int[] endsForSlices() {
		return array.endsForSlices();
	}

	public void sliceVectors(List<INDArray> list) {
		array.sliceVectors(list);
	}


	public INDArray putSlice(int slice, INDArray put) {
		return array.putSlice(slice, put);
	}

	public INDArray cond(Condition condition) {
		return array.cond(condition);
	}

	public INDArray condi(Condition condition) {
		return array.condi(condition);
	}

	public void iterateOverDimension(int dimension, SliceOp op, boolean modify) {
		array.iterateOverDimension(dimension, op, modify);
	}

	public INDArray repmat(int[] shape) {
		return array.repmat(shape);
	}

	public INDArray putRow(int row, INDArray toPut) {
		return array.putRow(row, toPut);
	}

	public INDArray putColumn(int column, INDArray toPut) {
		return array.putColumn(column, toPut);
	}

	public INDArray getScalar(int row, int column) {
		return array.getScalar(row, column);
	}

	public INDArray getScalar(int i) {
		return array.getScalar(i);
	}

	public int index(int row, int column) {
		return array.index(row, column);
	}

	public double squaredDistance(INDArray other) {
		return array.squaredDistance(other);
	}

	public double distance2(INDArray other) {
		return array.distance2(other);
	}

	public double distance1(INDArray other) {
		return array.distance1(other);
	}

	public INDArray put(NDArrayIndex[] indices, INDArray element) {
		return array.put(indices, element);
	}

	public INDArray put(NDArrayIndex[] indices, Number element) {
		return array.put(indices, element);
	}

	public INDArray put(int[] indices, INDArray element) {
		return array.put(indices, element);
	}

	public INDArray put(int i, int j, INDArray element) {
		return array.put(i, j, element);
	}

	public INDArray put(int i, int j, Number element) {
		return array.put(i, j, element);
	}

	public INDArray put(int i, INDArray element) {
		return array.put(i, element);
	}

	public INDArray diviColumnVector(INDArray columnVector) {
		return array.diviColumnVector(columnVector);
	}

	public INDArray divColumnVector(INDArray columnVector) {
		return array.divColumnVector(columnVector);
	}

	public INDArray diviRowVector(INDArray rowVector) {
		return array.diviRowVector(rowVector);
	}

	public INDArray divRowVector(INDArray rowVector) {
		return array.divRowVector(rowVector);
	}

	public INDArray rdiviColumnVector(INDArray columnVector) {
		return array.rdiviColumnVector(columnVector);
	}

	public INDArray rdivColumnVector(INDArray columnVector) {
		return array.rdivColumnVector(columnVector);
	}

	public INDArray rdiviRowVector(INDArray rowVector) {
		return array.rdiviRowVector(rowVector);
	}

	public INDArray rdivRowVector(INDArray rowVector) {
		return array.rdivRowVector(rowVector);
	}

	public INDArray muliColumnVector(INDArray columnVector) {
		return array.muliColumnVector(columnVector);
	}

	public INDArray mulColumnVector(INDArray columnVector) {
		return array.mulColumnVector(columnVector);
	}

	public INDArray muliRowVector(INDArray rowVector) {
		return array.muliRowVector(rowVector);
	}

	public INDArray mulRowVector(INDArray rowVector) {
		return array.mulRowVector(rowVector);
	}

	public INDArray rsubiColumnVector(INDArray columnVector) {
		return array.rsubiColumnVector(columnVector);
	}

	public INDArray rsubColumnVector(INDArray columnVector) {
		return array.rsubColumnVector(columnVector);
	}

	public INDArray rsubiRowVector(INDArray rowVector) {
		return array.rsubiRowVector(rowVector);
	}

	public INDArray rsubRowVector(INDArray rowVector) {
		return array.rsubRowVector(rowVector);
	}

	public INDArray subiColumnVector(INDArray columnVector) {
		return array.subiColumnVector(columnVector);
	}

	public INDArray subColumnVector(INDArray columnVector) {
		return array.subColumnVector(columnVector);
	}

	public INDArray subiRowVector(INDArray rowVector) {
		return array.subiRowVector(rowVector);
	}

	public INDArray subRowVector(INDArray rowVector) {
		return array.subRowVector(rowVector);
	}

	public INDArray addiColumnVector(INDArray columnVector) {
		return array.addiColumnVector(columnVector);
	}

	public INDArray addColumnVector(INDArray columnVector) {
		return array.addColumnVector(columnVector);
	}

	public INDArray addiRowVector(INDArray rowVector) {
		return array.addiRowVector(rowVector);
	}

	public INDArray addRowVector(INDArray rowVector) {
		return array.addRowVector(rowVector);
	}

	public INDArray mmul(INDArray other) {
		return array.mmul(other);
	}

	public INDArray mmul(INDArray other, INDArray result) {
		return array.mmul(other, result);
	}

	public INDArray div(INDArray other) {
		return array.div(other);
	}

	public INDArray div(INDArray other, INDArray result) {
		return array.div(other, result);
	}

	public INDArray mul(INDArray other) {
		return array.mul(other);
	}

	public INDArray mul(INDArray other, INDArray result) {
		return array.mul(other, result);
	}

	public INDArray sub(INDArray other) {
		return array.sub(other);
	}

	public INDArray sub(INDArray other, INDArray result) {
		return array.sub(other, result);
	}

	public INDArray add(INDArray other) {
		return array.add(other);
	}

	public INDArray add(INDArray other, INDArray result) {
		return array.add(other, result);
	}

	public INDArray mmuli(INDArray other) {
		return array.mmuli(other);
	}

	public INDArray mmuli(INDArray other, INDArray result) {
		return array.mmuli(other, result);
	}

	public INDArray divi(INDArray other) {
		return array.divi(other);
	}

	public INDArray divi(INDArray other, INDArray result) {
		return array.divi(other, result);
	}

	public INDArray muli(INDArray other) {
		return array.muli(other);
	}

	public INDArray muli(INDArray other, INDArray result) {
		return array.muli(other, result);
	}

	public INDArray subi(INDArray other) {
		return array.subi(other);
	}

	public INDArray subi(INDArray other, INDArray result) {
		return array.subi(other, result);
	}

	public INDArray addi(INDArray other) {
		return array.addi(other);
	}

	public INDArray addi(INDArray other, INDArray result) {
		return array.addi(other, result);
	}

	public INDArray normmax(int dimension) {
		return array.normmax(dimension);
	}

	public INDArray norm2(int dimension) {
		return array.norm2(dimension);
	}

	public INDArray norm1(int dimension) {
		return array.norm1(dimension);
	}

	public INDArray std(int dimension) {
		return array.std(dimension);
	}

	public INDArray prod(int dimension) {
		return array.prod(dimension);
	}

	public INDArray mean(int dimension) {
		return array.mean(dimension);
	}

	public INDArray var(int dimension) {
		return array.var(dimension);
	}

	public INDArray max(int dimension) {
		return array.max(dimension);
	}

	public INDArray min(int dimension) {
		return array.min(dimension);
	}

	public INDArray sum(int dimension) {
		return array.sum(dimension);
	}

	public void setStride(int[] stride) {
		array.setStride(stride);
	}

	public INDArray subArray(int[] offsets, int[] shape, int[] stride) {
		return array.subArray(offsets, shape, stride);
	}

	public INDArray getScalar(int[] indices) {
		return array.getScalar(indices);
	}

	public int getInt(int... indices) {
		return array.getInt(indices);
	}

	public double getDouble(int... indices) {
		return array.getDouble(indices);
	}

	public float getFloat(int[] indices) {
		return array.getFloat(indices);
	}


	public double getDouble(int i) {
		return array.getDouble(i);
	}

	public double getDouble(int i, int j) {
		return array.getDouble(i, j);
	}

	public float getFloat(int i) {
		return array.getFloat(i);
	}

	public float getFloat(int i, int j) {
		return array.getFloat(i, j);
	}

	public INDArray dup() {
		return array.dup();
	}

	public INDArray ravel() {
		return array.ravel();
	}

	public void setData(DataBuffer data) {
		array.setData(data);
	}

	public int slices() {
		return array.slices();
	}

	public INDArray slice(int i, int dimension) {
		return array.slice(i, dimension);
	}

	public INDArray slice(int i) {
		return array.slice(i);
	}

	public int offset() {
		return array.offset();
	}

	public INDArray reshape(int... newShape) {
		return array.reshape(newShape);
	}

	public INDArray reshape(int rows, int columns) {
		return array.reshape(rows, columns);
	}

	public INDArray transpose() {
		return array.transpose();
	}

	public INDArray transposei() {
		return array.transposei();
	}

	public INDArray swapAxes(int dimension, int with) {
		return array.swapAxes(dimension, with);
	}

	public INDArray permute(int... rearrange) {
		return array.permute(rearrange);
	}

	public INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable) {
		return array.dimShuffle(rearrange, newOrder, broadCastable);
	}

	public INDArray getColumn(int i) {
		return array.getColumn(i);
	}

	public INDArray getRow(int i) {
		return array.getRow(i);
	}

	public int columns() {
		return array.columns();
	}

	public int rows() {
		return array.rows();
	}

	public boolean isColumnVector() {
		return array.isColumnVector();
	}

	public boolean isRowVector() {
		return array.isRowVector();
	}

	public boolean isVector() {
		return array.isVector();
	}

	public boolean isSquare() {
		return array.isSquare();
	}

	public boolean isMatrix() {
		return array.isMatrix();
	}

	public boolean isScalar() {
		return array.isScalar();
	}

	public int[] shape() {
		return array.shape();
	}

	public int[] stride() {
		return array.stride();
	}

	public char ordering() {
		return array.ordering();
	}

	public int size(int dimension) {
		return array.size(dimension);
	}

	public int length() {
		return array.length();
	}

	public INDArray broadcast(int... shape) {
		return array.broadcast(shape);
	}

	public Object element() {
		return array.element();
	}

	public DataBuffer data() {
		return array.data();
	}

	public void setData(float[] data) {
		array.setData(data);
	}

	public IComplexNDArray rdiv(IComplexNumber n) {
		return array.rdiv(n);
	}

	public IComplexNDArray rdivi(IComplexNumber n) {
		return array.rdivi(n);
	}

	public IComplexNDArray rsub(IComplexNumber n) {
		return array.rsub(n);
	}

	public IComplexNDArray rsubi(IComplexNumber n) {
		return array.rsubi(n);
	}

	public IComplexNDArray div(IComplexNumber n) {
		return array.div(n);
	}

	public IComplexNDArray divi(IComplexNumber n) {
		return array.divi(n);
	}

	public IComplexNDArray mul(IComplexNumber n) {
		return array.mul(n);
	}

	public IComplexNDArray muli(IComplexNumber n) {
		return array.muli(n);
	}

	public IComplexNDArray sub(IComplexNumber n) {
		return array.sub(n);
	}

	public IComplexNDArray subi(IComplexNumber n) {
		return array.subi(n);
	}

	public IComplexNDArray add(IComplexNumber n) {
		return array.add(n);
	}

	public IComplexNDArray addi(IComplexNumber n) {
		return array.addi(n);
	}

	public IComplexNDArray rdiv(IComplexNumber n, IComplexNDArray result) {
		return array.rdiv(n, result);
	}

	public IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result) {
		return array.rdivi(n, result);
	}

	public IComplexNDArray rsub(IComplexNumber n, IComplexNDArray result) {
		return array.rsub(n, result);
	}

	public IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result) {
		return array.rsubi(n, result);
	}

	public IComplexNDArray div(IComplexNumber n, IComplexNDArray result) {
		return array.div(n, result);
	}

	public IComplexNDArray divi(IComplexNumber n, IComplexNDArray result) {
		return array.divi(n, result);
	}

	public IComplexNDArray mul(IComplexNumber n, IComplexNDArray result) {
		return array.mul(n, result);
	}

	public IComplexNDArray muli(IComplexNumber n, IComplexNDArray result) {
		return array.muli(n, result);
	}

	public IComplexNDArray sub(IComplexNumber n, IComplexNDArray result) {
		return array.sub(n, result);
	}

	public IComplexNDArray subi(IComplexNumber n, IComplexNDArray result) {
		return array.subi(n, result);
	}

	public IComplexNDArray add(IComplexNumber n, IComplexNDArray result) {
		return array.add(n, result);
	}

	public IComplexNDArray addi(IComplexNumber n, IComplexNDArray result) {
		return array.addi(n, result);
	}

}
