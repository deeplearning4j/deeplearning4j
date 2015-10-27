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

package org.deeplearning4j.spark.sql.types

import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericMutableRow
import org.apache.spark.sql.types._
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


/**
 * A multidimensional array structure.
 * @param array initialization data
 */
@Experimental
@SQLUserDefinedType(udt = classOf[TensorUDT])
case class Tensor(val array: INDArray) {

  def toArray: INDArray = array

  override def toString: String = array.shape.mkString("[", "x", "]")
}

/**
 * Tensor UDT for dataframe/SQL interoperability.
 *
 * @author Eron Wright
 */
@DeveloperApi
class TensorUDT extends UserDefinedType[Tensor] {
  override def sqlType: StructType = {
    StructType(Seq(
      StructField("shape", ArrayType(IntegerType, containsNull = false), nullable = false),
      StructField("stride", ArrayType(IntegerType, containsNull = false), nullable = false),
      StructField("type", ByteType, nullable = false),
      StructField("doubles", ArrayType(DoubleType, containsNull = false), nullable = true),
      StructField("floats", ArrayType(FloatType, containsNull = false), nullable = true),
      StructField("ints", ArrayType(IntegerType, containsNull = false), nullable = true)))
  }

  override def serialize(obj: Any): Row = {

    val row = new GenericMutableRow(1)
    obj match {
      case Tensor(values) =>
        row.update(0, values.shape())
        row.update(1, values.stride())
        val buf = values.data()
        buf.dataType() match {
          case DataBuffer.Type.DOUBLE =>
            row.setByte(2, 0)
            row.update(3, buf.asDouble())
            row.setNullAt(4)
            row.setNullAt(5)
          case DataBuffer.Type.FLOAT =>
            row.setByte(2, 1)
            row.setNullAt(3)
            row.update(4, buf.asFloat())
            row.setNullAt(5)
          case DataBuffer.Type.INT =>
            row.setByte(2, 2)
            row.setNullAt(3)
            row.setNullAt(4)
            row.update(5, buf.asInt())
        }
    }
    Row(row)
  }

  override def deserialize(datum: Any): Tensor = {
    datum match {
      case tensor: Tensor =>
        tensor
      case row: Row =>
        require(row.length == 6, s"TensorUDT.deserialize given row with length ${row.length} but requires length == 6")
        val shape = row.getAs[Iterable[Int]](0).toArray
        val stride = row.getAs[Iterable[Int]](1).toArray
        val tpe = row.getByte(2)
        val buf: DataBuffer = tpe match {
          case 0 => Nd4j.createBuffer(row.getAs[Iterable[Double]](3).toArray)
          case 1 => Nd4j.createBuffer(row.getAs[Iterable[Float]](4).toArray)
          case 2 => Nd4j.createBuffer(row.getAs[Iterable[Int]](5).toArray)
        }
        val values = Nd4j.create(buf, shape, stride, 0)
        new Tensor(values)
      case _ => throw new UnsupportedOperationException()
    }
  }

  override def userClass: Class[Tensor] = classOf[Tensor]

  override def equals(o: Any): Boolean = {
    o match {
      case v: TensorUDT => true
      case _ => false
    }
  }

  override def hashCode: Int = 7921

  override def typeName: String = "tensor"
}