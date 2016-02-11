package org.nd4s

import org.nd4j.linalg.api.complex.{IComplexNDArray, IComplexNumber}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{NDArrayIndex, INDArrayIndex}

object Implicits {

  implicit class RichINDArray[A <: INDArray, B](val underlying: A) extends SliceableNDArray[A] with OperatableNDArray[A] with CollectionLikeNDArray[A]

  implicit def rowProjection2NDArray(row:RowProjectedNDArray):INDArray = row.array

  implicit def columnProjection2NDArray(column:ColumnProjectedNDArray):INDArray = column.array

  implicit def sliceProjection2NDArray(sliced:SliceProjectedNDArray):INDArray = sliced.array

  /*
   Avoid using Numeric[T].toDouble(t:T) for sequence transformation in XXColl2INDArray to minimize memory consumption.
   */

  implicit def floatArray2INDArray(s:Array[Float]):FloatArray2INDArray = new FloatArray2INDArray(s)
  implicit def floatColl2INDArray(s:Seq[Float]):FloatArray2INDArray = new FloatArray2INDArray(s.toArray)
  class FloatArray2INDArray(val underlying: Array[Float]) extends AnyVal {
    def mkNDArray(shape: Array[Int], ord: NDOrdering = NDOrdering(Nd4j.order()), offset: Int = 0): INDArray = Nd4j.create(underlying, shape, ord.value, offset)

    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying, shape.toArray)

    def toNDArray: INDArray = Nd4j.create(underlying)
  }

  implicit def doubleArray2INDArray(s:Array[Double]):DoubleArray2INDArray = new DoubleArray2INDArray(s)
  implicit def doubleArray2CollArray(s:Seq[Double]):DoubleArray2INDArray = new DoubleArray2INDArray(s.toArray)
  class DoubleArray2INDArray(val underlying: Array[Double]) extends AnyVal {
    def mkNDArray(shape: Array[Int], ord: NDOrdering = NDOrdering(Nd4j.order()), offset: Int = 0): INDArray = Nd4j.create(underlying, shape, offset, ord.value)

    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying, shape.toArray)

    def toNDArray: INDArray = Nd4j.create(underlying)
  }

  implicit def intColl2INDArray(s:Seq[Int]):IntArray2INDArray = new IntArray2INDArray(s.toArray)
  implicit def intArray2INDArray(s:Array[Int]):IntArray2INDArray = new IntArray2INDArray(s)

  class IntArray2INDArray(val underlying: Array[Int]) extends AnyVal {
    def mkNDArray(shape: Array[Int], ord: NDOrdering = NDOrdering(Nd4j.order()), offset: Int = 0): INDArray = Nd4j.create(underlying.map(_.toFloat), shape, ord.value, offset)

    def asNDArray(shape: Int*): INDArray = Nd4j.create(underlying.map(_.toFloat), shape.toArray)

    def toNDArray: INDArray = Nd4j.create(underlying.map(_.toFloat).toArray)
  }

  implicit def complexArray2INDArray(underlying: Array[IComplexNumber]):ComplexArray2INDArray = new ComplexArray2INDArray(underlying)
  implicit def complexColl2INDArray(underlying: Seq[IComplexNumber]):ComplexArray2INDArray = new ComplexArray2INDArray(underlying.toArray)
  class ComplexArray2INDArray(val underlying: Array[IComplexNumber]) extends AnyVal {
    def mkNDArray(shape: Array[Int], ord: NDOrdering = NDOrdering(Nd4j.order()), offset: Int = 0): IComplexNDArray = Nd4j.createComplex(underlying, shape, offset,ord.value)

    def asNDArray(shape: Int*): IComplexNDArray = Nd4j.createComplex(underlying, shape.toArray)

    def toNDArray: IComplexNDArray = Nd4j.createComplex(underlying)
  }

  implicit class floatMtrix2INDArray(val underlying: Seq[Seq[Float]]) extends AnyVal {
    def mkNDArray(ord: NDOrdering): INDArray = Nd4j.create(underlying.map(_.toArray).toArray, ord.value)
    def toNDArray: INDArray = Nd4j.create(underlying.map(_.toArray).toArray)
  }

  implicit class floatArrayMtrix2INDArray(val underlying: Array[Array[Float]]) extends AnyVal {
    def mkNDArray(ord: NDOrdering): INDArray = Nd4j.create(underlying, ord.value)
    def toNDArray: INDArray = Nd4j.create(underlying)
  }

  implicit class doubleMtrix2INDArray(val underlying: Seq[Seq[Double]]) extends AnyVal {
    def mkNDArray(ord: NDOrdering): INDArray = Nd4j.create(underlying.map(_.toArray).toArray, ord.value)
    def toNDArray: INDArray = Nd4j.create(underlying.map(_.toArray).toArray)
  }

  implicit class doubleArrayMtrix2INDArray(val underlying: Array[Array[Double]]) extends AnyVal {
    def mkNDArray(ord: NDOrdering ): INDArray = Nd4j.create(underlying,ord.value)
    def toNDArray: INDArray = Nd4j.create(underlying)
  }

  implicit class intMtrix2INDArray(val underlying: Seq[Seq[Int]]) extends AnyVal {
    def mkNDArray(ord: NDOrdering): INDArray = Nd4j.create(underlying.map(_.map(_.toFloat).toArray).toArray, ord.value)
    def toNDArray: INDArray = Nd4j.create(underlying.map(_.map(_.toFloat).toArray).toArray)
  }

  implicit class intArrayMtrix2INDArray(val underlying: Array[Array[Int]]) extends AnyVal {
    def mkNDArray(ord: NDOrdering): INDArray = Nd4j.create(underlying.map(_.map(_.toFloat)), ord.value)
    def toNDArray: INDArray = Nd4j.create(underlying.map(_.map(_.toFloat)))
  }

  implicit class complexArrayMtrix2INDArray(val underlying: Array[Array[IComplexNumber]]) extends AnyVal {
    def toNDArray: IComplexNDArray = Nd4j.createComplex(underlying)
  }

  implicit class complexCollMtrix2INDArray(val underlying: Seq[Seq[IComplexNumber]]) extends AnyVal {
    def toNDArray: IComplexNDArray = Nd4j.createComplex(underlying.map(_.toArray).toArray)
  }

  implicit class num2Scalar[T](val underlying: T)(implicit ev: Numeric[T]) {
    def toScalar: INDArray = Nd4j.scalar(ev.toDouble(underlying))
  }

  implicit class icomplexNum2Scalar(val underlying: IComplexNumber){
    def toScalar: IComplexNDArray = Nd4j.scalar(underlying)
  }

  implicit def intArray2IndexRangeArray(arr:Array[Int]):Array[IndexRange] = arr.map(new IntRange(_))

  case object -> extends IndexRange{
    override def hasNegative: Boolean = false
  }

  case object ---> extends IndexRange{
    override def hasNegative: Boolean = false
  }

  implicit class IntRange(val underlying: Int) extends IndexNumberRange {
    protected[nd4s] override def asRange(max: => Int): DRange = DRange(underlying, underlying, true, 1, max)

    override protected[nd4s] def asNDArrayIndex(max: => Int): INDArrayIndex = NDArrayIndex.point(underlying)

    override def hasNegative: Boolean = false

    override def toString: String = s"$underlying"
  }

  implicit class TupleRange(val underlying: _root_.scala.Tuple2[Int, Int]) extends IndexNumberRange {
    protected[nd4s] override def asRange(max: => Int): DRange = DRange(underlying._1, underlying._2, false, 1, max)

    override protected[nd4s] def asNDArrayIndex(max: => Int): INDArrayIndex = IndexNumberRange.toNDArrayIndex(underlying._1, underlying._2, false, 1,max)

    override def toString: String = s"${underlying._1}->${underlying._2}"


    override def hasNegative: Boolean = underlying._1 < 0 || underlying._2 < 0

    def by(i: Int) = new IndexRangeWrapper(underlying._1 until underlying._2 by i)
  }

  implicit class IntRangeFromGen(val underlying: Int) extends AnyVal {
    def -> = IntRangeFrom(underlying)
  }

  implicit class IndexRangeWrapper(val underlying: Range) extends IndexNumberRange {
    protected[nd4s] override def asRange(max: => Int): DRange = DRange.from(underlying, max)

    override protected[nd4s] def asNDArrayIndex(max: => Int): INDArrayIndex = IndexNumberRange.toNDArrayIndex(underlying.start,underlying.end,underlying.isInclusive,underlying.step,max)

    override def toString: String = s"${underlying.start}->${underlying.end} by ${underlying.step}"

    override def hasNegative: Boolean = underlying.start < 0 || underlying.end < 0 || underlying.step < 0
  }

  implicit class NDArrayIndexWrapper(val underlying: INDArrayIndex) extends IndexNumberRange {
    protected[nd4s] override def asRange(max: => Int): DRange = DRange(underlying.current(),underlying.end(),false,underlying.stride(),max)

    override protected[nd4s] def asNDArrayIndex(max: => Int): INDArrayIndex = underlying

    override def toString: String = s"${underlying.current}->${underlying.end} by ${underlying.stride}"

    override def hasNegative: Boolean = false
  }

  lazy val NDOrdering = org.nd4s.NDOrdering

  implicit def int2ComplexNumberBuilder(underlying: Int): ComplexNumberBuilder[Integer] = new ComplexNumberBuilder[Integer](underlying)

  implicit def float2ComplexNumberBuilder(underlying: Float): ComplexNumberBuilder[java.lang.Float] = new ComplexNumberBuilder[java.lang.Float](underlying)

  implicit def double2ComplexNumberBuilder(underlying: Double): ComplexNumberBuilder[java.lang.Double] = new ComplexNumberBuilder[java.lang.Double](underlying)

  lazy val i = new ImaginaryNumber[Integer](1)
}

private[nd4s] class ComplexNumberBuilder[T <: Number](val value: T) extends AnyVal {
  def +[A <: Number](imaginary: ImaginaryNumber[A]): IComplexNumber = Nd4j.createComplexNumber(value, imaginary.value)

  def *(in: ImaginaryNumber[Integer]): ImaginaryNumber[T] = new ImaginaryNumber[T](value)
}

private[nd4s] class ImaginaryNumber[T <: Number](val value: T) extends AnyVal {
  override def toString: String = s"${value}i"
}

sealed trait IndexNumberRange extends IndexRange {
  protected[nd4s] def asRange(max: => Int): DRange
  protected[nd4s] def asNDArrayIndex(max: => Int):INDArrayIndex
}
object IndexNumberRange{
  def toNDArrayIndex(startR:Int,endR:Int,isInclusive:Boolean,step:Int,max:Int):INDArrayIndex = {
    val (start, end) = {
      val start = if (startR >= 0) startR else max + startR
      val diff = if(isInclusive) 1 else 0
      val endExclusive = if (endR >= 0) endR + diff else max + endR + diff
      (start, endExclusive)
    }

    NDArrayIndex.interval(start,step,end,false)
  }
}

sealed trait IndexRange{
  def hasNegative:Boolean
}

case class IntRangeFrom(underlying: Int) extends IndexRange {
  def apply[T](a:T): (Int, T) = (underlying, a)

  override def toString: String = s"$underlying->"

  override def hasNegative: Boolean = false
}

private[nd4s] case class DRange(startR: Int, endR: Int, isInclusive: Boolean, step: Int, max: Int) {
  lazy val (start, end) = {
    val start = if (startR >= 0) startR else max + startR
    val diff = if (isInclusive) 0 else if (step >= 0) -1 else +1
    val endInclusive = if (endR >= 0) endR + diff else max + endR + diff
    (start, endInclusive)
  }
  lazy val length = (end - start) / step + 1

  def toList: List[Int] = List.iterate(start, length)(_ + step)

  override def toString: String = s"[$start to $end by $step len:$length]"
}

private[nd4s] object DRange extends {
  def from(r: Range, max: => Int): DRange = DRange(r.start, r.end, r.isInclusive, r.step, max)

  def apply(startR: Int, endR: Int, step: Int): DRange = DRange(startR, endR, false, step, Int.MinValue)
}
