/*******************************************************************************
  * Copyright (c) 2015-2018 Skymind, Inc.
  *
  * This program and the accompanying materials are made available under the
  * terms of the Apache License, Version 2.0 which is available at
  * https://www.apache.org/licenses/LICENSE-2.0.
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  * License for the specific language governing permissions and limitations
  * under the License.
  *
  * SPDX-License-Identifier: Apache-2.0
  ******************************************************************************/
package org.nd4s.ops

import java.util.{ List, Map, Properties }

import org.bytedeco.javacpp.Pointer
import org.nd4j.linalg.api.buffer.{ DataBuffer, DataType, Utf8Buffer }
import org.nd4j.linalg.api.environment.Nd4jEnvironment
import org.nd4j.linalg.api.ndarray.{ INDArray, INDArrayStatistics }
import org.nd4j.linalg.api.ops.aggregates.{ Aggregate, Batch }
import org.nd4j.linalg.api.ops._
import org.nd4j.linalg.api.ops.executioner.OpExecutioner
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate
import org.nd4j.linalg.api.ops.impl.summarystats.Variance
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.api.shape.{ LongShapeDescriptor, TadPack }
import org.nd4j.linalg.cache.TADManager
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.profiler.ProfilerConfig
import org.slf4j.{ Logger, LoggerFactory }

object FunctionalOpExecutioner {
  def apply: FunctionalOpExecutioner = new FunctionalOpExecutioner()
}
class FunctionalOpExecutioner extends OpExecutioner {

  def log: Logger = LoggerFactory.getLogger(FunctionalOpExecutioner.getClass)

  private[this] var verboseEnabled: Boolean = false

  def isVerbose: Boolean = verboseEnabled

  def enableVerboseMode(reallyEnable: Boolean): Unit =
    verboseEnabled = reallyEnable

  /**
    * This method returns true if debug mode is enabled, false otherwise
    *
    * @return
    */
  private[this] var debugEnabled: Boolean = false

  def isDebug: Boolean = debugEnabled

  def enableDebugMode(reallyEnable: Boolean): Unit =
    debugEnabled = reallyEnable

  /**
    * This method returns type for this executioner instance
    *
    * @return
    */
  def `type`: OpExecutioner.ExecutionerType = ???

  /**
    * This method returns opName of the last invoked op
    *
    * @return
    */
  def getLastOp: String = ???

  /**
    * Execute the operation
    *
    * @param op the operation to execute
    */
  def exec(op: Op): INDArray =
    op match {
      case op: FilterOps    => exec(op.asInstanceOf[FilterOps])
      case op: BitFilterOps => exec(op.asInstanceOf[BitFilterOps])
      case op: MapOps       => exec(op.asInstanceOf[MapOps])
      case _                => op.z()
    }

  def exec(op: FilterOps): INDArray = {
    val retVal: INDArray = Nd4j.create(op.x.dataType(), op.x.shape().map(_.toLong): _*)
    for (i <- 0 until op.x().length().toInt) {
      val filtered = op.x.dataType() match {
        case DataType.DOUBLE => op.op(op.x.getDouble(i.toLong))
        case DataType.FLOAT  => op.op(op.x.getFloat(i.toLong))
        case DataType.INT    => op.op(op.x.getInt(i))
        case DataType.SHORT  => op.op(op.x.getInt(i))
        case DataType.LONG   => op.op(op.x.getLong(i.toLong))
      }
      retVal.putScalar(i, filtered)
    }
    retVal
  }

  def exec(op: BitFilterOps): INDArray = {
    val retVal: INDArray = Nd4j.create(op.x.dataType(), op.x.shape().map(_.toLong): _*)
    for (i <- 0 until op.x().length().toInt) {
      val current = if (op.x.dataType() == DataType.DOUBLE) op.x().getDouble(i.toLong) else op.x().getInt(i)
      val filtered = op.op(current)

      retVal.putScalar(i, filtered)
    }
    retVal
  }

  def exec(op: MapOps): INDArray = {
    val retVal: INDArray = Nd4j.create(op.x.dataType(), op.x.shape().map(_.toLong): _*)
    for (i <- 0 until op.x().length().toInt) {
      val current = if (op.x.dataType() == DataType.DOUBLE) op.x().getDouble(i.toLong) else op.x().getInt(i)
      val filtered = op.op(current)

      retVal.putScalar(i, filtered)
    }
    retVal
  }

  /** Execute a TransformOp and return the result
    *
    * @param op the operation to execute
    */
  def execAndReturn(op: TransformOp): TransformOp =
    Nd4j.getExecutioner.execAndReturn(op)

  /**
    * Execute and return the result from an accumulation
    *
    * @param op the operation to execute
    * @return the accumulated result
    */
  def execAndReturn(op: ReduceOp): ReduceOp =
    Nd4j.getExecutioner.execAndReturn(op)

  def execAndReturn(op: Variance): Variance =
    Nd4j.getExecutioner.execAndReturn(op)

  /** Execute and return the result from an index accumulation
    *
    * @param op the index accumulation operation to execute
    * @return the accumulated index
    */
  def execAndReturn(op: IndexAccumulation): IndexAccumulation =
    Nd4j.getExecutioner.execAndReturn(op)

  /** Execute and return the result from a scalar op
    *
    * @param op the operation to execute
    * @return the accumulated result
    */
  def execAndReturn(op: ScalarOp): ScalarOp =
    Nd4j.getExecutioner.execAndReturn(op)

  /** Execute and return the result from a vector op
    *
    * @param op */
  def execAndReturn(op: BroadcastOp): BroadcastOp =
    Nd4j.getExecutioner.execAndReturn(op)

  /**
    * Execute a reduceOp, possibly along one or more dimensions
    *
    * @param reduceOp the reduceOp
    * @return the reduceOp op
    */
  def exec(reduceOp: ReduceOp): INDArray =
    Nd4j.getExecutioner.exec(reduceOp)

  /**
    * Execute a broadcast op, possibly along one or more dimensions
    *
    * @param broadcast the accumulation
    * @return the broadcast op
    */
  def exec(broadcast: BroadcastOp): INDArray =
    Nd4j.getExecutioner.exec(broadcast)

  /**
    * Execute ScalarOp
    *
    * @param broadcast
    * @return
    */
  def exec(broadcast: ScalarOp): INDArray =
    Nd4j.exec(broadcast)

  /**
    * Execute an variance accumulation op, possibly along one or more dimensions
    *
    * @param accumulation the accumulation
    * @return the accmulation op
    */
  def exec(accumulation: Variance): INDArray =
    Nd4j.getExecutioner.exec(accumulation)

  /** Execute an index accumulation along one or more dimensions
    *
    * @param indexAccum the index accumulation operation
    * @return result
    */
  def exec(indexAccum: IndexAccumulation): INDArray =
    Nd4j.getExecutioner.exec(indexAccum)

  /**
    *
    * Execute and return  a result
    * ndarray from the given op
    *
    * @param op the operation to execute
    * @return the result from the operation
    */
  def execAndReturn(op: Op): Op =
    Nd4j.getExecutioner.execAndReturn(op)

  /**
    * Execute MetaOp
    *
    * @param op
    */
  def exec(op: MetaOp): Unit =
    Nd4j.getExecutioner.exec(op)

  /**
    * Execute GridOp
    *
    * @param op
    */
  def exec(op: GridOp): Unit =
    Nd4j.getExecutioner.exec(op)

  /**
    *
    * @param op
    */
  def exec(op: Aggregate): Unit =
    Nd4j.getExecutioner.exec(op)

  /**
    * This method executes previously built batch
    *
    * @param batch
    */
  def exec[T <: Aggregate](batch: Batch[T]): Unit =
    Nd4j.getExecutioner.exec(batch)

  /**
    * This method takes arbitrary sized list of aggregates,
    * and packs them into batches
    *
    * @param batch
    */
  def exec(batch: java.util.List[Aggregate]): Unit =
    Nd4j.getExecutioner.exec(batch)

  /**
    * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
    *
    * @param op
    */
  def exec(op: RandomOp): INDArray =
    Nd4j.getExecutioner.exec(op)

  /**
    * This method executes specific RandomOp against specified RNG
    *
    * @param op
    * @param rng
    */
  def exec(op: RandomOp, rng: Random): INDArray =
    Nd4j.getExecutioner.exec(op, rng)

  /**
    * This method return set of key/value and
    * key/key/value objects,
    * describing current environment
    *
    * @return
    */
  def getEnvironmentInformation: Properties =
    Nd4j.getExecutioner.getEnvironmentInformation

  /**
    * This method specifies desired profiling mode
    *
    * @param mode
    */
  @deprecated def setProfilingMode(mode: OpExecutioner.ProfilingMode): Unit = ???

  /**
    * This method stores specified configuration.
    *
    * @param config
    */
  def setProfilingConfig(config: ProfilerConfig): Unit =
    Nd4j.getExecutioner.setProfilingConfig(config)

  /**
    * Ths method returns current profiling
    *
    * @return
    */
  @deprecated def getProfilingMode: OpExecutioner.ProfilingMode = ???

  /**
    * This method returns TADManager instance used for this OpExecutioner
    *
    * @return
    */
  def getTADManager: TADManager =
    Nd4j.getExecutioner.getTADManager

  /**
    * This method prints out environmental information returned by getEnvironmentInformation() method
    */
  def printEnvironmentInformation(): Unit =
    Nd4j.getExecutioner.printEnvironmentInformation()

  /**
    * This method ensures all operations that supposed to be executed at this moment, are executed.
    */
  def push(): Unit = ???

  /**
    * This method ensures all operations that supposed to be executed at this moment, are executed and finished.
    */
  def commit(): Unit = ???

  /**
    * This method encodes array as thresholds, updating input array at the same time
    *
    * @param input
    * @return encoded array is returned
    */
  def thresholdEncode(input: INDArray, threshold: Double): INDArray = ???

  def thresholdEncode(input: INDArray, threshold: Double, boundary: Integer): INDArray = ???

  /**
    * This method decodes thresholds array, and puts it into target array
    *
    * @param encoded
    * @param target
    * @return target is returned
    */
  def thresholdDecode(encoded: INDArray, target: INDArray): INDArray = ???

  /**
    * This method returns number of elements affected by encoder
    *
    * @param indArray
    * @param target
    * @param threshold
    * @return
    */
  def bitmapEncode(indArray: INDArray, target: INDArray, threshold: Double): Long = ???

  /**
    *
    * @param indArray
    * @param threshold
    * @return
    */
  def bitmapEncode(indArray: INDArray, threshold: Double): INDArray = ???

  /**
    *
    * @param encoded
    * @param target
    * @return
    */
  def bitmapDecode(encoded: INDArray, target: INDArray): INDArray = ???

  /**
    * This method returns names of all custom operations available in current backend, and their number of input/output arguments
    *
    * @return
    */
  def getCustomOperations: java.util.Map[String, CustomOpDescriptor] = ???

  /**
    * This method executes given CustomOp
    *
    * PLEASE NOTE: You're responsible for input/output validation
    *
    * @param op
    */
  def execAndReturn(op: CustomOp): CustomOp = ???

  def exec(op: CustomOp): Array[INDArray] = ???

  /**
    * This method executes op with given context
    *
    * @param op
    * @param context
    * @return method returns output arrays defined within context
    */
  def exec(op: CustomOp, context: OpContext): Array[INDArray] =
    Nd4j.getExecutioner.exec(op, context)

  def calculateOutputShape(op: CustomOp): java.util.List[LongShapeDescriptor] =
    Nd4j.getExecutioner.calculateOutputShape(op)

  /**
    * Equivalent to calli
    */
  def allocateOutputArrays(op: CustomOp): Array[INDArray] =
    Nd4j.getExecutioner.allocateOutputArrays(op)

  def isExperimentalMode: Boolean = true

  def registerGraph(id: Long, graph: Pointer): Unit = ???

  def executeGraph(id: Long,
                   map: java.util.Map[String, INDArray],
                   reverseMap: java.util.Map[String, Integer]): java.util.Map[String, INDArray] = ???

  def forgetGraph(id: Long): Unit = ???

  /**
    * This method allows to set desired number of elements per thread, for performance optimization purposes.
    * I.e. if array contains 2048 elements, and threshold is set to 1024, 2 threads will be used for given op execution.
    *
    * Default value: 1024
    *
    * @param threshold
    */
  def setElementsThreshold(threshold: Int): Unit = ???

  /**
    * This method allows to set desired number of sub-arrays per thread, for performance optimization purposes.
    * I.e. if matrix has shape of 64 x 128, and threshold is set to 8, each thread will be processing 8 sub-arrays (sure, if you have 8 core cpu).
    * If your cpu has, say, 4, cores, only 4 threads will be spawned, and each will process 16 sub-arrays
    *
    * Default value: 8
    *
    * @param threshold
    */
  def setTadThreshold(threshold: Int): Unit = ???

  /**
    * This method extracts String from Utf8Buffer
    *
    * @param buffer
    * @param index
    * @return
    */
  def getString(buffer: Utf8Buffer, index: Long): String = ???

  /**
    * This method returns OpContext which can be used (and reused) to execute custom ops
    *
    * @return
    */
  def buildContext: OpContext = ???

  /**
    *
    * @param array
    */
  def inspectArray(array: INDArray): INDArrayStatistics = ???

  /**
    * This method returns shapeInfo DataBuffer
    *
    * @param shape
    * @param stride
    * @param elementWiseStride
    * @param order
    * @param dtype
    * @return
    */
  def createShapeInfo(shape: Array[Long],
                      stride: Array[Long],
                      elementWiseStride: Long,
                      order: Char,
                      dtype: DataType,
                      empty: Boolean): DataBuffer = ???

  /**
    * This method returns host/device tad buffers
    */
  def tadShapeInfoAndOffsets(array: INDArray, dimension: Array[Int]): TadPack = ???

  /**
    * This method returns constant buffer for the given jvm array
    *
    * @param values
    * @return
    */
  def createConstantBuffer(values: Array[Long], desiredType: DataType): DataBuffer = ???

  def createConstantBuffer(values: Array[Int], desiredType: DataType): DataBuffer = ???

  def createConstantBuffer(values: Array[Float], desiredType: DataType): DataBuffer = ???

  def createConstantBuffer(values: Array[Double], desiredType: DataType): DataBuffer = ???

  def runFullBenchmarkSuit(x: Boolean): String =
    Nd4j.getExecutioner.runFullBenchmarkSuit(x)

  def runLightBenchmarkSuit(x: Boolean): String =
    Nd4j.getExecutioner.runLightBenchmarkSuit(x)

  @deprecated def scatterUpdate(op: ScatterUpdate.UpdateOp,
                                array: INDArray,
                                indices: INDArray,
                                updates: INDArray,
                                axis: Array[Int]): Unit = ???
}
