package org.deeplearning4j.learn
import java.util.Arrays
import org.nd4j.api.linalg.DSL._
import org.nd4j.linalg.factory.Nd4j


/**
 * Examples of Scala Nd4j usage
 *
 * Before using Scala Nd4j, make sure you have included the nd4j-scala-api Maven dependency
 * Ensure you have the Scala plugin (IntelliJ: Preferences -> Plugins)
 * Add Maven Scala plugin and plugin repository to pom.xml
 *
 *
 * @author sonali
 */
object ScalaMain {

  def main (args: Array[String]) {
    /** Creating arrays in multiple ways, all using numpy syntax */
    var arr = Nd4j.create(4)
    var arr2 = Nd4j.ones(4)
    val arr3 = Nd4j.linspace(1, 10, 10)
    val arr4 = Nd4j.linspace(1, 6, 6).reshape(2, 3)

    /** Array addition in place */
    arr += arr2
    arr += 2

    /** Array multiplication in place */
    arr2 *= 5

    /** Transpose matrix */
    val arrT = arr.T

    /** Row (0) and Column (1) Sums */
    println(Nd4j.sum(arr4, 0) + "Calculate the sum for each row")
    println(Nd4j.sum(arr4, 1) + "Calculate the sum for each column")

    /** Checking array shape */
    println(Arrays.toString(arr2.shape) + "Checking array shape")

    /** Converting array to a string */
    println(arr2.toString() + "Array converted to string")

    /** Filling the array with the value 5 (same as numpy's fill method) */
    println(arr2.assign(5) + "Array assigned value of 5 (equivalent to fill method in numpy)")

    /** Reshaping the array */
    println(arr2.reshape(2, 2) + "Reshaping array")

    /** Raveling the array (returns a flattened array) */
    println(arr2.ravel + "Raveling array")

    /** Flattening the array (same as numpy's flatten method) */
    println(Nd4j.toFlattened(arr2) + "Flattening array (equivalent to flatten in numpy)")

    /** Array sorting */
    println(Nd4j.sort(arr2, 0, true) + "Sorting array")
    println(Nd4j.sortWithIndices(arr2, 0, true) + "Sorting array and returning sorted indices")

    /** Cumulative sum */
    println(Nd4j.cumsum(arr2) + "Cumulative sum")

    /** Basic stats methods */
    println(Nd4j.mean(arr) + "Calculate mean of array")
    println(Nd4j.std(arr2) + "Calculate standard deviation of array")
    println(Nd4j.`var`(arr2), "Calculate variance")

    /** Find min and max values */
    println(Nd4j.max(arr3), "Find max value in array")
    println(Nd4j.min(arr3), "Find min value in array")

  }

}
