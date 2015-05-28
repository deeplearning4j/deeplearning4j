
package org.apache.spark.ml.param

import org.apache.spark.ml.param._
import org.apache.spark.annotation.DeveloperApi

/**
 * Parameters for image dimensions.
 */
@DeveloperApi
trait HasDimensions extends Params {

  /**
   * Param for width
   * @group param
   */
  val width: IntParam = new IntParam(this, "width", "image width", None)

  /** @group getParam */
  def getWidth: Int = get(width)

  /**
   * Param for height
   * @group param
   */
  val height: IntParam = new IntParam(this, "height", "image height", None)

  /** @group getParam */
  def getHeight: Int = get(height)
}
