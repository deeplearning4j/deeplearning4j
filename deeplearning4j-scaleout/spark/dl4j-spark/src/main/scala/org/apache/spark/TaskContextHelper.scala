package org.apache.spark


/**
  * @author raver119@gmail.com
  */
object TaskContextHelper {
  def setTaskContext(tc: TaskContext): Unit = TaskContext.setTaskContext(tc)
}
