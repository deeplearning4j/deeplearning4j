package org.apache.spark


/**
  * This simple helper is used to get access to package-protected Scala TaskContext.setTaskContext method.
  * For more details, please read: https://issues.apache.org/jira/browse/SPARK-18406
  *
  * @author raver119@gmail.com
  */
object TaskContextHelper {
  def setTaskContext(tc: TaskContext): Unit = TaskContext.setTaskContext(tc)
}
