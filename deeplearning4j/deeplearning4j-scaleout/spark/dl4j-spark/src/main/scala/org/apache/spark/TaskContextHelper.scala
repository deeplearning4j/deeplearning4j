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
