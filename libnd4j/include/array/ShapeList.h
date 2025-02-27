/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SHAPELIST_H
#define LIBND4J_SHAPELIST_H
#include <helpers/shape.h>
#include <system/common.h>

#include <vector>

namespace sd {
class SD_LIB_EXPORT ShapeList {
 protected:

  std::vector< sd::LongType *> _shapes;
  bool _destroyed = false;
  bool _autoremovable = false;
  bool _workspace = false;

 public:
  ShapeList( sd::LongType *shape = nullptr);
  ShapeList(const std::vector< sd::LongType *> &shapes, bool isWorkspace);
  ShapeList(const std::vector< sd::LongType *> &shapes);

  ~ShapeList();

  void destroy();
  int size() const;
   sd::LongType *at(int idx);
  void push_back( sd::LongType *shape);

  /**
   * PLEASE NOTE: This method should be called ONLY if shapes were generated at workspaces. Otherwise you'll get memory
   * leak
   */
  void detach();
};
}  // namespace sd

#endif  // LIBND4J_SHAPELIST_H
