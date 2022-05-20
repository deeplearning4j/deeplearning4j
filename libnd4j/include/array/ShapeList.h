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
#if defined(__NEC__)
  const sd::LongType *_shapes[SD_MAX_INPUT_SIZE];
  int size_x = 0;
#else
  std::vector<const sd::LongType *> _shapes;
#endif
  bool _destroyed = false;
  bool _autoremovable = false;
  bool _workspace = false;

 public:
  ShapeList(const sd::LongType *shape = nullptr);
  ShapeList(const std::vector<const sd::LongType *> &shapes, bool isWorkspace);
  ShapeList(const std::vector<const sd::LongType *> &shapes);
  // ShapeList(bool autoRemovable);

  ~ShapeList();

  // std::vector<const sd::LongType *> *asVector();
  void destroy();
  int size() const;
  const sd::LongType *at(int idx);
  void push_back(const sd::LongType *shape);

  /**
   * PLEASE NOTE: This method should be called ONLY if shapes were generated at workspaces. Otherwise you'll get memory
   * leak
   */
  void detach();
};
}  // namespace sd

#endif  // LIBND4J_SHAPELIST_H
