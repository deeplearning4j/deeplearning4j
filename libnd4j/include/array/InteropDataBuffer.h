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

#include <array/DataBuffer.h>
#include <array/DataType.h>
#include <system/common.h>

#include <memory>

#ifndef LIBND4J_INTEROPDATABUFFER_H
#define LIBND4J_INTEROPDATABUFFER_H

namespace sd {
/**
 * This class is a wrapper for DataBuffer, suitable for sharing DataBuffer between front-end and back-end languages
 */
class SD_LIB_EXPORT InteropDataBuffer {
 private:
  std::shared_ptr<DataBuffer> _dataBuffer;
  uint64_t _offset = 0;
  bool owner;
 public:
  bool isConstant = false;

  InteropDataBuffer(InteropDataBuffer &dataBuffer, uint64_t length, uint64_t offset);
  InteropDataBuffer(std::shared_ptr<DataBuffer> databuffer);
  InteropDataBuffer(size_t lenInBytes, sd::DataType dtype, bool allocateBoth);
  ~InteropDataBuffer() {
    if(!isConstant)
      dataBuffer()->close();
  }
#ifndef __JAVACPP_HACK__
  std::shared_ptr<DataBuffer> getDataBuffer() const;
  std::shared_ptr<DataBuffer> dataBuffer();
#endif

  void *primary() const;
  void *special() const;

  void markConstant(bool reallyConstant) {
    isConstant = reallyConstant;
    dataBuffer()->markConstant(reallyConstant);
  }
  uint64_t offset() const;
  void setOffset(uint64_t offset);

  void setPrimary(void *ptr, size_t length);
  void setSpecial(void *ptr, size_t length);

  void expand(size_t newlength);

  int deviceId() const;
  void setDeviceId(int deviceId);

  //updates whether the buffer is the owner of its associated buffers or not.
  void markOwner(bool owner);

  int useCount() const;

  static void registerSpecialUse(const std::vector<const InteropDataBuffer *> &writeList,
                                 const std::vector<const InteropDataBuffer *> &readList);
  static void prepareSpecialUse(const std::vector<const InteropDataBuffer *> &writeList,
                                const std::vector<const InteropDataBuffer *> &readList,
                                bool synchronizeWritables = false);

  static void registerPrimaryUse(const std::vector<const InteropDataBuffer *> &writeList,
                                 const std::vector<const InteropDataBuffer *> &readList);
  static void preparePrimaryUse(const std::vector<const InteropDataBuffer *> &writeList,
                                const std::vector<const InteropDataBuffer *> &readList,
                                bool synchronizeWritables = false);
};
}  // namespace sd

#endif  // LIBND4J_INTEROPDATABUFFER_H
