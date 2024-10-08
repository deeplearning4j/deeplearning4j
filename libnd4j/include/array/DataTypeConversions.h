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
// Created by raver119 on 21.11.17.
//

#ifndef LIBND4J_DATATYPECONVERSIONS_H
#define LIBND4J_DATATYPECONVERSIONS_H

#include <array/DataType.h>
#include <execution/Threads.h>
#include <helpers/BitwiseUtils.h>
#include <helpers/logger.h>
#include <loops/type_conversions.h>
#include <system/common.h>
#include <system/op_boilerplate.h>
#include <types/float16.h>

namespace sd {
template <typename T>
class SD_LIB_EXPORT DataTypeConversions {
 private:
  template <typename T2>
  static SD_INLINE void rconv(bool isBe, bool canKeep, T *buffer, LongType length, void *src) {
    if (std::is_same<T, T2>::value && canKeep) {
      memcpy(buffer, src, length * sizeof(T));
    } else {
      auto tmp = new T2[length];
      memcpy(tmp, src, length * sizeof(T2));

#if __GNUC__ <= 4
      if (!canKeep)
        for (sd::LongType e = 0; e < length; e++) buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
      else
        TypeCast::convertGeneric<T2, T>(nullptr, tmp, length, buffer);
#else
      auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++)
          buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
      };

      samediff::Threads::parallel_for(func, 0, length);
#endif

      delete[] tmp;
    }
  }

 public:
  static SD_INLINE void convertType(void *vbuffer, void *src, DataType dataType, ByteOrder order, LongType length) {
    auto buffer = reinterpret_cast<T *>(vbuffer);
    bool isBe = BitwiseUtils::isBE();
    bool canKeep = (isBe && order == BE) || (!isBe && order == LE);

    switch (dataType) {
      case BOOL: {
        DataTypeConversions<T>::template rconv<bool>(isBe, canKeep, buffer, length, src);
      } break;
      case UINT8: {
        DataTypeConversions<T>::template rconv<uint8_t>(isBe, canKeep, buffer, length, src);
      } break;
      case INT8: {
        DataTypeConversions<T>::template rconv<int8_t>(isBe, canKeep, buffer, length, src);
      } break;
      case INT16: {
        DataTypeConversions<T>::template rconv<int16_t>(isBe, canKeep, buffer, length, src);
      } break;
      case INT32: {
        DataTypeConversions<T>::template rconv<int>(isBe, canKeep, buffer, length, src);
      } break;
      case INT64: {
        DataTypeConversions<T>::template rconv<LongType>(isBe, canKeep, buffer, length, src);
      } break;
      case FLOAT32: {
        if (std::is_same<T, float>::value && canKeep) {
          memcpy(buffer, src, length * sizeof(T));
        } else {
          auto tmp = new float[length];
          memcpy(tmp, src, length * sizeof(float));

#if __GNUC__ <= 4
          if (!canKeep)
            for (sd::LongType e = 0; e < length; e++) buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
          else
            TypeCast::convertGeneric<float, T>(nullptr, tmp, length, buffer);
#else
          auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e++)
              buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
          };

          samediff::Threads::parallel_for(func, 0, length);
#endif

          delete[] tmp;
        }
      } break;
      case DOUBLE: {
        if (std::is_same<T, double>::value && canKeep) {
          memcpy(buffer, src, length * sizeof(T));
        } else {
          auto tmp = new double[length];
          memcpy(tmp, src, length * sizeof(double));

#if __GNUC__ <= 4
          if (!canKeep)
            for (sd::LongType e = 0; e < length; e++) buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
          else
            TypeCast::convertGeneric<double, T>(nullptr, tmp, length, buffer);

#else
          auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e++)
              buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
          };

          samediff::Threads::parallel_for(func, 0, length);
#endif
          delete[] tmp;
        }
      } break;
      case HALF: {
        if (std::is_same<T, float16>::value && canKeep) {
          memcpy(buffer, src, length * sizeof(T));
        } else {
          auto tmp = new float16[length];
          memcpy(tmp, src, length * sizeof(float16));

#if __GNUC__ <= 4
          if (!canKeep)
            for (sd::LongType e = 0; e < length; e++) buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
          else
            TypeCast::convertGeneric<float16, T>(nullptr, tmp, length, buffer);
#else
          auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e++)
              buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
          };

          samediff::Threads::parallel_for(func, 0, length);
#endif
          delete[] tmp;
        }
      } break;
      default: {
        sd_printf("Unsupported DataType requested: [%i]\n", static_cast<int>(dataType));
        THROW_EXCEPTION("Unsupported DataType");
      }
    }
  }
};
}  // namespace sd

#endif  // LIBND4J_DATATYPECONVERSIONS_H
