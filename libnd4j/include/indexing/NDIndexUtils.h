//
// Created by agibsonccc on 4/26/22.
//

#ifndef LIBND4J_NDINDEXUTILS_H
#define LIBND4J_NDINDEXUTILS_H

#include <system/common.h>
#include <array/NDArrayFactory.h>
#include <array/NDArray.h>


namespace sd {

#define POINT_TYPE 0
#define INTERVAL_TYPE 1
#define ALL_TYPE 2
#define NEW_AXIS 3

#define DEFAULT_INCLUSIVE 1

class SD_LIB_EXPORT NDIndexUtils {

 public:
  static sd::NDArray createInterval(sd::LongType start,sd::LongType end,sd::LongType stride = 1,sd::LongType inclusive = 1);
  static sd::NDArray createInterval(LongType start, LongType end, LongType stride = 1, bool inclusive = true);
  static sd::NDArray createPoint(sd::LongType offset);
  static sd::NDArray createNewAxis();
  static sd::NDArray createAll();

};
}

#endif  // LIBND4J_NDINDEXUTILS_H
