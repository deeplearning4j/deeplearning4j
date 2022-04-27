//
// Created by agibsonccc on 4/26/22.
//
#include <indexing/NDIndexUtils.h>
namespace sd {



sd::NDArray NDIndexUtils::createInterval(sd::LongType start,sd::LongType end,sd::LongType stride,sd::LongType inclusive) {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint = NDArrayFactory::create<sd::LongType>('c',{7},{INTERVAL_TYPE,2,1,start,end,stride,inclusive});
  return indexFirstPoint;
}

sd::NDArray NDIndexUtils::createPoint(sd::LongType offset) {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint = NDArrayFactory::create<sd::LongType>('c',{5},{POINT_TYPE,1,1,offset,DEFAULT_INCLUSIVE});
  return indexFirstPoint;
}

sd::NDArray NDIndexUtils::createNewAxis(sd::LongType offset) {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint = NDArrayFactory::create<sd::LongType>('c',{5},{NEW_AXIS,1,1,offset,DEFAULT_INCLUSIVE});
  return indexFirstPoint;
}

sd::NDArray NDIndexUtils::createAll() {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint = NDArrayFactory::create<sd::LongType>('c',{4},{ALL_TYPE,0,1,DEFAULT_INCLUSIVE});
  return indexFirstPoint;
}
}
