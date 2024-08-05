//
// Created by agibsonccc on 4/26/22.
//
#include <indexing/NDIndexUtils.h>
namespace sd {

NDArray NDIndexUtils::createInterval(LongType start, LongType end, LongType stride, bool inclusive) {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint =
      NDArrayFactory::create<LongType>('c', {7}, {INTERVAL_TYPE, 2, 1, start, end, stride, inclusive ? 1 : 0});
  return indexFirstPoint;
}

NDArray NDIndexUtils::createInterval(LongType start, LongType end, LongType stride, LongType inclusive) {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint =
      NDArrayFactory::create<LongType>('c', {7}, {INTERVAL_TYPE, 2, 1, start, end, stride, inclusive});
  return indexFirstPoint;
}

NDArray NDIndexUtils::createPoint(LongType offset) {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint = NDArrayFactory::create<LongType>('c', {5}, {POINT_TYPE, 1, 1, offset, DEFAULT_INCLUSIVE});
  return indexFirstPoint;
}

NDArray NDIndexUtils::createNewAxis() {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint = NDArrayFactory::create<LongType>('c', {5}, {NEW_AXIS, 1, 1, 0, DEFAULT_INCLUSIVE});
  return indexFirstPoint;
}

NDArray NDIndexUtils::createAll() {
  // index type, num indices,stride, indices (length num indices), inclusive
  auto indexFirstPoint = NDArrayFactory::create<LongType>('c',{4},{ALL_TYPE,0,1,DEFAULT_INCLUSIVE});
  return indexFirstPoint;
}
}
