//
// Created by agibsonccc on 8/30/24.
//

#ifndef LIBND4J_RESHAPENOCOPY_H
#define LIBND4J_RESHAPENOCOPY_H
#include <system/op_boilerplate.h>
#include <helpers/reshapeNoCopy.h>
#include <helpers/shape.h>
namespace sd {
namespace ops {
namespace helpers {
bool reshapeNoAlloc(const sd::LongType* inShape,
                    const std::vector<sd::LongType>& newShape,
                    char order,
                    sd::LongType* outShape) {
  LongType oldnd = shape::rank(inShape);
  std::vector<sd::LongType> olddims(oldnd);
  std::vector<sd::LongType> oldstrides(oldnd);
  sd::LongType np, op, last_stride;
  int oi, oj, ok, ni, nj, nk;
  std::vector<sd::LongType> newStrides(newShape.size());

  int newnd = newShape.size();
  bool isFOrder = order == 'f';

  // Remove axes with dimension 1 from the old array
  int actual_oldnd = 0;
  for (oi = 0; oi < oldnd; oi++) {
    if (shape::shapeOf(inShape)[oi] != 1) {
      olddims[actual_oldnd] = shape::shapeOf(inShape)[oi];
      oldstrides[actual_oldnd] = shape::stride(inShape)[oi];
      actual_oldnd++;
    }
  }

  oldnd = actual_oldnd;

  np = 1;
  for (ni = 0; ni < newnd; ni++) {
    np *= newShape[ni];
  }
  op = 1;
  for (oi = 0; oi < oldnd; oi++) {
    op *= olddims[oi];
  }
  if (np != op) {
    printf("failed to reshape allocation point 1\n");
    fflush(stdout);

    return false;  // total sizes must match
  }

  if (np == 0) {
    printf("failed to reshape allocation point 2\n");
    fflush(stdout);

    return false;  // don't support empty arrays
  }



  // oi to oj and ni to nj give the axis ranges currently worked with
  oi = 0;
  oj = 1;
  ni = 0;
  nj = 1;
  while (ni < newnd && oi < oldnd) {
    np = newShape[ni];
    op = olddims[oi];

    while (np != op) {
      if (np < op) {
        np *= newShape[nj++];
      } else {
        op *= olddims[oj++];
      }
    }

    // Check whether the original axes can be combined
    for (ok = oi; ok < oj - 1; ok++) {
      if (isFOrder) {
        if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
          printf("failed to reshape allocation point 3\n");
          fflush(stdout);
          return false;  // not contiguous enough
        }
      } else {
        // C order
        if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
          printf("failed to reshape allocation point 4\n");
          fflush(stdout);
          return false;  // not contiguous enough
        }
      }
    }

    // Calculate new strides for all axes currently worked with
    if (isFOrder) {
      newStrides[ni] = oldstrides[oi];
      for (nk = ni + 1; nk < nj; nk++) {
        newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
      }
    } else {
      // C order
      newStrides[nj - 1] = oldstrides[oj - 1];
      for (nk = nj - 1; nk > ni; nk--) {
        newStrides[nk - 1] = newStrides[nk] * newShape[nk];
      }
    }
    ni = nj++;
    oi = oj++;
  }



  // Set strides corresponding to trailing 1s of the new shape
  if (ni >= 1) {
    last_stride = newStrides[ni - 1];
  } else {
    last_stride = 1;
  }
  if (isFOrder && ni >= 1) {
    last_stride *= newShape[ni - 1];
  }
  for (nk = ni; nk < newnd; nk++) {
    newStrides[nk] = last_stride;
  }

  // Update the output shape info
  outShape[0] = newnd;  // Set rank

  printf("final no reshape alloc\n");
  fflush(stdout);
  shape::setShape(outShape, const_cast<sd::LongType*>(newShape.data()));
  shape::setStride(outShape, newStrides.data());
  shape::setOrder(outShape, order);
  ArrayOptions::resetFlags(outShape);
  ArrayOptions::setDataType(outShape, ArrayOptions::dataType(inShape));
  return true;
}
}
}
}

#endif