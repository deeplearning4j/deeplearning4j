//
// Created by yurii@skymind.io on 02.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>


namespace nd4j {
    namespace ops {


template<typename T>
void reverseArray(T *dx, int *xShapeBuffer, T *result, int *zShapeBuffer) {
            
            Nd4jIndex xLength = shape::length(xShapeBuffer);
            int xEWS = shape::elementWiseStride(xShapeBuffer);
            char xOrder = shape::order(xShapeBuffer);
            Nd4jIndex sLength = xLength - 1;

            // two step phase here
            if (dx == result) {
                if (xEWS == 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jIndex e = 0; e < xLength / 2; e++) {
                        Nd4jIndex idx = sLength - e;
                        T tmp = dx[e];
                        dx[e] = dx[idx];
                        dx[idx] = tmp;
                    }
                } else if (xEWS > 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jIndex e = 0; e < xLength / 2; e++) {
                        Nd4jIndex idx1 = (sLength - e) * xEWS;
                        Nd4jIndex idx2 =  e * xEWS;
                        T tmp = dx[idx2];
                        dx[idx2] = dx[idx1];
                        dx[idx1] = tmp;
                    }
                } else {
                    int xRank = shape::rank(xShapeBuffer);
                    int *xShape = shape::shapeOf(xShapeBuffer);
                    int *xStride = shape::stride(xShapeBuffer);

                    int xCoord[MAX_RANK];
                    int zCoord[MAX_RANK];

#pragma omp parallel for private(xCoord, zCoord) schedule(guided)
                    for (Nd4jIndex e = 0; e < xLength / 2; e++) {
                        if (xOrder == 'c') {
                            shape::ind2subC(xRank, xShape, e, xCoord);
                            shape::ind2subC(xRank, xShape, sLength - e, zCoord);
                        } else {
                            shape::ind2sub(xRank, xShape, e, xCoord);
                            shape::ind2sub(xRank, xShape, sLength - e, zCoord);
                        }

                        Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        Nd4jIndex zOffset = shape::getOffset(0, xShape, xStride, zCoord, xRank);

                        result[zOffset] = dx[xOffset];
                    }
                }
            } else {
                // single step phase here
                int zEWS = shape::elementWiseStride(zShapeBuffer);
                char zOrder = shape::order(zShapeBuffer);

                if (xEWS == 1 && zEWS == 1 && xOrder == zOrder) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jIndex e = 0; e < xLength; e++) {
                        result[sLength - e] = dx[e];
                    }
                } else if (xEWS >= 1 && zEWS >= 1 && xOrder == zOrder) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jIndex e = 0; e < xLength; e++) {
                        result[(sLength - e) * zEWS] = dx[e * xEWS];
                    }
                } else {

                    int xRank = shape::rank(xShapeBuffer);
                    int *xShape = shape::shapeOf(xShapeBuffer);
                    int *xStride = shape::stride(xShapeBuffer);

                    int zRank = shape::rank(zShapeBuffer);
                    int *zShape = shape::shapeOf(zShapeBuffer);
                    int *zStride = shape::stride(zShapeBuffer);

                    int xCoord[MAX_RANK];
                    int zCoord[MAX_RANK];

#pragma omp parallel for private(xCoord, zCoord) schedule(guided)
                    for (Nd4jIndex e = 0; e < xLength; e++) {

                        if (xOrder == 'c')
                            shape::ind2subC(xRank, xShape, e, xCoord);
                        else
                            shape::ind2sub(xRank, xShape, e, xCoord);

                        if (zOrder == 'c')
                            shape::ind2subC(zRank, zShape, (sLength - e), zCoord);
                        else
                            shape::ind2sub(zRank, zShape, (sLength - e), zCoord);

                        Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        Nd4jIndex zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                        result[zOffset] = dx[xOffset];
                    }
                }
            }
}



//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(reverse, 1, 1, true, 0, -2) {
   
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    std::vector<int>* argI = block.getIArguments();
    std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), *argI);       

    auto listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);
    auto listIn  = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);
       
    NDArray<T>* subArrIn  = nullptr;
    NDArray<T>* subArrOut = nullptr;    
    for(int i=0; i<listIn->size(); ++i) {               // listIn->size() = listOut->size()
        subArrIn   = listIn->at(i);
        subArrOut  = listOut->at(i);        
        reverseArray<T>(subArrIn->getBuffer(), subArrIn->getShapeInfo(), subArrOut->getBuffer(), subArrOut->getShapeInfo());
    }

    STORE_RESULT(*output);

    delete listOut;
    delete listIn;

    return ND4J_STATUS_OK;
}




}
}