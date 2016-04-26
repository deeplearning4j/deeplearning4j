#include <iostream>
#include <unistd.h>
#include <pairwise_util.h>
#include <pairwise_transform.h>

using namespace std;

int main() {
    int numDims = 13;
    int numShapes = 8;
    constexpr int sumDims[][4] = {
            {0}, {1}, {2}, {3},
            {0,1}, {0,2}, {0,3}, {1,2}, {1,3},
            {0,1,2}, {0,1,3}, {0,2,3},
            {0,1,2,3}
    };

    int lengths[13] = {
            1,1,1,1,
            2,2,2,2,2,
            3,3,3,
            4
    };
    constexpr int shapes[][4] ={
            //Standard case:
            {2,2,3,4},
            //Leading 1s:
            {1,2,3,4},
            {1,1,2,3},
            //Trailing 1s:
            {4,3,2,1},
            {4,3,1,1},
            //1s for non-leading/non-trailing dimensions
            {4,1,3,2},
            {4,3,1,2},
            {4,1,1,2}
    };



    int rank = 4;
    //o = 1 and j = 3
    for(int o = 0; o < numShapes; o++) {
        for(int j = 0; j < numDims; j++) {
            printf("Processing j %d\n",j);
            int *shape = new int[rank];
            int *shapeF = new int[rank];
            memcpy(shape,shapes[o],rank * sizeof(int));
            memcpy(shapeF,shapes[o],rank * sizeof(int));

            int *shapeInfo = shape::shapeBuffer(rank,shape);
            int *shapeInfoFortran = shape::shapeBufferFortran(rank,shapeF);
            int dimensionLength = lengths[j];
            int *dimension = new int[dimensionLength];
            int *dimensionF = new int[dimensionLength];
            memcpy(dimension,sumDims[j],sizeof(int) * dimensionLength);
            memcpy(dimensionF,sumDims[j],sizeof(int) * dimensionLength);

            printf("Shape\n");
            for(int i = 0; i < 4; i++) {
                printf(" %d ",shape[i]);
            }

            printf("\n");
            printf("Dimension\n");
            printf("C TAD elementwise stride is %d and f is %d\n",shape::tadElementWiseStride(shapeInfo,dimension,dimensionLength),shape::tadElementWiseStride(shapeInfoFortran,dimension,dimensionLength));


            double *x = new double[shape::length(shapeInfo)];
            for(int i = 0; i < shape::length(shapeInfo); i++) {
                x[i] = i + 1;
            }

            double *xF = new double[shape::length(shapeInfo)];
            functions::pairwise_transforms::ops::Set<double> *set = new functions::pairwise_transforms::ops::Set<double>();
            set->exec(xF,shapeInfoFortran,x,shapeInfo,xF,shapeInfoFortran,NULL);
            for(int i = 0; i < dimensionLength; i++) {
                printf(" %d ",dimension[i]);
            }
            printf("\n");
            shape::TAD tad(shapeInfo,dimension,dimensionLength);
            shape::TAD tadF(shapeInfoFortran,dimensionF,dimensionLength);
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            tadF.createTadOnlyShapeInfo();
            tadF.createOffsets();
            /*  printf("C ordering*****************************\n");
              shape::printShapeInfo(tad.tadOnlyShapeInfo);
              printf("*****************************************\n");
              printf("F ordering*****************************\n");
              shape::printShapeInfo(tadF.tadOnlyShapeInfo);
              printf("*****************************************\n");
   */
            /*    printf("Num tads is %d\n",tad.numTads);

                printf("Offsets C");
                for(int i = 0; i < tad.numTads; i++) {
                    printf(" %d ",tad.tadOffsets[i]);
                }
                printf("\n");

                printf("Offsets F");
                for(int i = 0; i < tadF.numTads; i++) {
                    printf(" %d ",tadF.tadOffsets[i]);
                }
                printf("\n");*/


            bool scalar = shape::oneDimEqualToLength(tad.tadOnlyShapeInfo);
            if(tad.wholeThing) {
                for(int i = 0; i < tad.tadLength; i++) {
                    printf(" %f ",x[i]);
                }
                printf("\n");

                printf("F ordering\n");
                for(int i = 0; i < tad.tadLength; i++) {
                    printf(" %f ",xF[i]);
                }

                printf("\n");
            }
            else if(shape::elementWiseStride(tad.tadOnlyShapeInfo) > 0 && (tad.numTads == 1 || shape::isVector(tad.tadOnlyShapeInfo) ||
                    shape::isScalar(tad.tadOnlyShapeInfo) || tad.wholeThing)) {
                for(int i = 0; i < tad.numTads; i++) {
                    printf("TAD %d\n",i);
                    double *iter = x + tad.tadOffsets[i];
                    int eleStride = shape::elementWiseStride(tad.tadOnlyShapeInfo);
                    if(eleStride == 1) {
                        for(int i = 0; i < tad.tadLength; i++) {
                            printf(" %f ",iter[i]);
                        }
                    }
                    else {
                        for(int i = 0; i < tad.tadLength; i++) {
                            printf(" %f ",iter[i * eleStride]);

                        }
                    }

                    printf("\n");

                    printf("F ordering\n");
                    double *iterF = xF + tadF.tadOffsets[i];
                    int eleStrideF = shape::elementWiseStride(tadF.tadOnlyShapeInfo);
                    if(eleStrideF == 1) {
                        for(int i = 0; i < tad.tadLength; i++) {
                            printf(" %f ",iterF[i]);
                        }
                    }
                    else {
                        for(int i = 0; i < tad.tadLength; i++) {
                            printf(" %f ",iterF[i * eleStrideF]);

                        }
                    }

                    printf("\n");
                }
            }
            else {
                shape::printShapeInfo(tad.tadOnlyShapeInfo);
                printf("Iterating over tads: %d",tad.numTads);
                for (int i = 0; i < tad.numTads; i++) {
                    int offset = tad.tadOffsets[i];
                    int fOffset = tadF.tadOffsets[i];
                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int idim;
                    int xStridesIter[MAX_RANK];
                    double *xPointer = x + offset;
                    double *xFPointer = xF + fOffset;
                    int ndim = shape::rank(tad.tadOnlyShapeInfo);
                    int rankIter = ndim;
                    printf("\nValues for tad %d",i);
                    printf("\nC ordering \n");
                    if (PrepareOneRawArrayIter<double>(rankIter,
                                                       tad.tadShape,
                                                       xPointer,
                                                       tad.tadStride,
                                                       &rankIter,
                                                       shapeIter,
                                                       &xPointer,
                                                       xStridesIter) >= 0) {
                        Nd4jIndex offsetCurr = offset;
                        memset((coord), 0, (ndim) * sizeof(coord[0]));
                        do {
                          /*     printf("Coord ");
                               for(int i = 0; i < rankIter; i++) {
                                   printf(" %d ",coord[i]);
                               }*/
                            printf(" %f ",xPointer[0]);
                          //  printf("\n");

                            for ((idim) = 0; (idim) < (ndim); (idim)++) {
                                if(tad.tadShape[idim] == 1)
                                    continue;

                                if (++(coord)[idim] >= (tad.tadShape)[idim]) {
                                    (coord)[idim] = 0;
                                    (xPointer) -= ((tad.tadShape)[idim] - 1) * (xStridesIter)[idim];
                                    offsetCurr -= ((tad.tadShape)[idim] - 1) * (xStridesIter)[idim];
                                }
                                else if ((tad.tadShape)[idim] != 1) {
                                    (xPointer) += (xStridesIter)[idim];
                                    offsetCurr += (xStridesIter)[idim];
                                    break;
                                }
                            }
                        } while ((idim) < (ndim));




                        /*   ND4J_RAW_ITER_START(dim, shape::rank(tad.tadOnlyShapeInfo), coord, shapeIter); {
       // Process the innermost dimension

                                   printf(" %f ",xPointer[0]);
                               }
                           ND4J_RAW_ITER_ONE_NEXT(dim,
                                                  shape::rank(tad.tadOnlyShapeInfo),
                                                  coord,
                                                  shapeIter,
                                                  xPointer,
                                                  xStridesIter);*/
                        printf("\n");
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }


                    printf("F order \n");

                    if (PrepareOneRawArrayIter<double>(rankIter,
                                                       tadF.tadShape,
                                                       xFPointer,
                                                       tadF.tadStride,
                                                       &rankIter,
                                                       shapeIter,
                                                       &xFPointer,
                                                       xStridesIter) >= 0) {

                        //F ORDERING
                        memset((coord), 0, (ndim) * sizeof(coord[0]));
                        do {
                            double currVal = xFPointer[0];
                           /* printf("Coord ");
                            for(int i = 0; i < rankIter; i++) {
                                printf(" %d ",coord[i]);
                            }*/
                            printf(" %f ",currVal);
                           // printf("\n");

                            for ((idim) = 0; (idim) < (ndim); (idim)++) {
                                if (++(coord)[idim] >= (tadF.tadShape)[idim]) {
                                    (coord)[idim] = 0;
                                    (xFPointer) -= ((tadF.tadShape)[idim] - 1) * (xStridesIter)[idim];
                                }
                                else if ((tadF.tadShape)[idim] != 1) {
                                    (xFPointer) += (xStridesIter)[idim];
                                    break;
                                }
                            }
                        } while ((idim) < (ndim));


                        /*      ND4J_RAW_ITER_START(dim, shape::rank(tad.tadOnlyShapeInfo), coord, shapeIter); {
          // Process the innermost dimension

                                      printf(" %f ",xFPointer[0]);
                                  }
                              ND4J_RAW_ITER_ONE_NEXT(dim,
                                                     shape::rank(tadF.tadOnlyShapeInfo),
                                                     coord,
                                                     shapeIter,
                                                     xFPointer,
                                                     xStridesIter);*/
                        printf("\n");
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }


                }
            }


            delete set;
            delete[] shapeInfo;
            delete[] shapeInfoFortran;
            delete[] shape;
            delete[] dimension;
        }
    }

    usleep(10000000000000);


    return 0;
}