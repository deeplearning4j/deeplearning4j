/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//

#ifndef LIBND4J_TAD_H
#define LIBND4J_TAD_H


#include <helpers/shape.h>
#include <pointercast.h>


namespace shape {
    /**
     * Dimension collapse is an algorithm
     * for collapsing singular dimensions.
     * This algorithm will adjust the dimensions
     * wrt the original.
     *
     * The algorithm has 3 components:
     * trailing ones
     * middle ones
     * beginning ones
     *
     * dimensions that are specified to reduce along
     * that are singular should be truncated
     *
     * dimensions that are specified that are singular
     * at the beginning should be removed with middle dimensions
     * decremented.
     *
     * For any time there is a no op, a collapse will
     * set the first dimension to be -1.
     *
     *
     */
    class TAD {
    public:
        Nd4jLong tadIndex = 0;
        int dimensionLength;
        int* dimension = nullptr;
        Nd4jLong *shapeInfo = nullptr;
        Nd4jLong *tadOnlyShapeInfo = nullptr;
        Nd4jLong numTads = 0;
        int tadRank = 0;
        Nd4jLong *tadShape = nullptr;
        Nd4jLong *tadStride = nullptr;
        Nd4jLong *tadOffsets = nullptr;
        Nd4jLong tadOffsetForBlock = 0;
        int rank = 0;
        int numOnes = 0;
        //pointers to original
        int originalDimensionLength;
        int *originalDimension = nullptr;
        Nd4jLong *originalShapeInfo = nullptr;
        bool squeezed = false;
        bool newSqueezeDimensions = false;
        int numOnesInMiddle = 0;
        bool wholeThing = false;
        //need to track whether we create a new dimension array or not, we could have just moved the pointer forward
        //due to leading ones
        bool createdNewDimension = false;

        // special case for CUDA, we're passing in __shared__ memory pointers to be used instead of new/malloc
        void *ptrManager = nullptr;
        int *ptrOutput = nullptr;
#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF TAD() {}

#ifdef __CUDACC__
        __host__ __device__
#endif
        TAD(int tadIndex,Nd4jLong *shapeInfo,int *dimension,int dimensionLength);


#ifdef __CUDACC__
        __host__ __device__
#endif
        TAD(Nd4jLong *shapeInfo,int *dimension,int dimensionLength);


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void setExternalBuffers(void *ptrManager);



#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void setOutputBuffer(int *ptrOutput);

#ifdef __CUDACC__
        __host__ __device__
#endif
        /**
         * This method is for GPU mostly, it allows to initialize TAD instance with precalculated tadOnlyShapeInfo
         */
        INLINEDEF void initWithExternalTAD(Nd4jLong *existingTAD, Nd4jLong *originalShape, int *dimension, int dimensionLength);



#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void init(Nd4jLong *shapeInfo,int *dimension,int dimensionLength);



        template <typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void printTADsND(T *x);



#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void permuteShapeBufferInPlace(Nd4jLong *shapeBuffer, int* rearrange, Nd4jLong *out);

#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong* permuteShapeBuffer(Nd4jLong *shapeBuffer, int *rearrange);




#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void createTadOnlyShapeInfo();


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong lengthPerSlice(Nd4jLong *shapeBuffer);


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong* tad2Sub(Nd4jLong index);



#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF ~TAD();


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF  int* permuteDims();


        /**
        * Compute the tad offset given a dimension.
        *
        * The general pattern for computing a tad offset is as follows:
        * Every $STRIDE that was removed (the first dimension)
        * do a jump by the major stride of the parent array
        * (stride[0] of the parent array)
        *
        * For example given a c ordered 2,2,3,2 with stride 12,6,2,1
        * A tad of dimension 1 will jump 12 every 6 tads.
        *
        * You then end up with offsets of:
        * 0
        * 1
        * 2
        * 3
        * 4
        * 5
        * 12
        * 13
        * 14
        * 15
        * 16
        * 17
        *
        * notice there are 12 tads here. This same incremental jump will happen
        * every time.
        * Note here that by default the
        * stride of element wise stride is used for the hops.
        *
        * Sometimes a jump doesn't happen. If there are less tads
        * than the stride of the dimension you removed, the
        * element wise stride will always be used.
        *
        * For example in a dimension of 0,1, you end up with offsets of:
        * 0,1,2,3,4,5
        *
        * Given that the inner most stride of the dimensions that was removed (1)
        * had a stride of 6, we never need to do a major stride jump.
        *
        */
#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong tadOffset(Nd4jLong index);


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong* tensorShape();

#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong* tad2Sub(Nd4jLong index, void *ptrManager);


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void createOffsets();


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong* shapeInfoOnlyShapeAndStride();



        /**
       * Length of a tad given
       * the shape information
       */
#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong tadLength(Nd4jLong *shapeInfo, int *dimension, int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF Nd4jLong tensorsAlongDimension(Nd4jLong *shapeInfo, int *dimension, int dimensionLength);


#ifdef __CUDACC__
        __host__ __device__
    INLINEDEF void createOffsetForBlock(int blockIdx) {
        this->tadOffsetForBlock = this->tadOffset(blockIdx);
    }
#endif


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void collapse();
    };










    ////

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF TAD::TAD(int tadIndex,Nd4jLong *shapeInfo,int *dimension,int dimensionLength) {
        this->tadIndex = tadIndex;
        this->init(shapeInfo, dimension, dimensionLength);
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF TAD::TAD(Nd4jLong *shapeInfo,int *dimension,int dimensionLength) {
        this->init(shapeInfo, dimension, dimensionLength);
    }

    INLINEDEF void TAD::setExternalBuffers(void *ptrManager) {
        this->ptrManager = ptrManager;
    }

    INLINEDEF void TAD::setOutputBuffer(int *ptrOutput) {
        this->ptrOutput = ptrOutput;
    }

    INLINEDEF void TAD::initWithExternalTAD(Nd4jLong *existingTAD, Nd4jLong *originalShape, int *dimension, int dimensionLength) {
        this->tadOnlyShapeInfo = existingTAD;
        this->rank = shape::rank(originalShape);

        this->originalShapeInfo = originalShape;
        this->originalDimension = dimension;
        this->originalDimensionLength = dimensionLength;

        this->shapeInfo = originalShape;
        this->dimension = dimension;
        this->dimensionLength = dimensionLength;

        this->tadShape = shape::shapeOf(existingTAD);
        this->tadStride = shape::stride(existingTAD);

        Nd4jLong ews = shape::elementWiseStride(originalShape);

        this->numTads = shape::length(originalShape) / shape::length(existingTAD); // this->tensorsAlongDimension(this->shapeInfo, this->dimension, this->dimensionLength);//shape::length(originalShape) / shape::length(existingTAD);
        this->wholeThing = this->numTads == 1 || ((this->dimensionLength == this->rank || this->numTads == shape::length(this->shapeInfo)) && ews == 1);
    }

    INLINEDEF  void TAD::init(Nd4jLong *shapeInfo, int *dimension,int dimensionLength) {
        this->originalShapeInfo = shapeInfo;
        this->originalDimension = dimension;
        this->originalDimensionLength = dimensionLength;
        //start off as original references
        this->shapeInfo = shapeInfo;
        this->dimensionLength = dimensionLength;
        this->dimension = dimension;
        this->rank = shape::rank(shapeInfo);
        this->numTads = dimensionLength == 0 ? 1 : this->tensorsAlongDimension(this->shapeInfo, this->dimension, this->dimensionLength);

        Nd4jLong ews = shape::elementWiseStride(shapeInfo);

        if(!shape::isVector(shapeInfo)) {
            wholeThing = this->numTads == 1 // if number of TADs is 1, we just have input shape == TAD shape
                         || ((this->dimensionLength == this->rank // if number of dimensions is the same as input rank, that'll be wholeTad too, but only if EWS==1 (aka - not a View)
                         || (this->numTads == shape::length(shapeInfo) && shape::order(shapeInfo) == 'c')) // OR  number of tads equals to shapeInfo length AND input is in C order. if order is F - we'll have to calculate offsets
                         && ews == 1); // as mentioned above - last 2 rules apply only to non-views
        } else if(shape::isScalar(shapeInfo)) {
            wholeThing = true;
            //vector case
        } else {
            // if(dimensionLength == 1 && shape::shapeOf(shapeInfo)[dimension[0]] == 1) {
            if(dimension == 0 && shape::shapeOf(shapeInfo)[dimension[0]] == 1) {
                wholeThing = true;
            }
        }
    }

    template <typename T>
    INLINEDEF void TAD::printTADsND(T *x) {
        if(wholeThing) {
            for(int i = 0; i < shape::length(tadOnlyShapeInfo); i++) {
                printf(" %f ",x[i]);
            }
            printf("\n");
        }
        else {
            for (int i = 0; i <  numTads; i++) {
                auto  offset = tadOffsets[i];
                Nd4jLong shapeIter[MAX_RANK];
                Nd4jLong coord[MAX_RANK];
                int dim;
                int rankIter = shape::rank(tadOnlyShapeInfo);
                Nd4jLong xStridesIter[MAX_RANK];
                T *xPointer = x + offset;
                if (PrepareOneRawArrayIter<T>(rankIter,
                                              shape::shapeOf(tadOnlyShapeInfo),
                                              xPointer,
                                              shape::stride(tadOnlyShapeInfo),
                                              &rankIter,
                                              shapeIter,
                                              &xPointer,
                                              xStridesIter) >= 0) {
                    ND4J_RAW_ITER_START(dim, shape::rank(tadOnlyShapeInfo), coord, shapeIter); {
                        /* Process the innermost dimension */
                        printf(" %f ",xPointer[0]);
                    }
                    ND4J_RAW_ITER_ONE_NEXT(dim,
                                           rankIter,
                                           coord,
                                           shapeIter,
                                           xPointer,
                                           xStridesIter);
                    printf("\n");

                }
                else {
                    printf("Unable to prepare array\n");
                }
            }
        }
    }


    INLINEDEF void TAD::permuteShapeBufferInPlace(Nd4jLong* shapeBuffer, int* rearrange, Nd4jLong* out) {
        memcpy(out, shapeBuffer, sizeof(Nd4jLong) * shape::shapeInfoLength(this->rank));
        doPermuteShapeBuffer(this->rank, out, rearrange);
    }

    INLINEDEF Nd4jLong* TAD::permuteShapeBuffer(Nd4jLong* shapeBuffer, int *rearrange) {
        int len = shape::shapeInfoLength(this->rank);
        Nd4jLong *copy = shape::copyOf(len,shapeBuffer);
        doPermuteShapeBuffer(rank, copy,rearrange);
        return copy;
    }

    INLINEDEF void TAD::createTadOnlyShapeInfo() {
        this->tadOnlyShapeInfo = this->shapeInfoOnlyShapeAndStride();
        nd4j::ArrayOptions::setDataType(this->tadOnlyShapeInfo, nd4j::ArrayOptions::dataType(this->originalShapeInfo));

        if (this->tadShape != nullptr)
            delete[] this->tadShape;

        this->tadShape = shape::shapeOf(this->tadOnlyShapeInfo);
        this->tadStride = shape::stride(this->tadOnlyShapeInfo);
        /* if(tadIndex > 0) {
             this->createOffsets();
             this->tadOnlyShapeInfo[shape::shapeInfoLength(shape::rank(this->tadOnlyShapeInfo)) - 3] = this->tadOffsets[tadIndex];
         }*/
    }

    INLINEDEF Nd4jLong TAD::lengthPerSlice(Nd4jLong* shapeBuffer) {
        int dimension = 0;
        Nd4jLong *remove = shape::removeIndex(shape::shapeOf(shapeBuffer),&dimension,shape::rank(shapeBuffer),1);
        Nd4jLong prod = shape::prodLong(remove, shape::rank(shapeBuffer) - 1);
        delete[] remove;
        return prod;
    }


    INLINEDEF Nd4jLong* TAD::tad2Sub(Nd4jLong index) {
        Nd4jLong *shape = shape::shapeOf(shapeInfo);
        int rank = shape::rank(shapeInfo);
        int leftOverIndexLen = rank - originalDimensionLength;
#ifdef __CUDACC__
        Nd4jLong *ret;
        Nd4jLong *tadShape;
        Nd4jLong *leftOverIndexes;
        Nd4jLong *sub;
        if (ptrManager != nullptr) {
            UnifiedSharedMemory *manager = (UnifiedSharedMemory *) ptrManager;
            ret = (Nd4jLong *) manager->getTempRankBuffer1();
            tadShape = (Nd4jLong *) manager->getTempRankBuffer2();
            leftOverIndexes = (Nd4jLong *) manager->getTempRankBuffer3();
            sub =  (Nd4jLong *) manager->getTempRankBuffer4();
        } else {
            ret = new Nd4jLong[rank];
            tadShape = new Nd4jLong[leftOverIndexLen];
            leftOverIndexes = new Nd4jLong[leftOverIndexLen];
            sub = new Nd4jLong[rank];
        }
#else
        Nd4jLong *ret = new Nd4jLong[rank];
        //shape of the tad
        Nd4jLong *tadShape = new Nd4jLong[leftOverIndexLen];
        Nd4jLong *leftOverIndexes = new Nd4jLong[leftOverIndexLen];
        Nd4jLong *sub = new Nd4jLong[rank];
#endif

        //indexes not specified in the tad indexes

        //every coordinate starts as zero
        memset(ret,0, shape::shapeInfoByteLength(rank));

        //find the length of the elements we
        //are iterating over
        int len = 1;
        //left over index cursor for initializing elements
        int leftOverIndex = 0;
        for(int i = 0; i < rank; i++) {
            //look for dimensions NOT found in dimension length (basically compute shape - dimension (set difference)
            bool found = false;
            for(int j = 0; j < originalDimensionLength; j++) {
                //skip over specified dimensions when computing left over length
                if(i == originalDimension[j]) {
                    found = true;
                    break;
                }

            }

            //add to the indexes that aren't specified as part of the tad dimension
            //indexes
            if(!found) {
                //accumulate the list of indexes left over used for initializing the return value
                leftOverIndexes[leftOverIndex] = i;
                //accumulate the tad shape
                tadShape[leftOverIndex] = shape[i];
                //accumulate the length (product) of the indexes that will be iterated over
                len *= shape[i];
                leftOverIndex++;

            }
        }


        //sub for indices
        /* int *sub = new int[leftOverIndexLen];
         shape::ind2subOrder(tadShape,index,len,sub);
        */
        shape::ind2subC(leftOverIndexLen,tadShape, index,len, sub);


        for(int i = 0; i < leftOverIndexLen; i++) {
            ret[leftOverIndexes[i]] = sub[i];
        }

        if (ptrManager == nullptr) {
            delete[] tadShape;
            delete[] leftOverIndexes;
            delete[] sub;
        }

        return  ret;
    }


    INLINEDEF TAD::~TAD() {
        //we may have just moved the pointer forward, we may not need to delete the pointer here
        if(originalDimension != this->dimension && createdNewDimension) {
            delete[] this->dimension;
        }
        if(this->originalShapeInfo != this->shapeInfo) {
            delete[] this->shapeInfo;
        }
        if(this->tadOffsets != nullptr) {
            delete[] this->tadOffsets;
        }

        if(this->tadOnlyShapeInfo != nullptr && this->tadOnlyShapeInfo != shapeInfo) {
            delete[] this->tadOnlyShapeInfo;
        }
    }

    INLINEDEF int* TAD::permuteDims() {
        //permute dimensions for tad
        int dimIdx = 0;
        //loop backwards assuming dimension is sorted

        int *permuteDims = new int[shape::rank(shapeInfo)];

        for(int i = 0; i < shape::rank(shapeInfo); i++) {
            bool found = false;
            for(int j = 0; j < originalDimensionLength; j++) {
                if(i == originalDimension[j]) {
                    found = true;
                    break;
                }
            }

            //not found, append it to the end for permute
            if(!found)
                permuteDims[dimIdx++] = i;
        }



        for(int i = originalDimensionLength - 1; i >= 0; i--) {
            permuteDims[dimIdx++] = originalDimension[i];
        }

/*
            for (int i = 0; i < originalDimensionLength; i++) {
                permuteDims[i] = originalDimension[i];
            }
*/

        //permute dimensions for tad
        return permuteDims;
    }


    INLINEDEF Nd4jLong TAD::tadOffset(Nd4jLong index) {
        if(tadOnlyShapeInfo == nullptr) {
            this->createTadOnlyShapeInfo();
        }

        if(wholeThing)
            return index;

        if(dimensionLength > 1) {
            Nd4jLong *tad2Sub = this->tad2Sub(index, ptrManager);

            Nd4jLong ret = shape::getOffset(0,shape::shapeOf(shapeInfo),shape::stride(shapeInfo),tad2Sub,shape::rank(shapeInfo));

            if(ret < 0) {
                if (ptrManager == nullptr)
                    delete[] tad2Sub;
                return -1;
            }
            if (ptrManager == nullptr)
                delete[] tad2Sub;

            return ret;

        }
        else {
            Nd4jLong *tad2Sub = this->tad2Sub(index, ptrManager);

            Nd4jLong ret = shape::getOffset(0,shape::shapeOf(shapeInfo),shape::stride(shapeInfo),tad2Sub,shape::rank(shapeInfo));

            if (ptrManager == nullptr)
                delete[] tad2Sub;

            return ret;
        }
    }


    INLINEDEF Nd4jLong* TAD::tensorShape() {
        if(this->tadShape != nullptr)
            return this->tadShape;

        Nd4jLong *theShape = shape::shapeOf(shapeInfo);
        Nd4jLong *tensorShape = shape::keep(theShape, this->dimension, dimensionLength,shape::rank(shapeInfo));
        this->tadShape = tensorShape;
        this->tadRank = dimensionLength;
        return tensorShape;
    }

    INLINEDEF Nd4jLong* TAD::tad2Sub(Nd4jLong index, void *ptrManager) {
        auto shape = shape::shapeOf(shapeInfo);
        int rank = shape::rank(shapeInfo);
        int leftOverIndexLen = rank - originalDimensionLength;
        Nd4jLong *tadShape;
        Nd4jLong *leftOverIndexes;
        Nd4jLong *sub;
        Nd4jLong *ret;

#ifdef __CUDACC__

        if (ptrManager != nullptr) {
                UnifiedSharedMemory *manager = (UnifiedSharedMemory *) ptrManager;
                ret = (Nd4jLong *) manager->getTempRankBuffer1();
                tadShape = (Nd4jLong *) manager->getTempRankBuffer2();
                leftOverIndexes = (Nd4jLong *) manager->getTempRankBuffer3();
                sub = (Nd4jLong *) manager->getTempRankBuffer4();
            } else {
                ret = new Nd4jLong[rank];
                //shape of the tad
                leftOverIndexes = new Nd4jLong[leftOverIndexLen];
                sub = new Nd4jLong[rank];
                tadShape = new Nd4jLong[leftOverIndexLen];
            }
#else
        ret = new Nd4jLong[rank];
        //shape of the tad
        leftOverIndexes = new Nd4jLong[leftOverIndexLen];
        sub = new Nd4jLong[rank];
        tadShape = new Nd4jLong[leftOverIndexLen];
#endif

        //indexes not specified in the tad indexes

        //every coordinate starts as zero
        memset(ret,0,sizeof(Nd4jLong) * rank);


        //find the length of the elements we
        //are iterating over
        int len = 1;
        //left over index cursor for initializing elements
        int leftOverIndex = 0;
        for(int i = 0; i < rank; i++) {
            //look for dimensions NOT found in dimension length (basically compute shape - dimension (set difference)
            bool found = false;
            for(int j = 0; j < originalDimensionLength; j++) {
                //skip over specified dimensions when computing left over length
                if(i == originalDimension[j])  {
                    found = true;
                    break;
                }

            }

            //add to the indexes that aren't specified as part of the tad dimension
            //indexes
            if(!found) {
                //accumulate the list of indexes left over used for initializing the return value
                leftOverIndexes[leftOverIndex] = i;
                //accumulate the tad shape
                tadShape[leftOverIndex] = shape[i];
                //accumulate the length (product) of the indexes that will be iterated over
                leftOverIndex++;
                len *= shape[i];

            }
        }


        //sub for indices
        /* int *sub = new int[leftOverIndexLen];
         shape::ind2subOrder(tadShape,index,len,sub);
        */
        shape::ind2subC(leftOverIndexLen,tadShape,index,len, sub);

        for(int i = 0; i < leftOverIndexLen; i++) {
            ret[leftOverIndexes[i]] = sub[i];
        }

        if (ptrManager == nullptr) {
            delete[] leftOverIndexes;
            delete[] tadShape;
            delete[] sub;
        }

        return  ret;
    }

    INLINEDEF void TAD::createOffsets() {
        this->tadOffsets = new Nd4jLong[this->numTads];
#pragma omp parallel for if (this->numTads > 128) schedule(static) proc_bind(close) default(shared)
        for(int i = 0; i < this->numTads; i++) {
            this->tadOffsets[i] = this->tadOffset(i);
        }
    }


    INLINEDEF Nd4jLong* TAD::shapeInfoOnlyShapeAndStride() {
        if(wholeThing && (dimensionLength == 1 && dimension[0] == MAX_DIMENSION) || shape::isScalar(shapeInfo))
            return shape::createScalarShapeInfo();

        //ensure tad shapes get setup right for vectors
        if(dimensionLength > 1 && shape::isVector(shapeInfo)) 
            return shape::copyOf(shape::shapeInfoLength(shape::rank(shapeInfo)),shapeInfo);

        // case when tad coincides with whole array
        if( this->numTads == 1 && ((shape::rank(originalShapeInfo) == originalDimensionLength) || originalDimensionLength == 0)) {
            // we might have special case here: skipped dimensions might be just full of ones
            Nd4jLong *ret = shape::copyOf(shape::shapeInfoLength(shape::rank(shapeInfo)), shapeInfo);
            if (shape::isDimPermuted<int>(dimension, (Nd4jLong) dimensionLength))    // check whether we need permutation
                shape::doPermuteShapeBuffer(ret, dimension);

            return ret;
        }

        Nd4jLong *theShape = shape::shapeOf(shapeInfo);
        int rank = shape::rank(shapeInfo);

        if(dimensionLength == 1) {
            if(dimension[0] == 0 && shape::isVector(shapeInfo) && theShape[1] == 1) {
                int permuted[2] = {1,0};
                Nd4jLong *permutedRet2 = shape::permuteShapeBuffer(shapeInfo, permuted);
                return permutedRet2;
            } else if(dimension[0] == 1 && shape::isVector(shapeInfo) && theShape[0] == 1) {
                Nd4jLong *ret = shape::copyOf(shape::shapeInfoLength(shape::rank(shapeInfo)),shapeInfo);
                return ret;
            }
            else if(shape::shapeOf(shapeInfo)[dimension[0]] == 1) {
                Nd4jLong *scalarInfo = shape::createScalarShapeInfo();
                scalarInfo[shape::shapeInfoLength(shape::rank(scalarInfo)) - 3] = this->tadIndex;
                return scalarInfo;
            }
        }

        Nd4jLong *tensorShape = this->tensorShape();
        int *reverseDimensions = shape::reverseCopy(dimension, dimensionLength);
        int *rankRange = shape::range<int>(0, rank);
        int *remove  = shape::removeIndex<int>(rankRange, dimension, (Nd4jLong) rank, (Nd4jLong) dimensionLength);
        //concat is wrong here with the length
        int *newPermuteDims = shape::concat(remove,rank - dimensionLength,reverseDimensions,dimensionLength);

        Nd4jLong* permuted = shape::permuteShapeBuffer(shapeInfo,newPermuteDims);


        Nd4jLong sliceIndex = shape::sliceOffsetForTensor(shape::rank(permuted),
                                                     this->tadIndex,
                                                     shape::shapeOf(shapeInfo),
                                                     tensorShape,
                                                     dimensionLength,
                                                     dimension,
                                                     dimensionLength);



        Nd4jLong *ret2 = shape::sliceOfShapeBuffer(sliceIndex, permuted);
        Nd4jLong tensorLength = shape::prodLong(tensorShape,tadRank);

        int compLength = shape::isVector(ret2) ? shape::length(ret2) : shape::prod(tensorShape,tadRank);
        // int temp;
        // const bool isLikeVector = shape::isLikeVector(ret2, temp);

        // if(dimensionLength == tadRank && compLength == shape::length(ret2) && !isLikeVector) {
        if(dimensionLength == tadRank && compLength == shape::length(ret2)) {
            if(dimensionLength == 1 && shape::isVector(ret2) && shape::shapeOf(ret2)[0] == 1) {
                //go to the bottom and return ret2 after proper freeing of pointers
                //basic idea; we *don't* permute row vectors
            }
            else if(dimensionLength > 1) {
                //permute *then* return ret2
                int *finalPermuteDims = new int[shape::rank(ret2)];
                int forward = 0;
                for(int i = shape::rank(ret2) - 1; i >= 0; i--) {
                    finalPermuteDims[forward++] = i;
                }
                shape::permuteShapeBufferInPlace(ret2,finalPermuteDims,ret2);
                delete[] finalPermuteDims;

            }

        }
        else {
            Nd4jLong length = tensorLength;
            Nd4jLong lengthPerSlice = this->lengthPerSlice(ret2);
            if(lengthPerSlice < 1) {
                return ret2;
            }

            Nd4jLong offset = tadIndex * tensorLength /lengthPerSlice;
            if(sliceIndex == 0 && length == lengthPerSlice) {
                Nd4jLong *newRet2 = shape::sliceOfShapeBuffer(offset, ret2);
                delete[] ret2;
                ret2 = newRet2;
                int *finalPermuteDims = new int[shape::rank(ret2)];
                int forward = 0;
                for(int i = shape::rank(ret2) - 1; i >= 0; i--) {
                    finalPermuteDims[forward++] = i;
                }
                // bool isRowVector2 = shape::isRowVector(ret2) && !isLikeVector;
                bool isRowVector2 = shape::isRowVector(ret2);
                if(isRowVector2 == false) {
                    shape::permuteShapeBufferInPlace(ret2, finalPermuteDims, ret2);
                }

                delete[] finalPermuteDims;

            }
            else if(length == lengthPerSlice) {
                offset -= shape::slices(ret2) * (offset / shape::slices(ret2));
                Nd4jLong *newRet2 = shape::sliceOfShapeBuffer(offset,ret2);
                delete[] ret2;
                ret2 = newRet2;
                if(dimensionLength == 1 && shape::isVector(ret2) && shape::shapeOf(ret2)[0] == 1) {
                    //go to the bottom and return ret2 after proper freeing of pointers
                    //basic idea; we *don't* permute row vectors
                }
                else {
                    int *finalPermuteDims = new int[shape::rank(ret2)];
                    int forward = 0;
                    for(int i = shape::rank(ret2) - 1; i >= 0; i--) {
                        finalPermuteDims[forward++] = i;
                    }
                    Nd4jLong *newRet = shape::permuteShapeBuffer(ret2, finalPermuteDims);
                    delete[] ret2;
                    delete[] finalPermuteDims;
                    ret2 = newRet;

                }

            }
            else {
                //execute final part, note that this is mainly so delete[] gets called
                //at the bottom of the method
                while(shape::length(ret2) > length) {
                    auto lengthPerSlice2 = this->lengthPerSlice(ret2);
                    sliceIndex =    sliceOffsetForTensor(sliceIndex,shape::length(ret2),lengthPerSlice2);
                    sliceIndex -= shape::slices(ret2) * (sliceIndex / shape::slices(ret2));
                    auto newRet2 = shape::sliceOfShapeBuffer(sliceIndex,ret2);
                    delete[] ret2;
                    ret2 = newRet2;
                }

                //don't permute on a row vector
                if(dimensionLength == 1 &&  shape::isVector(ret2) && shape::shapeOf(ret2)[0] == 1) {
                    //go to the bottom and return ret2 after proper freeing of pointers
                    //basic idea; we *don't* permute row vectors
                }
                else if(dimensionLength > 1){
                    //permute *then* return ret
                    int *finalPermuteDims = new int[shape::rank(ret2)];
                    int forward = 0;
                    for(int i = shape::rank(ret2) - 1; i >= 0; i--) {
                        finalPermuteDims[forward++] = i;
                    }
                    auto newPermute = shape::permuteShapeBuffer(ret2,finalPermuteDims);
                    delete[] ret2;
                    delete[] finalPermuteDims;
                    ret2 = newPermute;
                }

            }
        }


        delete[] permuted;
        delete[] newPermuteDims;
        delete[] rankRange;
        delete[] remove;
        delete[] reverseDimensions;
        return ret2;
    }


    INLINEDEF Nd4jLong TAD::tadLength(Nd4jLong *shapeInfo, int *dimension, int dimensionLength) {
        if(dimensionLength == 1) {
            return shape::shapeOf(shapeInfo)[dimension[0]];
        }
        else {
            Nd4jLong ret = 1;
            for(int i = 0; i < shape::rank(shapeInfo); i++) {
                for(int j = 0; j < dimensionLength; j++) {
                    if(i == dimension[j])
                        ret *= shape::shapeOf(shapeInfo)[dimension[j]];
                }
            }
            return ret;
        }
    }


    INLINEDEF Nd4jLong TAD::tensorsAlongDimension(Nd4jLong *shapeInfo, int *dimension, int dimensionLength) {
        return shape::length(shapeInfo) / this->tadLength(shapeInfo,dimension,dimensionLength);
    }


    INLINEDEF void TAD::collapse() {
        auto shape = shape::shapeOf(shapeInfo);
        //handle negative dimensions/backwards indexing
        for(int i = 0; i < dimensionLength; i++) {
            if((dimension)[i] < 0)
                (dimension)[i] += shape::rank(this->shapeInfo);
        }

        this->dimension =  new int[dimensionLength];
        memcpy(this->dimension,this->originalDimension, sizeof(int) * dimensionLength);

        //we can drop trailing dimensions where it's all singular for example:
        // shape: 4,3,1,2
        //dimension: 0,2
        // the problem for 0,2 is equivalent to: 0
        //the rest of the algorithm handles cases suchas
        //shape: 4,1,1,2
        //dimension: 0,1
        //when this happens there are other dimensions (eg: at the end) that matter
        int trailingOneDimensions = 0;
        //trailing ones
        for(int i = dimensionLength - 1; i >= 0; i--) {
            if(shape[dimension[i]] != 1) {
                break;
            }
            else if(shape[dimension[i]] == 1)
                trailingOneDimensions++;
        }

        dimensionLength -= trailingOneDimensions;

        int leadingOneDimensions = 0;
        //trailing ones
        for(int i = 0; i < dimensionLength; i++) {
            if(shape[dimension[i]] != 1) {
                break;
            }
            else if(shape[dimension[i]] == 1)
                leadingOneDimensions++;
        }

        //bump the dimension pointer forward for however many leadingones there are
        dimension += leadingOneDimensions;
        //decrease the dimension length by the amount of leading ones
        dimensionLength -= leadingOneDimensions;


        bool preConverged = true;
        for(int i = 0; i < dimensionLength; i++) {
            if(shape[dimension[i]] == 1) {
                preConverged = false;
                break;
            }
        }

        //we took away all the singular dimensions, we can just return
        if(preConverged)
            return;

        //no more singular dimensions specified
        bool done = false;
        int onesDecrement = 0;
        bool changed = false;
        while(!done) {
            //terminate early: only singular dimensions specified for reduce
            if((dimensionLength) < 1) {
                done = true;
                //signal as a no op
                dimension[0] = -1;
                break;
            }
            //captures intermediary result from the for loop
            traceNew(3);

            int intermediaryResult[MAX_RANK];
            for(int i = 0; i < dimensionLength; i++) {
                intermediaryResult[i] = (dimension)[i];
            }

            bool oneEncountered = false;
            bool nonOneEncountered = false;
            bool hitBeginning = false;
            //assume intermediate collapsing of dimensions
            bool collapseMiddleDimensions = true;
            //note here that dimension length MAY end up being zero
            for(int i = (dimensionLength) - 1; i >= 0; i--) {
                if(shape[(dimension)[i]] == 1) {
                    oneEncountered = true;
                    //trailing ones
                    if(!nonOneEncountered) {
                        //just drop trailing ones
                        dimensionLength--;
                        nonOneEncountered = false;
                        collapseMiddleDimensions = false;
                        //intermediary result just needs to have the results copied from dimension since we're just removing the tail
                        memcpy(intermediaryResult,dimension, sizeof(int) * dimensionLength);
                        changed = true;
                        //break the for loop and force it to go back around starting from the new index
                        break;
                    }
                    else {
                        //already decremented all dimensions
                        //this was a result of hitting beginning ones
                        //we will only need to loop once
                        if(i == 0) {
                            hitBeginning = true;
                        }
                        //will need to shift dimensions that aren't trailing ones
                        //back by onesDecrement
                        //mark the intermediary result as -1 for non inclusion
                        intermediaryResult[i] = -1;
                        onesDecrement++;
                    }
                }
                else {
                    intermediaryResult[i] = (dimension)[i];
                    nonOneEncountered = true;
                }
            }

            if(collapseMiddleDimensions && oneEncountered) {
                //collapse dimensions
                int newIntermediary[MAX_RANK];
                int idx = 0;
                for(int i = 0; i < dimensionLength; i++) {
                    //of note: dimension will decrease by the number of ones encountered
                    if(intermediaryResult[i] >= 0) {
                        //dimension 0 doesn't need to be decremented
                        if(intermediaryResult[i] > 0)
                            newIntermediary[idx++] = intermediaryResult[i] - onesDecrement;
                        else
                            newIntermediary[idx++] = intermediaryResult[i];
                    }
                }


                //decrement by the number of dimensions where ones appeared
                (dimensionLength) -= onesDecrement;
                //update to current result
                memcpy(dimension,newIntermediary, sizeof(int) * (dimensionLength));
                changed = true;

            }
                //converged: no need to change result
            else {
                //update to current result
                memcpy(dimension,intermediaryResult, sizeof(int) * dimensionLength);
            }

            //converge when there are no singular dimensions specified in the reduce
            done = (!oneEncountered && nonOneEncountered) || hitBeginning;
            //delete[] intermediaryResult;
        }

        //nothing changed but need to collapse dimension
        if(!changed && this->numOnes > 0) {
            for(int i = 0; i < dimensionLength ;i++) {
                dimension[i] -= numOnes;
            }
        }


    }
}

#endif //LIBND4J_TAD_H
