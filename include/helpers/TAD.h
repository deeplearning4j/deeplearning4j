//
// @author Adam Gibson
//

#ifndef LIBND4J_TAD_H
#define LIBND4J_TAD_H


#include <shape.h>
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
        int tadIndex = 0;
        int dimensionLength;
        int *dimension = nullptr;
        int *shapeInfo = nullptr;
        int *tadOnlyShapeInfo = nullptr;
        int numTads = 0;
        int tadRank = 0;
        int *tadShape = nullptr;
        int *tadStride = nullptr;
        Nd4jIndex *tadOffsets = nullptr;
        int tadOffsetForBlock = 0;
        int rank = 0;
        int numOnes = 0;
        //pointers to original
        int originalDimensionLength;
        int *originalDimension = nullptr;
        int *originalShapeInfo = nullptr;
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
        TAD() {}

#ifdef __CUDACC__
        __host__ __device__
#endif
        TAD(int tadIndex,int *shapeInfo,int *dimension,int dimensionLength);


#ifdef __CUDACC__
        __host__ __device__
#endif
        TAD(int *shapeInfo,int *dimension,int dimensionLength);


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
        INLINEDEF void initWithExternalTAD(int *existingTAD, int *originalShape, int *dimension, int dimensionLength);



#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void init(int *shapeInfo,int *dimension,int dimensionLength);



        template <typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif
        void printTADsND(T *x);



#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF void permuteShapeBufferInPlace(int *shapeBuffer,int *rearrange,int *out);

#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF int *permuteShapeBuffer(int *shapeBuffer,int *rearrange);




#ifdef __CUDACC__
        __host__ __device__
#endif
        void createTadOnlyShapeInfo();


#ifdef __CUDACC__
        __host__ __device__
#endif
        int lengthPerSlice(int *shapeBuffer);


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF int * tad2Sub(int index);



#ifdef __CUDACC__
        __host__ __device__
#endif
        ~TAD();


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
        INLINEDEF Nd4jIndex tadOffset(int index);


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF int *tensorShape();

#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF int * tad2Sub(int index, void *ptrManager);


#ifdef __CUDACC__
        __host__ __device__
#endif
        void createOffsets();


#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF int *shapeInfoOnlyShapeAndStride();



        /**
       * Length of a tad given
       * the shape information
       */
#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF int tadLength(int *shapeInfo, int *dimension, int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
        __host__ __device__
#endif
        INLINEDEF int tensorsAlongDimension(int *shapeInfo, int *dimension, int dimensionLength);


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
}

#endif //LIBND4J_TAD_H
