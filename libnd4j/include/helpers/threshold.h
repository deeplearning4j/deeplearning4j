//
// Created by raver119 on 01.06.17.
//

#ifndef LIBND4J_THRESHOLD_H
#define LIBND4J_THRESHOLD_H

#include <loops/type_conversions.h>


void prescanArrayRecursive(Nd4jPointer *extras, int *z, int *x, int numElements, int level) {

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extras[1]);
    int **g_scanBlockSums = reinterpret_cast<int **>(&extras[2]);

    unsigned int blockSize = 512; // max size of the thread blocks
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock =
            numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;

    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);

        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(int) * (numEltsPerBlock + extraSpace);

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3 grid(max(1, numBlocks - np2LastBlock), 1, 1);
    dim3 threads(numThreads, 1, 1);

    if (sharedMemSize < 2048)
        sharedMemSize = 2048;

    if (sharedMemLastBlock < 2048)
        sharedMemLastBlock = 2048;

    // make sure there are no CUDA errors before we start
    //CUT_CHECK_ERROR("prescanArrayRecursive before kernels");

    // execute the scan
    if (numBlocks > 1) {
        prescan<true, false><<<grid, threads, sharedMemSize, *stream>>>(z, x, g_scanBlockSums[level], numThreads * 2, 0, 0);
        //CUT_CHECK_ERROR("prescanWithBlockSums");
        if (np2LastBlock) {
            prescan<true, true><<<1, numThreadsLastBlock, sharedMemLastBlock, *stream>>>(z, x, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
            //CUT_CHECK_ERROR("prescanNP2WithBlockSums");
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we
        // need to take all of the last values of the sub-blocks and scan those.
        // This will give us a new value that must be sdded to each block to
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(extras, g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level+1);

        uniformAdd<<<grid, threads, 1024, *stream>>>(z, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);
        //CUT_CHECK_ERROR("uniformAdd");
        if (np2LastBlock) {
            uniformAdd<<<1, numThreadsLastBlock, 1024, *stream>>>(z, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
            //CUT_CHECK_ERROR("uniformAdd");
        }
    } else if (isPowerOfTwo(numElements)) {
        prescan<false, false><<<grid, threads, sharedMemSize, *stream>>>(z, x, 0, numThreads * 2, 0, 0);
        //CUT_CHECK_ERROR("prescan");
    } else {
        prescan<false, true><<<grid, threads, sharedMemSize, *stream>>>(z, x, 0, numElements, 0, 0);
        //CUT_CHECK_ERROR("prescanNP2");
    }
}




#endif //LIBND4J_THRESHOLD_H
