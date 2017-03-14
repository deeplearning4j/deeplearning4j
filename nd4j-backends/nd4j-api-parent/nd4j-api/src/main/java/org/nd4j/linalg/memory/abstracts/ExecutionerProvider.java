package org.nd4j.linalg.memory.abstracts;

import lombok.extern.slf4j.Slf4j;

/**
 * This class is a thin wrapper/holder for initialized in different contexts OpExecutioner implementations
 * Main idea behind this thing: OpExecutioner being NUMA-aware & CUDA-aware & detached from host master thread.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class ExecutionerProvider {
}
