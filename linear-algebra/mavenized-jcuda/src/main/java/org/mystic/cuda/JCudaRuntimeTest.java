package org.mystic.cuda;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

/**
 * Hello, World in JCuda
 * Taken from http://www.jcuda.org/tutorial/TutorialIndex.html
 */
public class JCudaRuntimeTest {
    public static void main(String args[]) {
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: " + pointer);
        JCuda.cudaFree(pointer);
    }
}