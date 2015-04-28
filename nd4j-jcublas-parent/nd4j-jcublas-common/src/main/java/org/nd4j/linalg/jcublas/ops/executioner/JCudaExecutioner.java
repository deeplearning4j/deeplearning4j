/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jcublas.ops.executioner;

import static jcuda.driver.JCudaDriver.cuMemGetInfo;

import java.util.HashSet;
import java.util.Set;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.ArrayUtil;


/**
 * JCuda executioner.
 * <p/>
 * Runs ops directly on the gpu
 *
 * @author Adam Gibson
 */
public class JCudaExecutioner implements OpExecutioner {
    private JCudaBuffer dummyFloatPointer, dummyDoublePointer;

    public JCudaExecutioner() {
        SimpleJCublas.init();
        dummyFloatPointer = KernelFunctions.alloc(new float[]{1});
        dummyDoublePointer =KernelFunctions.alloc(new double[]{1});
    }
    
    private static class PreparedKernelParams implements AutoCloseable {
    	final public Object[] kernelParameters;
    	private Set<JCudaBuffer> toFree;
    	
    	private Object[] getKernelParameters() {
    		return kernelParameters;
    	}
    	
    	public PreparedKernelParams(Object... kernelParams) {
    		kernelParameters = new Object[kernelParams.length];
    		toFree = new HashSet<>();
    		for(int i = 0; i<kernelParams.length; i++) {
    			Object arg = kernelParams[i];
	    		if(arg instanceof JCudaBuffer) {
	            	
	            	JCudaBuffer bufferToFree = (JCudaBuffer)arg;
					kernelParameters[i] = bufferToFree.getDevicePointer();
	            	
					checkResult(JCuda.cudaMemcpy(bufferToFree.getDevicePointer(), bufferToFree.getHostPointer(), bufferToFree.getLength()*bufferToFree.getElementSize(), cudaMemcpyKind.cudaMemcpyHostToDevice));
	            	
	            	toFree.add(bufferToFree);
	            } else {
	            	kernelParameters[i] = arg;
	            }
    		}
    	}

		@Override
		public void close() throws Exception {
			for(JCudaBuffer buffer : toFree) {
	        	buffer.freeDevicePointer();
	        }
	        
	        long[] free = new long[1];
	        long[] total = new long[1];
	        checkResult(cuMemGetInfo(free, total));
		}

    	
    }

    @Override
    public Op exec(Op op) {
        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t);
        } else if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            invoke(acc);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            invoke(sc);
        }
        return op;
    }

    @Override
    public void iterateOverAllRows(Op op) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void iterateOverAllColumns(Op op) {
        throw new UnsupportedOperationException();

    }


    private JCudaBuffer dummyDouble() {
        return dummyDoublePointer;
    }

    private JCudaBuffer dummyFloat() {
        return dummyFloatPointer;
    }

    @Override
    public INDArray execAndReturn(TransformOp op) {
        invoke(op);
        return op.z();
    }


    @Override
    public Accumulation execAndReturn(Accumulation op) {
        return (Accumulation) exec(op);
    }

    @Override
    public INDArray execAndReturn(ScalarOp op) {
        return exec(op).z();
    }


    @Override
    public Op exec(Op op, int dimension) {
        //only accumulate along a particular dimension
        if (op instanceof Accumulation) {
            Accumulation a = (Accumulation) op;
            return exec(a);
        }
        for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i, dimension);
            exec(op2);
            if (op instanceof TransformOp) {
                TransformOp t = (TransformOp) op;
                TransformOp t2 = (TransformOp) op2;
                t.z().vectorAlongDimension(i, dimension).assign(t2.z());
            }


        }
        return op;
    }

    @Override
    public INDArray exec(Accumulation op, int dimension) {
    	if(dimension == Integer.MAX_VALUE) {
            if(op.x() instanceof IComplexNDArray)
                return Nd4j.scalar(execAndReturn(op).currentResultComplex());
            else
                return Nd4j.scalar(execAndReturn(op).currentResult());
        }
        else if(op.x().isScalar())
            return op.x();
        if(op.x() instanceof IComplexNDArray) {
            IComplexNDArray ret = Nd4j.createComplex(ArrayUtil.removeIndex(op.x().shape(), dimension));
            IComplexNDArray linear = ret.linearView();
            if(op.x().isRowVector()) {
                //same shape
                if(dimension == 0) {
                    //no reduction
                    return op.x();
                }
                else if(dimension == 1) {
                    return Nd4j.scalar(execAndReturn(op).currentResult());
                }
            }
            else if(op.x().isColumnVector()) {
                if(dimension == 0) {
                    return Nd4j.scalar(execAndReturn(op).currentResult());

                }
                //row vector
                else if(dimension == 1) {
                    //make a row vector
                    return Nd4j.scalar(execAndReturn(op).currentResult());

                }
            }

            for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).currentResultComplex();
                linear.putScalar(i, result);

            }

            return ret;
        }
        else {
            if(op.x().isRowVector()) {
                //same shape
                if(dimension == 0) {
                    //no reduction
                    return op.x();
                }
                else if(dimension == 1) {
                    return Nd4j.scalar(execAndReturn(op).currentResult());
                }
            }
            else if(op.x().isColumnVector()) {
                if(dimension == 0) {
                    return Nd4j.scalar(execAndReturn(op).currentResult());

                }
                //row vector
                else if(dimension == 1) {
                    //make a row vector
                    return Nd4j.scalar(execAndReturn(op).currentResult());

                }
            }

            INDArray ret = Nd4j.create(ArrayUtil.removeIndex(op.x().shape(), dimension));
            INDArray linear = ret.linearView();

            for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                Number result = execAndReturn((Accumulation) op2).currentResult();
                linear.putScalar(i,result.doubleValue());

            }

            return ret;

        }
    }


    @Override
    public INDArray execAndReturn(TransformOp op, int dimension) {
        for (int i = 0; i < op.x().vectorsAlongDimension(dimension); i++) {
            Op op2 = op.opForDimension(i, dimension);
            exec(op2);
            if (op instanceof TransformOp) {
                TransformOp t = op;
                TransformOp t2 = (TransformOp) op2;
                t.z().vectorAlongDimension(i, dimension).assign(t2.z());
            }


        }
        return op.z();
    }

    @Override
    public INDArray execAndReturn(ScalarOp op, int dimension) {
        return exec(op, dimension).z();
    }

    /**
     * Converts the given parameters
     * in to extra arguments to
     * pass to the kernel
     *
     * @param extraArgs the extra arguments
     * @param dataType  the data type
     * @return
     */
    private JCudaBuffer toArgs(Object[] extraArgs, String dataType) {
        if (dataType.equals("double")) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyDouble();
            return KernelFunctions.alloc(PointerUtil.toDoubles(extraArgs));
        } else if (dataType.equals("float")) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyFloat();
            return KernelFunctions.alloc(PointerUtil.toFloats(extraArgs));
        }
        throw new IllegalArgumentException("Illegal datatype");
    }


    private void invoke(Accumulation op)  {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        //Pointer xPointer = xBuffer.getHostPointer().withByteOffset(xBuffer.getElementSize() * op.x().offset());
        
	    JCudaBuffer result = null;
	   
        int resultLength = 32;
        if (op.x().data().dataType() == DataBuffer.DOUBLE) {
//	            double[] resultBuffer = new double[resultLength];
//	            for (int i = 0; i < resultBuffer.length; i++)
//	                resultBuffer[i] = op.zero().doubleValue();
            result = new CudaDoubleDataBuffer(resultLength);


        } else {
            result = new CudaFloatDataBuffer(resultLength);
        }

        if (op.y() != null) {
            JCudaBuffer yBuffer = (JCudaBuffer) op.y().data();
            //Pointer yPointer = yBuffer.getHostPointer().withByteOffset(op.y().offset() * yBuffer.getElementSize());

            //int n,int xOffset,int yOffset, double *dx, double *dy,int incx,int incy,double *result
            Object[] kernelParams = new Object[] {
                    new int[]{op.n()},
                    new int[]{op.x().offset()},
                    new int[]{op.y().offset()},
                    xBuffer,
                    yBuffer,
                    new int[]{op.x().majorStride()},
                    new int[]{op.y().majorStride()},
                    toArgs(op.extraArgs(), getType(op)),
                    result
            };
            
            try(PreparedKernelParams kParams = new PreparedKernelParams(kernelParams)) {
	            
	            invokeFunction(op, kParams.getKernelParameters());
	            setResultForOp(op, Pointer.to(result.getDevicePointer()));
            } catch(Exception e) {
            	throw new RuntimeException("Could not execute kernel", e);
            }

            


        } else {
            //int n, int xOffset,double *dx,int incx,double result
            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    xBuffer,
                    op.x().majorStride(),
                    toArgs(op.extraArgs(), getType(op)),
                    result
            };

            try(PreparedKernelParams kParams = new PreparedKernelParams(kernelParams)) {
	            
	            invokeFunction(op, kParams.getKernelParameters());
	            setResultForOp(op, result.getDevicePointer());
            } catch(Exception e) {
            	throw new RuntimeException("Could not execute kernel", e);
            }
        	


        }
    }


    private void invokeFunction(Op op, Object... kernelParams) {
        String functionName = op instanceof TransformOp || op instanceof Accumulation ? op.name() + "_strided" : op.name();
        int blocks = PointerUtil.getNumBlocks(op.n(), KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(op.n(), KernelFunctions.THREADS);
        KernelFunctions.invoke(blocks,threads,functionName,getType(op),kernelParams);

    }
    
    

    private void setResultForOp(Accumulation acc, Pointer devicePointer) {
        JCudaBuffer buff = (JCudaBuffer) acc.x().data();

        if (buff.dataType() == DataBuffer.DOUBLE) {
            double[] data = new double[1];
            Pointer get = Pointer.to(data);
            JCuda.cudaMemcpy(get, devicePointer, Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            acc.setCurrentResult(data[0]);
        } else {
            float[] data = new float[1];
            Pointer get = Pointer.to(data);
            JCuda.cudaMemcpy(get, devicePointer, Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            acc.setCurrentResult(data[0]);
        }
    }


    private void invoke(ScalarOp op) {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        //Pointer xPointer = xBuffer.getHostPointer().withByteOffset(op.x().offset() * xBuffer.getElementSize());

        JCudaBuffer zBuffer = (JCudaBuffer) op.z().data();
        //Pointer zPointer = zBuffer.getHostPointer().withByteOffset(zBuffer.getElementSize() * op.z().offset());

        if (op.y() != null) {
            JCudaBuffer yBuffer = (JCudaBuffer) op.y().data();
            Pointer yPointer = yBuffer.getHostPointer().withByteOffset(yBuffer.getElementSize() * op.y().offset());
            Object[] kernelParams = new Object[]{
                    new int[]{op.n()},
                    new int[]{op.x().offset()},
                    new int[]{op.y().offset()},
                    xBuffer,
                    yPointer,
                    new int[]{op.x().majorStride()},
                    new int[]{op.y().majorStride()},
                    toArgs(op.extraArgs(), getType(op)),
                    zBuffer
            };

            try(PreparedKernelParams kParams = new PreparedKernelParams(kernelParams)) {
	            
	            invokeFunction(op, kParams.getKernelParameters());
            } catch(Exception e) {
            	throw new RuntimeException("Could not execute kernel", e);
            }


        } else {
            //int n,int idx,double *dy,int incy,double *result
            //int n, int idx,double dx,double *dy,int incy,double *result

            Object[] kernelParams = new Object[]{
                    new int[]{op.n()},
                    new int[]{op.x().offset()},
                    PointerUtil.getPointer(op),
                    xBuffer,
                    new int[]{op.x().majorStride()},
                    toArgs(op.extraArgs(), getType(op)),
                    zBuffer
            };

            try(PreparedKernelParams kParams = new PreparedKernelParams(kernelParams)) {
	            
	            invokeFunction(op, kParams.getKernelParameters());
            } catch(Exception e) {
            	throw new RuntimeException("Could not execute kernel", e);
            }


        }

    }


    private String getType(Op op) {
        return op.x().data().dataType() == DataBuffer.DOUBLE ? "double" : "float";
    }


    private void invoke(TransformOp op) {
        JCudaBuffer xBuffer = (JCudaBuffer) op.x().data();
        //Pointer xPointer = xBuffer.getHostPointer().withByteOffset(xBuffer.getElementSize() * op.x().offset());

        JCudaBuffer zBuffer = (JCudaBuffer) op.z().data();
        //Pointer zPointer = zBuffer.getHostPointer().withByteOffset(zBuffer.getElementSize() * op.z().offset());

        if (op.y() != null) {
            JCudaBuffer yBuffer = (JCudaBuffer) op.y().data();
            //Pointer yPointer = yBuffer.getHostPointer().withByteOffset(op.y().offset() * yBuffer.getElementSize());
            
	            /**
	             * Construct pointer arguments in the following order:
	             * n
	             * offset,
	             * pointer to buffer
	             * increment,
	             * extraArgs,
	             * result
	             */
            
        	Object[] kernelParams = new Object[]{
        			op.n(),
        			op.x().offset(),
        			op.y().offset(),
        			xBuffer,
        			yBuffer,
        			op.x().majorStride(),
        			op.y().majorStride(),
        			toArgs(op.extraArgs(), getType(op)),
        			zBuffer
        	};
        	
        	try(PreparedKernelParams kParams = new PreparedKernelParams(kernelParams)) {
        		invokeFunction(op, kParams.getKernelParameters());
        		zBuffer.copyToHost();
        	} catch(Exception e) {
            	throw new RuntimeException("Could not execute kernel", e);
            }


        } else {
            //int n,int idx,double *dy,int incy,double *result
            Object[] kernelParams = new Object[]{
                    new int[]{op.n()},
                    new int[]{op.x().offset()},
                    xBuffer,
                    new int[]{op.x().majorStride()},
                    toArgs(op.extraArgs(), getType(op)),
                    zBuffer
            };

            try(PreparedKernelParams kParams = new PreparedKernelParams(kernelParams)) {
        		invokeFunction(op, kParams.getKernelParameters());
        	} catch(Exception e) {
            	throw new RuntimeException("Could not execute kernel", e);
            }
            


        }

    }
    
    private static int checkResult(int result)
    {
        if (result != cudaError.cudaSuccess)
        {
            throw new CudaException(cudaError.stringFor(result));
        }
        return result;
    }


}


