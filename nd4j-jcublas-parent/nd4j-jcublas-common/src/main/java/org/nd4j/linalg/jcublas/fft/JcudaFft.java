package org.nd4j.linalg.jcublas.fft;

import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorFFT;
import org.nd4j.linalg.api.ops.impl.transforms.VectorIFFT;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.DefaultFFTInstance;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.ComplexNDArrayUtil;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author Adam Gibson
 */
public class JcudaFft extends DefaultFFTInstance {
    private final Map<String,cufftHandle> handles;
    public JcudaFft() {
        this.handles = new ConcurrentHashMap<>();
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                for (cufftHandle handle : handles.values()) {
                    JCufft.cufftDestroy(handle);
                }
            }
        }));

        JCufft.setExceptionsEnabled(true);

    }


    public cufftHandle getHandle() {
        cufftHandle handle =  handles.get(Thread.currentThread().getName());
        if(handle == null) {
            handle = new cufftHandle();
            JCufft.cufftCreate(handle);
            handles.put(Thread.currentThread().getName(), handle);
        }

        return handle;
    }




    /**
     * FFT along a particular dimension
     *
     * @param transform   the ndarray to op
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    @Override
    public IComplexNDArray fft(INDArray transform, int numElements, int dimension) {
        IComplexNDArray inputC = Nd4j.createComplex(transform);
        if (inputC.isVector())
            return (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorFFT(inputC,numElements));
        else {
            int[] finalShape = ArrayUtil.replace(transform.shape(), dimension, numElements);
            IComplexNDArray transform2 = Nd4j.createComplex(transform);
            IComplexNDArray result = transform2.dup();

            int desiredElementsAlongDimension = result.size(dimension);

            if(numElements > desiredElementsAlongDimension) {
                result = ComplexNDArrayUtil.padWithZeros(result, finalShape);
            }

            else if(numElements < desiredElementsAlongDimension)
                result = ComplexNDArrayUtil.truncate(result,numElements,dimension);

            return rawfft(result, numElements, dimension);
        }
    }

    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    @Override
    public IComplexNDArray fft(IComplexNDArray inputC, int numElements, int dimension) {
        if (inputC.isVector()) {
            return doFft(inputC,numElements,JCufft.CUFFT_FORWARD);

        }
        else
            return rawfft(inputC, numElements, dimension);

    }



    private IComplexNDArray doFft(IComplexNDArray inputC,int n,int fftKind) {
        INDArray workerArea = Nd4j.create(inputC.length() * 2);
        CublasPointer inputPointer = new CublasPointer(inputC);
        CublasPointer workerPointer = new CublasPointer(workerArea);

        JCufft.cufftSetWorkArea(
                getHandle()
                , workerPointer.getDevicePointer());

        JCufft.cufftSetStream(
                getHandle()
                , ContextHolder.getInstance().getCudaStream());

        JCufft.cufftSetAutoAllocation(
                getHandle()
                , JCufft.cufftSetAutoAllocation(getHandle(), 1));


        JCufft.cufftPlan1d(
                getHandle()
                , n
                , cufftType.CUFFT_C2C, 1);


        JCufft.cufftExecC2C(
                getHandle()
                , inputPointer.getDevicePointer()
                , inputPointer.getDevicePointer()
                , fftKind);

        try {
            inputPointer.copyToHost();
            inputPointer.close();
            workerPointer.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        return inputC;


    }

    /**
     * IFFT along a particular dimension
     *
     * @param transform   the ndarray to op
     * @param numElements the desired number of elements in each fft
     * @param dimension   the dimension to do fft along
     * @return the iffted output
     */
    @Override
    public IComplexNDArray ifft(INDArray transform, int numElements, int dimension) {
        IComplexNDArray inputC = Nd4j.createComplex(transform);
        if (inputC.isVector())
            return (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorIFFT(inputC,numElements));
        else
            return rawifft(inputC, numElements, dimension);

    }

    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    @Override
    public IComplexNDArray ifft(IComplexNDArray inputC, int numElements, int dimension) {
        if (inputC.isVector())
            return (IComplexNDArray) Nd4j.getExecutioner().execAndReturn(new VectorIFFT(inputC,numElements));
        else {
            return rawifft(inputC, numElements, dimension);
        }
    }

    /**
     * FFT along a particular dimension
     *
     * @param transform   the ndarray to op
     * @param numElements the desired number of elements in each fft
     * @return the ffted output
     */
    @Override
    public IComplexNDArray ifft(INDArray transform, int numElements) {
        IComplexNDArray inputC = Nd4j.createComplex(transform);
        if (inputC.isVector())
            return doFft(inputC,numElements,JCufft.CUFFT_INVERSE);

        else {
            return rawifft(inputC, numElements, inputC.shape().length - 1);
        }
    }

    /**
     * 1d discrete fourier op, note that this will
     * throw an exception if the passed in input
     * isn't a vector.
     * See matlab's fft2 for more information
     *
     * @param inputC the input to op
     * @return the the discrete fourier op of the passed in input
     */
    @Override
    public IComplexNDArray ifft(IComplexNDArray inputC) {
        if (inputC.isVector())
            return doFft(inputC,inputC.length(),JCufft.CUFFT_FORWARD);

        else {
            return rawifft(inputC, inputC.size(inputC.shape().length - 1), inputC.shape().length - 1);
        }
    }


    /**
     * Underlying fft algorithm
     *
     * @param transform the ndarray to op
     * @param n         the desired number of elements
     * @param dimension the dimension to do fft along
     * @return the transformed ndarray
     */
    @Override
    public IComplexNDArray rawfft(IComplexNDArray transform, int n, int dimension) {
        IComplexNDArray result = transform.dup();

        if (transform.size(dimension) != n) {
            int[] shape = ArrayUtil.copy(result.shape());
            shape[dimension] = n;
            if (transform.size(dimension) > n)
                result = ComplexNDArrayUtil.truncate(result, n, dimension);

            else
                result = ComplexNDArrayUtil.padWithZeros(result, shape);

        }


        if (dimension != result.shape().length - 1)
            result = result.swapAxes(result.shape().length - 1, dimension);

        INDArray workerArea = Nd4j.create(transform.length() * 2);

        CublasPointer workerPointer = new CublasPointer(workerArea);

        JCufft.cufftSetWorkArea(getHandle(),workerPointer.getDevicePointer());

        JCufft.cufftSetStream(
                getHandle()
                , ContextHolder.getInstance().getCudaStream());

        JCufft.cufftSetAutoAllocation(
                getHandle()
                , JCufft.cufftSetAutoAllocation(getHandle(), 1));

        CublasPointer pointer = new CublasPointer(transform);


        long[] workAreaLength = new long[]{workerArea.length()  * workerArea.data().getElementSize()};
        if(transform.data().dataType() == DataBuffer.Type.FLOAT) {
            JCufft.cufftMakePlanMany(
                    getHandle()
                    , transform.slices()
                    , transform.shape()
                    , transform.shape()
                    , transform.majorStride(), transform.stride(-1), transform.shape()
                    , transform.majorStride()
                    , transform.stride(-1)
                    , cufftType.CUFFT_C2C
                    , transform.vectorsAlongDimension(0)
                    , workAreaLength);

            JCufft.cufftExecC2C(
                    getHandle()
                    ,pointer.getDevicePointer()
                    ,pointer.getDevicePointer()
                    ,JCufft.CUFFT_FORWARD);

        }
        else {
            JCufft.cufftMakePlanMany(
                    getHandle()
                    , transform.slices()
                    , transform.shape()
                    , transform.shape()
                    , transform.majorStride(), transform.stride(-1), transform.shape()
                    , transform.majorStride()
                    , transform.stride(-1)
                    , cufftType.CUFFT_C2C
                    , transform.vectorsAlongDimension(0)
                    , workAreaLength);
            JCufft.cufftExecC2C(

                    getHandle()
                    ,pointer.getDevicePointer()
                    ,pointer.getDevicePointer()
                    ,JCufft.CUFFT_FORWARD);

        }


        try {
            pointer.copyToHost();
            pointer.close();
            workerPointer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (dimension != result.shape().length - 1)
            result = result.swapAxes(result.shape().length - 1, dimension);

        return result;
    }


    //underlying fftn
    @Override
    public IComplexNDArray rawifft(IComplexNDArray transform, int n, int dimension) {
        IComplexNDArray result = transform.dup();

        if (transform.size(dimension) != n) {
            int[] shape = ArrayUtil.copy(result.shape());
            shape[dimension] = n;
            if (transform.size(dimension) > n)
                result = ComplexNDArrayUtil.truncate(result, n, dimension);

            else
                result = ComplexNDArrayUtil.padWithZeros(result, shape);

        }


        if (dimension != result.shape().length - 1)
            result = result.swapAxes(result.shape().length - 1, dimension);

        INDArray workerArea = Nd4j.create(transform.length() * 2);
        CublasPointer workerPointer = new CublasPointer(workerArea);
        JCufft.cufftSetWorkArea(getHandle(),workerPointer.getDevicePointer());
        JCufft.cufftSetStream(
                getHandle()
                , ContextHolder.getInstance().getCudaStream());
        JCufft.cufftSetAutoAllocation(getHandle(), JCufft.cufftSetAutoAllocation(getHandle(), 1));
        CublasPointer pointer = new CublasPointer(transform);
        long[] workAreaLength = new long[]{workerArea.length()  * workerArea.data().getElementSize()};
        if(transform.data().dataType() == DataBuffer.Type.FLOAT) {
            JCufft.cufftMakePlanMany(
                    getHandle()
                    , transform.slices()
                    , transform.shape()
                    , null
                    , transform.majorStride(), transform.stride(-1), transform.shape()
                    , transform.majorStride()
                    , transform.stride(-1)
                    , cufftType.CUFFT_C2C
                    , transform.vectorsAlongDimension(0)
                    , workAreaLength);
            JCufft.cufftExecC2C(
                    getHandle()
                    ,pointer.getDevicePointer()
                    ,pointer.getDevicePointer()
                    ,JCufft.CUFFT_INVERSE);

        }
        else {
            JCufft.cufftMakePlanMany(
                    getHandle()
                    , transform.slices()
                    , transform.shape()
                    , null
                    , transform.majorStride(), transform.stride(-1), transform.shape()
                    , transform.majorStride()
                    , transform.stride(-1)
                    , cufftType.CUFFT_C2C
                    , transform.vectorsAlongDimension(0)
                    , workAreaLength);
            JCufft.cufftExecC2C(
                    getHandle()
                    ,pointer.getDevicePointer()
                    ,pointer.getDevicePointer()
                    ,JCufft.CUFFT_INVERSE);

        }


        try {
            pointer.copyToHost();
            pointer.close();
            workerPointer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (dimension != result.shape().length - 1)
            result = result.swapAxes(result.shape().length - 1, dimension);

        return result;
    }


}
