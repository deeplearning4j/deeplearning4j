package org.nd4j.linalg.jcublas.fft;

import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.DefaultFFTInstance;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.util.ArrayUtil;

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
        return doFft(Nd4j.createComplex(transform),numElements,dimension,JCufft.CUFFT_FORWARD);
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
        return doFft(inputC,numElements,dimension,JCufft.CUFFT_FORWARD);

    }


    private IComplexNDArray doFftn(IComplexNDArray transform,int[] shape,int[] axes,int fftType) {
        IComplexNDArray result = transform.dup();
        if(axes == null)
            axes = ArrayUtil.range(0,transform.shape().length);

        INDArray workerArea = Nd4j.create(result.length() * 2);

        CublasPointer workerPointer = new CublasPointer(workerArea);

        JCufft.cufftSetWorkArea(getHandle(),workerPointer.getDevicePointer());

        JCufft.cufftSetStream(
                getHandle()
                , ContextHolder.getInstance().getCudaStream());

        JCufft.cufftSetAutoAllocation(
                getHandle()
                , JCufft.cufftSetAutoAllocation(getHandle(), 1));

        CublasPointer pointer = new CublasPointer(result);


        long[] workAreaLength = new long[]{workerArea.length()  * workerArea.data().getElementSize()};
        if(transform.isVector()) {
            JCufft.cufftMakePlan1d(getHandle(),transform.length(),fftType,1,workAreaLength);
        }
        else if(transform.isMatrix()) {
            JCufft.cufftMakePlan2d(getHandle(),transform.rows(),transform.columns(),fftType,workAreaLength);
        }
        else if(transform.shape().length == 3) {
            JCufft.cufftMakePlan3d(getHandle(),transform.size(0),transform.size(1),transform.size(2),fftType,workAreaLength);

        }
        else {
            JCufft.cufftMakePlanMany(
                    getHandle()
                    , axes.length
                    , transform.shape()
                    , shape
                    , transform.elementStride()
                    , transform.stride(0)
                    , shape
                    , transform.elementStride()
                    , transform.stride(0)
                    , cufftType.CUFFT_C2C
                    , 1
                    , workAreaLength);


        }

        JCufft.cufftExecC2C(
                getHandle()
                , pointer.getDevicePointer()
                , pointer.getDevicePointer()
                , fftType);

        try {
            pointer.copyToHost();
            pointer.close();
            workerPointer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }



        return result;

    }

    private IComplexNDArray doFft(IComplexNDArray inputC,int n,int dimension,int fftKind) {
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
        return doFft(inputC,numElements,dimension,JCufft.CUFFT_INVERSE);

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
        return doFft(inputC,numElements,dimension,JCufft.CUFFT_INVERSE);
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
        return doFft(inputC,numElements,- 1,JCufft.CUFFT_INVERSE);
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
        return doFft(inputC,inputC.length(),-1,JCufft.CUFFT_INVERSE);

    }

    @Override
    public IComplexNDArray rawifftn(IComplexNDArray transform, int[] shape, int[] axes) {
        return doFftn(transform,shape,axes,JCufft.CUFFT_INVERSE);
    }




    @Override
    public IComplexNDArray rawfftn(IComplexNDArray transform, int[] shape, int[] axes) {
        return doFftn(transform,shape,axes,JCufft.CUFFT_FORWARD);
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
        return doFft(transform,n,dimension,JCufft.CUFFT_FORWARD);
    }


    //underlying fftn
    @Override
    public IComplexNDArray rawifft(IComplexNDArray transform, int n, int dimension) {
        return doFft(transform,n,dimension,JCufft.CUFFT_INVERSE);
    }


}
