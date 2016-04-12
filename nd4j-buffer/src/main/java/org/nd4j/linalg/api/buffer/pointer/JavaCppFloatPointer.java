package org.nd4j.linalg.api.buffer.pointer;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.buffer.util.LibUtils;


/**
 * Java cpp float pointer
 *
 * @author Adam Gibson
 */
@Platform(include="NativeBuffer.h",link = "buffer")
public class JavaCppFloatPointer extends FloatPointer {

    static {
        try {
            LibUtils.addLibraryPath(System.getProperty("java.io.tmpdir"));
            LibUtils.loadTempBinaryFile("buffer");
            LibUtils.loadTempBinaryFile("jniJavaCppFloatPointer");
            Loader.load();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public JavaCppFloatPointer(float... array) {
        super((Pointer) null);
        allocateArray(array.length);
        put(array);

    }

    public JavaCppFloatPointer(int size) {
        super((Pointer) null);
        allocateArray(size);

    }

    public native void putFloat(int i,float val);




    private native void allocate();

    public native float[] bufferRef();

    public native void create(int length);

    private native void allocateArray(int size);


    public native long bufferAddress();

}
