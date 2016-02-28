package org.nd4j.linalg.api.buffer.pointer;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.buffer.util.LibUtils;


/**
 * Created by agibsonccc on 2/25/16.
 */
@Platform(include="NativeBuffer.h",link = "buffer")
public class JavaCppDoublePointer extends DoublePointer {

    static {
        try {
            LibUtils.addLibraryPath(System.getProperty("java.io.tmpdir"));
            LibUtils.loadTempBinaryFile("buffer");
            LibUtils.loadTempBinaryFile("jniJavaCppDoublePointer");
            Loader.load();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public JavaCppDoublePointer() {
    }

    public JavaCppDoublePointer(double... array) {
        super((Pointer) null);


    }

    public JavaCppDoublePointer(int size) {
        super((Pointer) null);
    }

    private native void allocateArray(int size);


    public native void putDouble(int i,double val);

    public native double[] bufferRef();

    public native void create(int length);

    public native long bufferAddress();

    private native void allocate();

}
