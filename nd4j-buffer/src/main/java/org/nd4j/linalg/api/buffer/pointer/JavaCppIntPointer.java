package org.nd4j.linalg.api.buffer.pointer;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.buffer.util.LibUtils;


/**
 * Created by agibsonccc on 2/25/16.
 */
@Platform(include="NativeBuffer.h",link = "buffer")
public class JavaCppIntPointer  extends IntPointer {


    static {
        try {
            LibUtils.addLibraryPath(System.getProperty("java.io.tmpdir"));
            LibUtils.loadTempBinaryFile("buffer");
            LibUtils.loadTempBinaryFile("jniJavaCppIntPointer");
            Loader.load();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public JavaCppIntPointer(int... array) {
        super((Pointer) null);
        allocateArray(array.length);
        put(array);
    }

    public JavaCppIntPointer(int size) {
        super((Pointer)null);
        allocateArray(size);
    }



    public native void putInt(int i,int val);

    private native void allocate();

    private native void allocateArray(int size);

    public native int[] bufferRef();

    public native void create(int length);

    public native long bufferAddress();
}
