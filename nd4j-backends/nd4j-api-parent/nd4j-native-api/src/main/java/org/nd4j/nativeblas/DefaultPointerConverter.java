package org.nd4j.nativeblas;

import org.nd4j.linalg.api.buffer.unsafe.UnsafeHolder;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.Buffer;

/**
 * Created by agibsonccc on 2/20/16.
 */
public class DefaultPointerConverter implements PointerConverter {
    @Override
    public long toPointer(IComplexNDArray arr) {
        return arr.data().address();
    }

    @Override
    public long toPointer(INDArray arr) {
        return arr.data().address();
    }

    @Override
    public long toPointer(Buffer buffer) {
        if(buffer.isDirect())
            try {
                return  UnsafeHolder.getUnsafe().objectFieldOffset(UnsafeHolder.getAddressField());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        else {
            try {
                //http://stackoverflow.com/questions/8820164/is-there-a-way-to-get-a-reference-address
                int offset = UnsafeHolder.getUnsafe().arrayBaseOffset(buffer.array().getClass());
                int scale = UnsafeHolder.getUnsafe().arrayIndexScale(buffer.array().getClass());
                switch (scale) {
                    case 4:
                        long factor = UnsafeHolder.is64Bit() ? 8 : 1;
                        final long i1 = (UnsafeHolder.getUnsafe().getInt(buffer.array(), offset) & 0xFFFFFFFFL) * factor;
                        return i1;
                    case 8:
                        throw new AssertionError("Not supported");
                }
            }catch(Exception e) {
                throw new IllegalStateException("Unable to get address");
            }
        }

        throw new IllegalStateException("Unable to get address");
    }
}
