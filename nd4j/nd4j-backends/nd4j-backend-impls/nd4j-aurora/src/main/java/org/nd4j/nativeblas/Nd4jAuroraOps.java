package org.nd4j.nativeblas;

import java.io.File;
import java.lang.reflect.Field;
import java.util.Arrays;
import lombok.Getter;
import org.bytedeco.javacpp.BooleanPointer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.nd4j.nativeblas.Nd4jAurora.*;

/**
 *
 * @author saudet
 */
public class Nd4jAuroraOps implements NativeOps {
    private static Logger log = LoggerFactory.getLogger(Nd4jAuroraOps.class);
    public static final boolean LOAD_SHARED_LIBRARY = true;

    /* Load "Nd4jAuroraOps" on VE node 0 */
    @Getter int deviceId = 0;
    @Getter String veobin = null;
    @Getter veo_proc_handle proc = null;
    @Getter long handle = 0;/* find a function in the executable */
    @Getter veo_thr_ctxt ctx = null;
    Field pointerArrayField;

    public Nd4jAuroraOps() {
        try {
            String s = System.getenv("VE_NODE_NUMBER");
            if (s != null) {
                deviceId = Integer.parseInt(s);
            }
            File f = Loader.cacheResource(Loader.getPlatform() + (LOAD_SHARED_LIBRARY ? "/libnd4jaurora.so" : "/nd4jaurora"));
            f.setExecutable(true);
            veobin = f.getAbsolutePath();
            setDevice(deviceId);
            pointerArrayField = PointerPointer.class.getDeclaredField("pointerArray");
            pointerArrayField.setAccessible(true);
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    public int callInt(String symname, Object... args) {
        int i = (int)(long)call(symname, args);
        log.debug("return int " + i);
        return i;
    }

    public long callLong(String symname, Object... args) {
        long l = (long)call(symname, args);
        log.debug("return long " + l);
        return l;
    }

    public float callFloat(String symname, Object... args) {
        float f = Float.intBitsToFloat((int)((long)call(symname, args) >> 32));
        log.debug("return float " + f);
        return f;
    }

    public double callDouble(String symname, Object... args) {
        double d = Double.longBitsToDouble((long)call(symname, args));
        log.debug("return double " + d);
        return d;
    }

    public boolean callBoolean(String symname, Object... args) {
        boolean b = (long)call(symname, args) != 0;
        log.debug("return boolean " + b);
        return b;
    }

    public char callChar(String symname, Object... args) {
        char c = (char)(long)call(symname, args);
        log.debug("return char " + c);
        return c;
    }

    public Pointer callPointer(String symname, Object... args) {
        Pointer p = new PagedPointer((long)call(symname, args));
        if (p.isNull()) {
            p = null;
        }
        log.debug("return pointer " + p);
        return p;
    }

    public synchronized String callString(String symname, Object... args) {
        BytePointer dst = new BytePointer(64 * 1024);
        long src = callLong(symname, args);
        int error = veo_read_mem(proc, dst, src, 1);
        if (error != 0) {
            throw new RuntimeException("veo_read_mem(): error " + error);
        }
        for (int i = 0; i < dst.limit(); i++) {
            error = veo_read_mem(proc, dst.position(i), src + i, 1);
            if (error != 0) {
                throw new RuntimeException("veo_read_mem(): error " + error);
            }
            if (dst.get() == 0) {
                dst.position(0).limit(i);
                break;
            }
        }
        String s = dst.position(0).getString();
        log.debug("return string \"" + s + "\"");
        return s;
    }

    public synchronized Object call(String symname, Object... args) {
        log.debug("call(" + symname + ", " + Arrays.deepToString(args) + ")");

        long sym = veo_get_sym(proc, handle, symname);
        if (sym == 0) {
            throw new RuntimeException("veo_get_sym(): failed to find symbol");
        }
        veo_args argp = veo_args_alloc();
        if (argp == null) {
            throw new RuntimeException("veo_args_alloc(): allocation of veo_args failed");
        }
        long[] pointers = new long[args.length];
        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];
            if (arg == null) {
                int error = veo_args_set_i32(argp, i, 0);
                if (error != 0) {
                    throw new RuntimeException("veo_args_set_i32(): error " + error);
                }
            } else if (arg instanceof Integer) {
                int error = veo_args_set_i32(argp, i, (Integer)arg);
                if (error != 0) {
                    throw new RuntimeException("veo_args_set_i32(): error " + error);
                }
            } else if (arg instanceof Long) {
                int error = veo_args_set_i64(argp, i, (Long)arg);
                if (error != 0) {
                    throw new RuntimeException("veo_args_set_i64(): error " + error);
                }
            } else if (arg instanceof Float) {
                int error = veo_args_set_float(argp, i, (Float)arg);
                if (error != 0) {
                    throw new RuntimeException("veo_args_set_float(): error " + error);
                }
            } else if (arg instanceof Double) {
                int error = veo_args_set_double(argp, i, (Double)arg);
                if (error != 0) {
                    throw new RuntimeException("veo_args_set_double(): error " + error);
                }
            } else if (arg instanceof Boolean) {
                int error = veo_args_set_i32(argp, i, (Boolean)arg ? 1 : 0);
                if (error != 0) {
                    throw new RuntimeException("veo_args_set_i32(): error " + error);
                }
            } else if (arg instanceof Character) {
                int error = veo_args_set_i32(argp, i, (Character)arg);
                if (error != 0) {
                    throw new RuntimeException("veo_args_set_i32(): error " + error);
                }
            } else if (arg instanceof Pointer) {
                Pointer p = (Pointer)arg;
                if (p.limit() <= 0) {
                    int error = veo_args_set_i64(argp, i, p.address() + p.position() * p.sizeof());
                    if (error != 0) {
                        throw new RuntimeException("veo_args_set_i64(): error " + error);
                    }
                } else {
                    long size = (p.limit() - p.position()) * p.sizeof();
                    long[] addr = {0};
                    int error = veo_alloc_mem(proc, addr, size);
                    if (error != 0) {
                        throw new RuntimeException("veo_alloc_mem(): error " + error);
                    }
                    error = veo_args_set_i64(argp, i, pointers[i] = addr[0]);
                    if (error != 0) {
                        throw new RuntimeException("veo_args_set_i64(): error " + error);
                    }
                    try {
                        Pointer[] array;
                        if (p instanceof PointerPointer && (array = (Pointer[])pointerArrayField.get((PointerPointer)p)) != null) {
                            for (int j = 0; j < array.length; j++) {
                                Pointer p2 = array[j];
                                LongPointer addr2 = new LongPointer(1);
                                if (p2 == null || p2.limit() <= 0) {
                                    error = veo_write_mem(proc, pointers[i] + j * 8, addr2.put(p2 == null ? 0 : p2.address()), 8);
                                    if (error != 0) {
                                        throw new RuntimeException("veo_write_mem(): error " + error);
                                    }
                                } else {
                                    long size2 = (p2.limit() - p2.position()) * p2.sizeof();
                                    error = veo_alloc_mem(proc, addr2, size2);
                                    if (error != 0) {
                                        throw new RuntimeException("veo_alloc_mem(): error " + error);
                                    }
                                    error = veo_write_mem(proc, addr2.get(0), p2, size2);
                                    if (error != 0) {
                                        throw new RuntimeException("veo_write_mem(): error " + error);
                                    }
                                    error = veo_write_mem(proc, pointers[i] + j * 8, addr2, 8);
                                    if (error != 0) {
                                        throw new RuntimeException("veo_write_mem(): error " + error);
                                    }
                                }
                            }
                        } else {
                            error = veo_write_mem(proc, pointers[i], p, size);
                            if (error != 0) {
                                throw new RuntimeException("veo_write_mem(): error " + error);
                            }
                        }
                    } catch (Exception ex) {
                        throw new RuntimeException(ex);
                    }
                }
            } else {
                throw new UnsupportedOperationException("Not supported yet: " + arg);
            }
        }
        long id = veo_call_async(ctx, sym, argp);
        if (id == VEO_REQUEST_ID_INVALID) {
            throw new RuntimeException("veo_call_async(): request failed");
        }
        long[] retval = {0};
        int error = veo_call_wait_result(ctx, id, retval);
        if (error != 0) {
            throw new RuntimeException("veo_call_wait_result(): error " + error);
        }
        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];
            if (pointers[i] != 0) {
                Pointer p = (Pointer)arg;
                long size = (p.limit() - p.position()) * p.sizeof();
                try {
                    Pointer[] array;
                    if (p instanceof PointerPointer && (array = (Pointer[])pointerArrayField.get((PointerPointer)p)) != null) {
                        for (int j = 0; j < array.length; j++) {
                            Pointer p2 = array[j];
                            LongPointer addr2 = new LongPointer(1);
                            if (p2 == null || p2.limit() <= 0) {
                                error = veo_read_mem(proc, addr2, pointers[i] + j * 8, 8);
                                ((PointerPointer)p).put(j, array[j] = new PagedPointer(addr2.get()));
                                if (error != 0) {
                                    throw new RuntimeException("veo_read_mem(): error " + error);
                                }
                            } else {
                                long size2 = (p2.limit() - p2.position()) * p2.sizeof();
                                error = veo_read_mem(proc, addr2, pointers[i] + j * 8, 8);
                                if (error != 0) {
                                    throw new RuntimeException("veo_read_mem(): error " + error);
                                }
                                error = veo_read_mem(proc, p2, addr2.get(0), size2);
                                if (error != 0) {
                                    throw new RuntimeException("veo_read_mem(): error " + error);
                                }
                                error = veo_free_mem(proc, addr2.get(0));
                                if (error != 0) {
                                    throw new RuntimeException("veo_free_mem(): error " + error);
                                }
                            }
                        }
                    } else {
                        error = veo_read_mem(proc, p, pointers[i], size);
                        if (error != 0) {
                            throw new RuntimeException("veo_read_mem(): error " + error);
                        }
                    }
                } catch (Exception ex) {
                    throw new RuntimeException(ex);
                }
                error = veo_free_mem(proc, pointers[i]);
                if (error != 0) {
                    throw new RuntimeException("veo_free_mem(): error " + error);
                }
            }
        }
        veo_args_free(argp);

        log.debug("return " + retval[0]);
        return retval[0];
    }

    @Override
    public synchronized int setDevice(int deviceId) {
        this.deviceId = deviceId;
        if (ctx != null) {
            veo_context_close(ctx);
        }
        if (proc != null) {
            veo_proc_destroy(proc);
        }
        if (LOAD_SHARED_LIBRARY) {
            proc = veo_proc_create(deviceId);
            handle = veo_load_library(proc, veobin);
        } else {
            proc = veo_proc_create_static(deviceId, veobin);
            handle = 0;
        }
        ctx = veo_context_open(proc);
        if (proc == null || ctx == null) {
            throw new RuntimeException("setDevice() failed");
        }
        return 1; // ??
    }

    @Override
    public int getDevice() {
        return deviceId;
    }

    @Override
    public synchronized Pointer mallocDevice(long memorySize, int deviceId, int flags) {
        log.debug("mallocDevice(" + memorySize + ")");
        long[] addr = {0};
        int error = veo_alloc_mem(proc, addr, memorySize);
        if (error != 0) {
            throw new RuntimeException("veo_alloc_mem(): error " + error);
        }
        Pointer p = new PagedPointer(addr[0]);
        log.debug("return " + p);
        return p;
    }

    @Override
    public synchronized int freeDevice(Pointer p, int deviceId) {
        log.debug("freeDevice(" + p + ")");
        int i = veo_free_mem(proc, p.address());
        if (i != 0) {
            throw new RuntimeException("veo_free_mem(): error " + i);
        }
        log.debug("return " + i);
        return i;
    }

    /**
     *
     * @param dst
     * @param src
     * @param size
     * @param flags 0 == guess, 1 == host->device, 2 == device->host, 3 == device->device
     * @param reserved
     * @return
     */
    @Override
    public synchronized int memcpySync(Pointer dst, Pointer src, long size, int flags, Pointer reserved) {
        if (log.isDebugEnabled()) {
            log.debug("memcpySync(" + dst + ", " + src + ", " + size + ", " + flags + ")");
        }
        int i = -1;
        if (flags == 3 || (flags == 0 && src.limit() <= 0 && dst.limit() <= 0)) {
            i = callInt("memcpySync", dst, src, size, flags, reserved);
        } else if (flags == 2 || (flags == 0 && dst.limit() > 0)) {
            // dst is host, src is device
            i = veo_read_mem(proc, dst, src.address() + src.position(), size);
            if (i != 0) {
                throw new RuntimeException("veo_read_mem(): error " + i);
            }
        } else if (flags == 1 || (flags == 0 && src.limit() > 0)) {
            // dst is device, src is host
            i = veo_write_mem(proc, dst.address() + dst.position(), src, size);
            if (i != 0) {
                throw new RuntimeException("veo_write_mem(): error " + i);
            }
        }
        if (log.isDebugEnabled()) {
            log.debug("return " + i);
        }
        return i;
    }

    @Override
    public void setElementThreshold(int arg0) {
        call("setElementThreshold", arg0);
    }

    @Override
    public void setTADThreshold(int arg0) {
        call("setTADThreshold", arg0);
    }

    @Override
    public void execIndexReduceScalar(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8) {
        call("execIndexReduceScalar", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execIndexReduce(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11) {
        call("execIndexReduce", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execBroadcast(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, OpaqueDataBuffer arg8, LongPointer arg9, LongPointer arg10, OpaqueDataBuffer arg11, LongPointer arg12, LongPointer arg13) {
        call("execBroadcast", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13);
    }

    @Override
    public void execBroadcastBool(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, OpaqueDataBuffer arg8, LongPointer arg9, LongPointer arg10, Pointer arg11, OpaqueDataBuffer arg12, LongPointer arg13, LongPointer arg14) {
        call("execBroadcastBool", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14);
    }

    @Override
    public void execPairwiseTransform(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, OpaqueDataBuffer arg8, LongPointer arg9, LongPointer arg10, Pointer arg11) {
        call("execPairwiseTransform", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execPairwiseTransformBool(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, OpaqueDataBuffer arg8, LongPointer arg9, LongPointer arg10, Pointer arg11) {
        call("execPairwiseTransformBool", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execReduceFloat(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8) {
        call("execReduceFloat", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execReduceSame(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8) {
        call("execReduceSame", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execReduceBool(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8) {
        call("execReduceBool", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execReduceLong(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8) {
        call("execReduceLong", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execReduceFloat2(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11) {
        call("execReduceFloat2", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execReduceSame2(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11) {
        call("execReduceSame2", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execReduceBool2(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11) {
        call("execReduceBool2", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execReduceLong2(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11) {
        call("execReduceLong2", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execReduce3(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11) {
        call("execReduce3", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execReduce3Scalar(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11) {
        call("execReduce3Scalar", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execReduce3Tad(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11, OpaqueDataBuffer arg12, LongPointer arg13, LongPointer arg14, LongPointer arg15, LongPointer arg16, LongPointer arg17, LongPointer arg18) {
        call("execReduce3Tad", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18);
    }

    @Override
    public void execReduce3All(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11, OpaqueDataBuffer arg12, LongPointer arg13, LongPointer arg14, LongPointer arg15, LongPointer arg16, LongPointer arg17, LongPointer arg18) {
        call("execReduce3All", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18);
    }

    @Override
    public void execScalar(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, OpaqueDataBuffer arg8, LongPointer arg9, LongPointer arg10, Pointer arg11) {
        call("execScalar", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execScalarBool(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, OpaqueDataBuffer arg8, LongPointer arg9, LongPointer arg10, Pointer arg11) {
        call("execScalarBool", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void execSummaryStatsScalar(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, boolean arg9) {
        call("execSummaryStatsScalar", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    @Override
    public void execSummaryStats(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, boolean arg9) {
        call("execSummaryStats", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    @Override
    public void execSummaryStatsTad(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, Pointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11, boolean arg12, LongPointer arg13, LongPointer arg14) {
        call("execSummaryStatsTad", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14);
    }

    @Override
    public void execTransformFloat(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, Pointer arg8) {
        call("execTransformFloat", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execTransformSame(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, Pointer arg8) {
        call("execTransformSame", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execTransformStrict(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, Pointer arg8) {
        call("execTransformStrict", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execTransformBool(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, Pointer arg8) {
        call("execTransformBool", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execTransformAny(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, Pointer arg8) {
        call("execTransformAny", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public void execScalarTad(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, OpaqueDataBuffer arg8, LongPointer arg9, LongPointer arg10, Pointer arg11, OpaqueDataBuffer arg12, LongPointer arg13, LongPointer arg14, LongPointer arg15, LongPointer arg16, LongPointer arg17, LongPointer arg18) {
        call("execScalarTad", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18);
    }

    @Override
    public void execScalarBoolTad(PointerPointer arg0, int arg1, OpaqueDataBuffer arg2, LongPointer arg3, LongPointer arg4, OpaqueDataBuffer arg5, LongPointer arg6, LongPointer arg7, OpaqueDataBuffer arg8, LongPointer arg9, LongPointer arg10, Pointer arg11, OpaqueDataBuffer arg12, LongPointer arg13, LongPointer arg14, LongPointer arg15, LongPointer arg16, LongPointer arg17, LongPointer arg18) {
        call("execScalarBoolTad", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18);
    }

    @Override
    public void specialConcat(PointerPointer arg0, int arg1, int arg2, PointerPointer arg3, PointerPointer arg4, Pointer arg5, LongPointer arg6, PointerPointer arg7, PointerPointer arg8) {
        call("specialConcat", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    @Override
    public int ompGetMaxThreads() {
        return callInt("ompGetMaxThreads");
    }

    @Override
    public int ompGetNumThreads() {
        return callInt("ompGetNumThreads");
    }

    @Override
    public void setOmpNumThreads(int arg0) {
        call("setOmpNumThreads", arg0);
    }

    @Override
    public void setOmpMinThreads(int arg0) {
        call("setOmpMinThreads", arg0);
    }

    @Override
    public void initializeDevicesAndFunctions() {
        call("initializeDevicesAndFunctions");
    }

    @Override
    public void initializeFunctions(PointerPointer arg0) {
        call("initializeFunctions", arg0);
    }

    @Override
    public Pointer mallocHost(long arg0, int arg1) {
        return callPointer("mallocHost", arg0, arg1);
    }

    @Override
    public int freeHost(Pointer arg0) {
        return callInt("freeHost", arg0);
    }

    @Override
    public Pointer createContext() {
        return callPointer("createContext");
    }

    @Override
    public Pointer createStream() {
        return callPointer("createStream");
    }

    @Override
    public Pointer createEvent() {
        return callPointer("createEvent");
    }

    @Override
    public int registerEvent(Pointer arg0, Pointer arg1) {
        return callInt("registerEvent", arg0, arg1);
    }

    @Override
    public int destroyEvent(Pointer arg0) {
        return callInt("destroyEvent", arg0);
    }

    @Override
    public int streamSynchronize(Pointer arg0) {
        return callInt("streamSynchronize", arg0);
    }

    @Override
    public int eventSynchronize(Pointer arg0) {
        return callInt("eventSynchronize", arg0);
    }

    @Override
    public long getDeviceFreeMemory(int arg0) {
        return callLong("getDeviceFreeMemory", arg0);
    }

    @Override
    public long getDeviceFreeMemoryDefault() {
        return callLong("getDeviceFreeMemoryDefault");
    }

    @Override
    public long getDeviceTotalMemory(int arg0) {
        return callLong("getDeviceTotalMemory", arg0);
    }

    @Override
    public int getDeviceMajor(int arg0) {
        return callInt("getDeviceMajor", arg0);
    }

    @Override
    public int getDeviceMinor(int arg0) {
        return callInt("getDeviceMinor", arg0);
    }

    @Override
    public String getDeviceName(int arg0) {
        return callString("getDeviceName", arg0);
    }

    @Override
    public int memcpyAsync(Pointer arg0, Pointer arg1, long arg2, int arg3, Pointer arg4) {
        return callInt("memcpyAsync", arg0, arg1, arg2, arg3, arg4);
    }

    @Override
    public int memcpyConstantAsync(long arg0, Pointer arg1, long arg2, int arg3, Pointer arg4) {
        return callInt("memcpyConstantAsync", arg0, arg1, arg2, arg3, arg4);
    }

    @Override
    public int memsetSync(Pointer arg0, int arg1, long arg2, int arg3, Pointer arg4) {
        return callInt("memsetSync", arg0, arg1, arg2, arg3, arg4);
    }

    @Override
    public int memsetAsync(Pointer arg0, int arg1, long arg2, int arg3, Pointer arg4) {
        return callInt("memsetAsync", arg0, arg1, arg2, arg3, arg4);
    }

    @Override
    public Pointer getConstantSpace() {
        return callPointer("getConstantSpace");
    }

    @Override
    public int getAvailableDevices() {
        return callInt("getAvailableDevices");
    }

    @Override
    public void enableDebugMode(boolean arg0) {
        call("enableDebugMode", arg0);
    }

    @Override
    public void enableVerboseMode(boolean arg0) {
        call("enableVerboseMode", arg0);
    }

    @Override
    public void setGridLimit(int arg0) {
        call("setGridLimit", arg0);
    }

    @Override
    public OpaqueTadPack tadOnlyShapeInfo(LongPointer arg0, IntPointer arg1, int arg2) {
        return new OpaqueTadPack(callPointer("tadOnlyShapeInfo", arg0, arg1, arg2));
    }

    @Override
    public LongPointer getPrimaryShapeInfo(OpaqueTadPack arg0) {
        return new LongPointer(callPointer("getPrimaryShapeInfo", arg0));
    }

    @Override
    public LongPointer getPrimaryOffsets(OpaqueTadPack arg0) {
        return new LongPointer(callPointer("getPrimaryOffsets", arg0));
    }

    @Override
    public LongPointer getSpecialShapeInfo(OpaqueTadPack arg0) {
        return new LongPointer(callPointer("getSpecialShapeInfo", arg0));
    }

    @Override
    public LongPointer getSpecialOffsets(OpaqueTadPack arg0) {
        return new LongPointer(callPointer("getSpecialOffsets", arg0));
    }

    @Override
    public long getNumberOfTads(OpaqueTadPack arg0) {
        return callLong("getNumberOfTads", arg0);
    }

    @Override
    public int getShapeInfoLength(OpaqueTadPack arg0) {
        return callInt("getShapeInfoLength", arg0);
    }

    @Override
    public void deleteTadPack(OpaqueTadPack arg0) {
        call("deleteTadPack", arg0);
    }

    @Override
    public void pullRows(PointerPointer arg0, OpaqueDataBuffer arg1, LongPointer arg2, LongPointer arg3, OpaqueDataBuffer arg4, LongPointer arg5, LongPointer arg6, long arg7, LongPointer arg8, LongPointer arg9, LongPointer arg10, LongPointer arg11, LongPointer arg12) {
        call("pullRows", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12);
    }

    @Override
    public void average(PointerPointer arg0, PointerPointer arg1, LongPointer arg2, PointerPointer arg3, LongPointer arg4, Pointer arg5, LongPointer arg6, Pointer arg7, LongPointer arg8, int arg9, long arg10, boolean arg11) {
        call("average", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
    }

    @Override
    public void accumulate(PointerPointer arg0, PointerPointer arg1, LongPointer arg2, PointerPointer arg3, LongPointer arg4, Pointer arg5, LongPointer arg6, Pointer arg7, LongPointer arg8, int arg9, long arg10) {
        call("accumulate", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    @Override
    public void enableP2P(boolean arg0) {
        call("enableP2P", arg0);
    }

    @Override
    public void checkP2P() {
        call("checkP2P");
    }

    @Override
    public boolean isP2PAvailable() {
        return callBoolean("isP2PAvailable");
    }

    @Override
    public void shuffle(PointerPointer arg0, PointerPointer arg1, PointerPointer arg2, PointerPointer arg3, PointerPointer arg4, PointerPointer arg5, PointerPointer arg6, PointerPointer arg7, PointerPointer arg8, int arg9, IntPointer arg10, PointerPointer arg11, PointerPointer arg12) {
        call("shuffle", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12);
    }

    @Override
    public void convertTypes(PointerPointer arg0, int arg1, Pointer arg2, long arg3, int arg4, Pointer arg5) {
        call("convertTypes", arg0, arg1, arg2, arg3, arg4, arg5);
    }

    @Override
    public boolean isExperimentalEnabled() {
        return callBoolean("isExperimentalEnabled");
    }

    @Override
    public void execAggregate(PointerPointer arg0, int arg1, PointerPointer arg2, int arg3, PointerPointer arg4, int arg5, IntPointer arg6, int arg7, PointerPointer arg8, int arg9, Pointer arg10, int arg11, int arg12) {
        call("execAggregate", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12);
    }

    @Override
    public void execAggregateBatch(PointerPointer arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8, Pointer arg9, int arg10) {
        call("execAggregateBatch", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    @Override
    public void execRandom(PointerPointer arg0, int arg1, Pointer arg2, OpaqueDataBuffer arg3, LongPointer arg4, LongPointer arg5, Pointer arg6) {
        call("execRandom", arg0, arg1, arg2, arg3, arg4, arg5, arg6);
    }

    @Override
    public void execRandom3(PointerPointer arg0, int arg1, Pointer arg2, OpaqueDataBuffer arg3, LongPointer arg4, LongPointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, OpaqueDataBuffer arg9, LongPointer arg10, LongPointer arg11, Pointer arg12) {
        call("execRandom3", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12);
    }

    @Override
    public void execRandom2(PointerPointer arg0, int arg1, Pointer arg2, OpaqueDataBuffer arg3, LongPointer arg4, LongPointer arg5, OpaqueDataBuffer arg6, LongPointer arg7, LongPointer arg8, Pointer arg9) {
        call("execRandom2", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    @Override
    public Pointer initRandom(PointerPointer arg0, long arg1, long arg2, Pointer arg3) {
        return callPointer("initRandom", arg0, arg1, arg2, arg3);
    }

    @Override
    public void refreshBuffer(PointerPointer arg0, long arg1, Pointer arg2) {
        call("refreshBuffer", arg0, arg1, arg2);
    }

    @Override
    public void reSeedBuffer(PointerPointer arg0, long arg1, Pointer arg2) {
        call("reSeedBuffer", arg0, arg1, arg2);
    }

    @Override
    public void destroyRandom(Pointer arg0) {
        call("destroyRandom", arg0);
    }

    @Override
    public Pointer numpyFromNd4j(Pointer arg0, Pointer arg1, long arg2) {
        return callPointer("numpyFromNd4j", arg0, arg1, arg2);
    }

    @Override
    public int elementSizeForNpyArrayHeader(Pointer arg0) {
        return callInt("elementSizeForNpyArrayHeader", arg0);
    }

    @Override
    public Pointer dataPointForNumpyStruct(Pointer arg0) {
        return callPointer("dataPointForNumpyStruct", arg0);
    }

    @Override
    public Pointer numpyHeaderForNd4j(Pointer arg0, Pointer arg1, long arg2, LongPointer arg3) {
        return callPointer("numpyHeaderForNd4j", arg0, arg1, arg2, arg3);
    }

    @Override
    public Pointer loadNpyFromHeader(Pointer arg0) {
        return callPointer("loadNpyFromHeader", arg0);
    }

    @Override
    public Pointer dataPointForNumpyHeader(Pointer arg0) {
        return callPointer("dataPointForNumpyHeader", arg0);
    }

    @Override
    public Pointer shapeBufferForNumpyHeader(Pointer arg0) {
        return callPointer("shapeBufferForNumpyHeader", arg0);
    }

    @Override
    public Pointer dataPointForNumpy(Pointer arg0) {
        return callPointer("dataPointForNumpy", arg0);
    }

    @Override
    public Pointer shapeBufferForNumpy(Pointer arg0) {
        return callPointer("shapeBufferForNumpy", arg0);
    }

    @Override
    public void releaseNumpy(Pointer arg0) {
        call("releaseNumpy", arg0);
    }

    @Override
    public Pointer numpyFromFile(BytePointer arg0) {
        return callPointer("numpyFromFile", arg0);
    }

    @Override
    public int lengthForShapeBufferPointer(Pointer arg0) {
        return callInt("lengthForShapeBufferPointer", arg0);
    }

    @Override
    public int elementSizeForNpyArray(Pointer arg0) {
        return callInt("elementSizeForNpyArray", arg0);
    }

    @Override
    public Pointer pointerForAddress(long arg0) {
        return callPointer("pointerForAddress", arg0);
    }

    @Override
    public Pointer mapFromNpzFile(BytePointer arg0) {
        return callPointer("mapFromNpzFile", arg0);
    }

    @Override
    public int getNumNpyArraysInMap(Pointer arg0) {
        return callInt("getNumNpyArraysInMap", arg0);
    }

    @Override
    public String getNpyArrayNameFromMap(Pointer arg0, int arg1) {
        return callString("getNpyArrayNameFromMap", arg0, arg1);
    }

    @Override
    public Pointer getNpyArrayFromMap(Pointer arg0, int arg1) {
        return callPointer("getNpyArrayFromMap", arg0, arg1);
    }

    @Override
    public Pointer getNpyArrayData(Pointer arg0) {
        return callPointer("getNpyArrayData", arg0);
    }

    @Override
    public LongPointer getNpyArrayShape(Pointer arg0) {
        return new LongPointer(callPointer("getNpyArrayShape", arg0));
    }

    @Override
    public int getNpyArrayRank(Pointer arg0) {
        return callInt("getNpyArrayRank", arg0);
    }

    @Override
    public char getNpyArrayOrder(Pointer arg0) {
        return callChar("getNpyArrayOrder", arg0);
    }

    @Override
    public int getNpyArrayElemSize(Pointer arg0) {
        return callInt("getNpyArrayElemSize", arg0);
    }

    @Override
    public void tear(PointerPointer arg0, OpaqueDataBuffer arg1, LongPointer arg2, LongPointer arg3, PointerPointer arg4, LongPointer arg5, LongPointer arg6, LongPointer arg7) {
        call("tear", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    @Override
    public void sort(PointerPointer arg0, Pointer arg1, LongPointer arg2, Pointer arg3, LongPointer arg4, boolean arg5) {
        call("sort", arg0, arg1, arg2, arg3, arg4, arg5);
    }

    @Override
    public void sortTad(PointerPointer arg0, Pointer arg1, LongPointer arg2, Pointer arg3, LongPointer arg4, IntPointer arg5, int arg6, LongPointer arg7, LongPointer arg8, boolean arg9) {
        call("sortTad", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    @Override
    public void sortCooIndices(PointerPointer arg0, LongPointer arg1, Pointer arg2, long arg3, int arg4) {
        call("sortCooIndices", arg0, arg1, arg2, arg3, arg4);
    }

    @Override
    public LongPointer mmapFile(PointerPointer arg0, String arg1, long arg2) {
        return new LongPointer(callPointer("mmapFile", arg0, arg1, arg2));
    }

    @Override
    public void munmapFile(PointerPointer arg0, LongPointer arg1, long arg2) {
        call("munmapFile", arg0, arg1, arg2);
    }

    @Override
    public OpaqueResultWrapper executeFlatGraph(PointerPointer arg0, Pointer arg1) {
        return new OpaqueResultWrapper(callPointer("executeFlatGraph", arg0, arg1));
    }

    @Override
    public long getResultWrapperSize(OpaqueResultWrapper arg0) {
        return callLong("getResultWrapperSize", arg0);
    }

    @Override
    public Pointer getResultWrapperPointer(OpaqueResultWrapper arg0) {
        return callPointer("getResultWrapperPointer", arg0);
    }

    @Override
    public String getAllCustomOps() {
        return callString("getAllCustomOps");
    }

    @Override
    public String getAllOperations() {
        return callString("getAllOperations");
    }

    @Override
    public int execCustomOp2(PointerPointer arg0, long arg1, Pointer arg2) {
        return callInt("execCustomOp2", arg0, arg1, arg2);
    }

    @Override
    public int execCustomOp(PointerPointer arg0, long arg1, PointerPointer arg2, PointerPointer arg3, int arg4, PointerPointer arg5, PointerPointer arg6, int arg7, DoublePointer arg8, int arg9, LongPointer arg10, int arg11, BooleanPointer arg12, int arg13, boolean arg14) {
        return callInt("execCustomOp", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14);
    }

    @Override
    public OpaqueShapeList calculateOutputShapes(PointerPointer arg0, long arg1, PointerPointer arg2, int arg3, DoublePointer arg4, int arg5, LongPointer arg6, int arg7) {
        return new OpaqueShapeList(callPointer("calculateOutputShapes", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7));
    }

    @Override
    public OpaqueShapeList calculateOutputShapes2(PointerPointer arg0, long arg1, PointerPointer arg2, PointerPointer arg3, int arg4, DoublePointer arg5, int arg6, LongPointer arg7, int arg8, BooleanPointer arg9, int arg10, IntPointer arg11, int arg12) {
        return new OpaqueShapeList(callPointer("calculateOutputShapes2", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12));
    }

    @Override
    public long getShapeListSize(OpaqueShapeList arg0) {
        return callLong("getShapeListSize", arg0);
    }

    @Override
    public LongPointer getShape(OpaqueShapeList arg0, long arg1) {
        return new LongPointer(callPointer("getShape", arg0, arg1));
    }

    @Override
    public int registerGraph(PointerPointer arg0, long arg1, Pointer arg2) {
        return callInt("registerGraph", arg0, arg1, arg2);
    }

    @Override
    public OpaqueVariablesSet executeStoredGraph(PointerPointer arg0, long arg1, PointerPointer arg2, PointerPointer arg3, IntPointer arg4, int arg5) {
        return new OpaqueVariablesSet(callPointer("executeStoredGraph", arg0, arg1, arg2, arg3, arg4, arg5));
    }

    @Override
    public long getVariablesSetSize(OpaqueVariablesSet arg0) {
        return callLong("getVariablesSetSize", arg0);
    }

    @Override
    public int getVariablesSetStatus(OpaqueVariablesSet arg0) {
        return callInt("getVariablesSetStatus", arg0);
    }

    @Override
    public OpaqueVariable getVariable(OpaqueVariablesSet arg0, long arg1) {
        return new OpaqueVariable(callPointer("getVariable", arg0, arg1));
    }

    @Override
    public int getVariableId(OpaqueVariable arg0) {
        return callInt("getVariableId", arg0);
    }

    @Override
    public int getVariableIndex(OpaqueVariable arg0) {
        return callInt("getVariableIndex", arg0);
    }

    @Override
    public String getVariableName(OpaqueVariable arg0) {
        return callString("getVariableName", arg0);
    }

    @Override
    public LongPointer getVariableShape(OpaqueVariable arg0) {
        return new LongPointer(callPointer("getVariableShape", arg0));
    }

    @Override
    public Pointer getVariableBuffer(OpaqueVariable arg0) {
        return callPointer("getVariableBuffer", arg0);
    }

    @Override
    public void deleteResultWrapper(Pointer arg0) {
        call("deleteResultWrapper", arg0);
    }

    @Override
    public void deleteShapeList(Pointer arg0) {
        call("deleteShapeList", arg0);
    }

    @Override
    public int unregisterGraph(PointerPointer arg0, long arg1) {
        return callInt("unregisterGraph", arg0, arg1);
    }

    @Override
    public void deleteIntArray(Pointer arg0) {
        call("deleteIntArray", arg0);
    }

    @Override
    public void deleteLongArray(Pointer arg0) {
        call("deleteLongArray", arg0);
    }

    @Override
    public void deletePointerArray(Pointer arg0) {
        call("deletePointerArray", arg0);
    }

    @Override
    public void deleteNPArrayStruct(Pointer arg0) {
        call("deleteNPArrayStruct", arg0);
    }

    @Override
    public void deleteNPArrayMap(Pointer arg0) {
        call("deleteNPArrayMap", arg0);
    }

    @Override
    public void deleteVariablesSet(OpaqueVariablesSet arg0) {
        call("deleteVariablesSet", arg0);
    }

    @Override
    public Pointer getGraphState(long arg0) {
        return callPointer("getGraphState", arg0);
    }

    @Override
    public void deleteGraphState(Pointer arg0) {
        call("deleteGraphState", arg0);
    }

    @Override
    public int estimateThreshold(PointerPointer arg0, Pointer arg1, LongPointer arg2, int arg3, float arg4) {
        return callInt("estimateThreshold", arg0, arg1, arg2, arg3, arg4);
    }

    @Override
    public int execCustomOpWithScope(PointerPointer arg0, Pointer arg1, long arg2, long[] arg3, int arg4, PointerPointer arg5, PointerPointer arg6, int arg7, PointerPointer arg8, PointerPointer arg9, int arg10) {
        return callInt("execCustomOpWithScope", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    @Override
    public void scatterUpdate(PointerPointer arg0, int arg1, int arg2, Pointer arg3, LongPointer arg4, LongPointer arg5, Pointer arg6, LongPointer arg7, LongPointer arg8, Pointer arg9, LongPointer arg10, LongPointer arg11, Pointer arg12, LongPointer arg13, LongPointer arg14, Pointer arg15, LongPointer arg16, Pointer arg17, LongPointer arg18) {
        call("scatterUpdate", arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18);
    }

    @Override
    public Pointer createUtf8String(PointerPointer arg0, String arg1, int arg2) {
        return callPointer("createUtf8String", arg0, arg1, arg2);
    }

    @Override
    public long getUtf8StringLength(PointerPointer arg0, Pointer arg1) {
        return callLong("getUtf8StringLength", arg0, arg1);
    }

    @Override
    public BytePointer getUtf8StringBuffer(PointerPointer arg0, Pointer arg1) {
        return new BytePointer(callPointer("getUtf8StringBuffer", arg0, arg1));
    }

    @Override
    public void deleteUtf8String(PointerPointer arg0, Pointer arg1) {
        call("deleteUtf8String", arg0, arg1);
    }

    @Override
    public void inspectArray(PointerPointer arg0, Pointer arg1, LongPointer arg2, Pointer arg3, LongPointer arg4, Pointer arg5) {
        call("inspectArray", arg0, arg1, arg2, arg3, arg4, arg5);
    }

    @Override
    public void tryPointer(Pointer arg0, Pointer arg1, int arg2) {
        call("tryPointer", arg0, arg1, arg2);
    }

    @Override
    public int dataTypeFromNpyHeader(Pointer arg0) {
        return callInt("dataTypeFromNpyHeader", arg0);
    }

    @Override
    public OpaqueConstantDataBuffer shapeBuffer(int arg0, LongPointer arg1, LongPointer arg2, int arg3, char arg4, long arg5, boolean arg6) {
        return new OpaqueConstantDataBuffer(callPointer("shapeBuffer", arg0, arg1, arg2, arg3, arg4, arg5, arg6));
    }

    @Override
    public OpaqueConstantDataBuffer constantBufferDouble(int arg0, DoublePointer arg1, int arg2) {
        return new OpaqueConstantDataBuffer(callPointer("constantBufferDouble", arg0, arg1, arg2));
    }

    @Override
    public OpaqueConstantDataBuffer constantBufferLong(int arg0, LongPointer arg1, int arg2) {
        return new OpaqueConstantDataBuffer(callPointer("constantBufferLong", arg0, arg1, arg2));
    }

    @Override
    public Pointer getConstantDataBufferPrimary(OpaqueConstantDataBuffer arg0) {
        return callPointer("getConstantDataBufferPrimary", arg0);
    }

    @Override
    public Pointer getConstantDataBufferSpecial(OpaqueConstantDataBuffer arg0) {
        return callPointer("getConstantDataBufferSpecial", arg0);
    }

    @Override
    public long getConstantDataBufferLength(OpaqueConstantDataBuffer arg0) {
        return callLong("getConstantDataBufferLength", arg0);
    }

    @Override
    public long getConstantDataBufferSizeOf(OpaqueConstantDataBuffer arg0) {
        return callLong("getConstantDataBufferSizeOf", arg0);
    }

    @Override
    public void deleteShapeBuffer(OpaqueConstantDataBuffer arg0) {
        call("deleteShapeBuffer", arg0);
    }

    @Override
    public OpaqueContext createGraphContext(int arg0) {
        return new OpaqueContext(callPointer("createGraphContext", arg0));
    }

    @Override
    public OpaqueRandomGenerator getGraphContextRandomGenerator(OpaqueContext arg0) {
        return new OpaqueRandomGenerator(callPointer("getGraphContextRandomGenerator", arg0));
    }

    @Override
    public void markGraphContextInplace(OpaqueContext arg0, boolean arg1) {
        call("markGraphContextInplace", arg0, arg1);
    }

    @Override
    public void setGraphContextCudaContext(OpaqueContext arg0, Pointer arg1, Pointer arg2, Pointer arg3) {
        call("setGraphContextCudaContext", arg0, arg1, arg2, arg3);
    }

    @Override
    public void setGraphContextInputArray(OpaqueContext arg0, int arg1, Pointer arg2, Pointer arg3, Pointer arg4, Pointer arg5) {
        call("setGraphContextInputArray", arg0, arg1, arg2, arg3, arg4, arg5);
    }

    @Override
    public void setGraphContextOutputArray(OpaqueContext arg0, int arg1, Pointer arg2, Pointer arg3, Pointer arg4, Pointer arg5) {
        call("setGraphContextOutputArray", arg0, arg1, arg2, arg3, arg4, arg5);
    }

    @Override
    public void setGraphContextInputBuffer(OpaqueContext arg0, int arg1, OpaqueDataBuffer arg2, Pointer arg3, Pointer arg4) {
        call("setGraphContextInputBuffer", arg0, arg1, arg2, arg3, arg4);
    }

    @Override
    public void setGraphContextOutputBuffer(OpaqueContext arg0, int arg1, OpaqueDataBuffer arg2, Pointer arg3, Pointer arg4) {
        call("setGraphContextOutputBuffer", arg0, arg1, arg2, arg3, arg4);
    }

    @Override
    public void setGraphContextTArguments(OpaqueContext arg0, DoublePointer arg1, int arg2) {
        call("setGraphContextTArguments", arg0, arg1, arg2);
    }

    @Override
    public void setGraphContextIArguments(OpaqueContext arg0, LongPointer arg1, int arg2) {
        call("setGraphContextIArguments", arg0, arg1, arg2);
    }

    @Override
    public void setGraphContextDArguments(OpaqueContext arg0, IntPointer arg1, int arg2) {
        call("setGraphContextDArguments", arg0, arg1, arg2);
    }

    @Override
    public void setGraphContextBArguments(OpaqueContext arg0, BooleanPointer arg1, int arg2) {
        call("setGraphContextBArguments", arg0, arg1, arg2);
    }

    @Override
    public void ctxAllowHelpers(OpaqueContext arg0, boolean arg1) {
        call("ctxAllowHelpers", arg0, arg1);
    }

    @Override
    public void ctxSetExecutionMode(OpaqueContext arg0, int arg1) {
        call("ctxSetExecutionMode", arg0, arg1);
    }

    @Override
    public void ctxShapeFunctionOverride(OpaqueContext arg0, boolean arg1) {
        call("ctxShapeFunctionOverride", arg0, arg1);
    }

    @Override
    public void ctxPurge(OpaqueContext arg0) {
        call("ctxPurge", arg0);
    }

    @Override
    public void deleteGraphContext(OpaqueContext arg0) {
        call("deleteGraphContext", arg0);
    }

    @Override
    public OpaqueRandomGenerator createRandomGenerator(long arg0, long arg1) {
        return new OpaqueRandomGenerator(callPointer("createRandomGenerator", arg0, arg1));
    }

    @Override
    public long getRandomGeneratorRootState(OpaqueRandomGenerator arg0) {
        return callLong("getRandomGeneratorRootState", arg0);
    }

    @Override
    public long getRandomGeneratorNodeState(OpaqueRandomGenerator arg0) {
        return callLong("getRandomGeneratorNodeState", arg0);
    }

    @Override
    public void setRandomGeneratorStates(OpaqueRandomGenerator arg0, long arg1, long arg2) {
        call("setRandomGeneratorStates", arg0, arg1, arg2);
    }

    @Override
    public int getRandomGeneratorRelativeInt(OpaqueRandomGenerator arg0, long arg1) {
        return callInt("getRandomGeneratorRelativeInt", arg0, arg1);
    }

    @Override
    public long getRandomGeneratorRelativeLong(OpaqueRandomGenerator arg0, long arg1) {
        return callLong("getRandomGeneratorRelativeLong", arg0, arg1);
    }

    @Override
    public void deleteRandomGenerator(OpaqueRandomGenerator arg0) {
        call("deleteRandomGenerator", arg0);
    }

    @Override
    public String runLightBenchmarkSuit(boolean arg0) {
        return callString("runLightBenchmarkSuit", arg0);
    }

    @Override
    public String runFullBenchmarkSuit(boolean arg0) {
        return callString("runFullBenchmarkSuit", arg0);
    }

    @Override
    public long getCachedMemory(int arg0) {
        return callLong("getCachedMemory", arg0);
    }

    @Override
    public OpaqueLaunchContext defaultLaunchContext() {
        return new OpaqueLaunchContext(callPointer("defaultLaunchContext"));
    }

    @Override
    public Pointer lcScalarPointer(OpaqueLaunchContext lc) {
        return callPointer("lcScalarPointer");
    }

    @Override
    public Pointer lcReductionPointer(OpaqueLaunchContext lc) {
        return callPointer("lcReductionPointer");
    }

    @Override
    public Pointer lcAllocationPointer(OpaqueLaunchContext lc) {
        return callPointer("lcAllocationPointer");
    }

    @Override
    public Pointer lcExecutionStream(OpaqueLaunchContext lc) {
        return callPointer("lcExecutionStream");
    }

    @Override
    public Pointer lcCopyStream(OpaqueLaunchContext lc) {
        return callPointer("lcCopyStream");
    }

    @Override
    public Pointer lcBlasHandle(OpaqueLaunchContext lc) {
        return callPointer("lcBlasHandle");
    }

    @Override
    public Pointer lcSolverHandle(OpaqueLaunchContext lc) {
        return callPointer("lcSolverHandle");
    }

    @Override
    public int lastErrorCode() {
        return callInt("lastErrorCode");
    }

    @Override
    public String lastErrorMessage() {
        return callString("lastErrorMessage");
    }

    @Override
    public boolean isBlasVersionMatches(int major, int minor, int build) {
        return callBoolean("isBlasVersionMatches", major, minor, build);
    }

    @Override
    public int binaryLevel() {
        return callInt("binaryLevel");
    }

    @Override
    public int optimalLevel() {
        return callInt("optimalLevel");
    }

    @Override
    public boolean isMinimalRequirementsMet() {
        return callBoolean("isMinimalRequirementsMet");
    }

    @Override
    public boolean isOptimalRequirementsMet() {
        return callBoolean("isOptimalRequirementsMet");
    }

    @Override
    public OpaqueDataBuffer allocateDataBuffer(long elements, int dataType, boolean allocateBoth) {
        return new OpaqueDataBuffer(callPointer("allocateDataBuffer", elements, dataType, allocateBoth));
    }

    @Override
    public OpaqueDataBuffer dbAllocateDataBuffer(long elements, int dataType, boolean allocateBoth) {
        return new OpaqueDataBuffer(callPointer("dbAllocateDataBuffer", elements, dataType, allocateBoth));
    }

    @Override
    public OpaqueDataBuffer dbCreateExternalDataBuffer(long elements, int dataType, Pointer primary, Pointer special) {
        return new OpaqueDataBuffer(callPointer("dbCreateExternalDataBuffer", elements, dataType, primary, special));
    }

    @Override
    public OpaqueDataBuffer dbCreateView(OpaqueDataBuffer dataBuffer, long length, long offset) {
        return new OpaqueDataBuffer(callPointer("dbCreateView", dataBuffer, length, offset));
    }

    @Override
    public Pointer dbPrimaryBuffer(OpaqueDataBuffer dataBuffer) {
        return callPointer("dbPrimaryBuffer", dataBuffer);
    }

    @Override
    public Pointer dbSpecialBuffer(OpaqueDataBuffer dataBuffer) {
        return callPointer("dbSpecialBuffer", dataBuffer);
    }

    @Override
    public void dbExpandBuffer(OpaqueDataBuffer dataBuffer, long elements) {
        call("dbExpandBuffer", dataBuffer, elements);
    }

    @Override
    public void dbAllocatePrimaryBuffer(OpaqueDataBuffer dataBuffer) {
        call("dbAllocatePrimaryBuffer", dataBuffer);
    }

    @Override
    public void dbAllocateSpecialBuffer(OpaqueDataBuffer dataBuffer) {
        call("dbAllocateSpecialBuffer", dataBuffer);
    }

    @Override
    public void dbSetPrimaryBuffer(OpaqueDataBuffer dataBuffer, Pointer primaryBuffer, long numBytes) {
        call("dbSetPrimaryBuffer", dataBuffer, primaryBuffer, numBytes);
    }

    @Override
    public void dbSetSpecialBuffer(OpaqueDataBuffer dataBuffer, Pointer specialBuffer, long numBytes) {
        call("dbSetSpecialBuffer", dataBuffer, specialBuffer, numBytes);
    }

    @Override
    public void dbSyncToSpecial(OpaqueDataBuffer dataBuffer) {
        call("dbSyncToSpecial", dataBuffer);
    }

    @Override
    public void dbSyncToPrimary(OpaqueDataBuffer dataBuffer) {
        call("dbSyncToPrimary", dataBuffer);
    }

    @Override
    public void dbTickHostRead(OpaqueDataBuffer dataBuffer) {
        call("dbTickHostRead", dataBuffer);
    }

    @Override
    public void dbTickHostWrite(OpaqueDataBuffer dataBuffer) {
        call("dbTickHostWrite", dataBuffer);
    }

    @Override
    public void dbTickDeviceRead(OpaqueDataBuffer dataBuffer) {
        call("dbTickDeviceRead", dataBuffer);
    }

    @Override
    public void dbTickDeviceWrite(OpaqueDataBuffer dataBuffer) {
        call("dbTickDeviceWrite", dataBuffer);
    }

    @Override
    public void deleteDataBuffer(OpaqueDataBuffer dataBuffer) {
        call("deleteDataBuffer", dataBuffer);
    }

    @Override
    public void dbClose(OpaqueDataBuffer dataBuffer) {
        call("dbClose", dataBuffer);
    }

    @Override
    public int dbLocality(OpaqueDataBuffer dataBuffer) {
        return callInt("dbLocality", dataBuffer);
    }

    @Override
    public int dbDeviceId(OpaqueDataBuffer dataBuffer) {
        return callInt("dbDeviceId", dataBuffer);
    }

    @Override
    public void dbSetDeviceId(OpaqueDataBuffer dataBuffer, int deviceId) {
        call("dbSetDeviceId", dataBuffer, deviceId);
    }

    @Override
    public void dbExpand(OpaqueDataBuffer dataBuffer, long newLength) {
        call("dbExpand", dataBuffer, newLength);
    }

}
