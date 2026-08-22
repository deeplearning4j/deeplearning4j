/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.jtpu;

import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.config.MemoryConfig;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

/** Binding-neutral Environment facade over the generated TPU JavaCPP runtime. */
public final class TpuEnvironment {

    private static final String NATIVE_ENVIRONMENT_CLASS =
            "org.nd4j.linalg.jtpu.bindings.Nd4jTpu$Environment";

    private TpuEnvironment() {
    }

    private static final class Holder {
        private static final Environment INSTANCE = createEnvironment();
    }

    public static Environment getInstance() {
        return Holder.INSTANCE;
    }

    private static Environment createEnvironment() {
        try {
            Class<?> nativeClass = Class.forName(
                    NATIVE_ENVIRONMENT_CLASS, true, TpuEnvironment.class.getClassLoader());
            Object nativeEnvironment = nativeClass.getMethod("getInstance").invoke(null);
            InvocationHandler handler = new NativeEnvironmentHandler(nativeEnvironment);
            return (Environment) Proxy.newProxyInstance(
                    TpuEnvironment.class.getClassLoader(),
                    new Class<?>[]{Environment.class}, handler);
        } catch (ReflectiveOperationException failure) {
            throw new IllegalStateException(
                    "Unable to initialize the generated TPU native Environment binding", failure);
        }
    }

    private static final class NativeEnvironmentHandler implements InvocationHandler {
        private final Object delegate;
        private volatile boolean truncateLogStrings;
        private volatile boolean trackWorkspaceOpenClose;
        private volatile boolean funcTracePrintJavaOnly;
        private volatile boolean variableTracingEnabled;
        private volatile int workspaceEventsToKeep = -1;

        private NativeEnvironmentHandler(Object delegate) {
            this.delegate = delegate;
        }

        @Override
        public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
            String name = method.getName();
            if (method.getDeclaringClass() == Object.class) {
                switch (name) {
                    case "toString":
                        return "TpuEnvironment[" + delegate + "]";
                    case "hashCode":
                        return System.identityHashCode(proxy);
                    case "equals":
                        return proxy == args[0];
                    default:
                        throw new UnsupportedOperationException(name);
                }
            }

            switch (name) {
                case "isTruncateNDArrayLogStrings": return truncateLogStrings;
                case "setTruncateLogStrings": truncateLogStrings = (Boolean) args[0]; return null;
                case "numWorkspaceEventsToKeep": return workspaceEventsToKeep;
                case "isTrackWorkspaceOpenClose": return trackWorkspaceOpenClose;
                case "setTrackWorkspaceOpenClose": trackWorkspaceOpenClose = (Boolean) args[0]; return null;
                case "isFuncTracePrintJavaOnly": return funcTracePrintJavaOnly;
                case "setFuncTracePrintJavaOnly": funcTracePrintJavaOnly = (Boolean) args[0]; return null;
                case "isVariableTracingEnabled": return variableTracingEnabled;
                case "setVariableTracingEnabled": variableTracingEnabled = (Boolean) args[0]; return null;
                case "memory": return MemoryConfig.getInstance();
                default:
                    break;
            }

            String nativeName;
            switch (name) {
                case "setMaxSpecialMemory": nativeName = "setMaxSpecialyMemory"; break;
                case "setFuncTraceForAllocate": nativeName = "setFuncTracePrintAllocate"; break;
                case "setFuncTraceForDeallocate": nativeName = "setFuncTracePrintDeallocate"; break;
                default: nativeName = name;
            }
            Method nativeMethod = findNativeMethod(nativeName, method.getParameterTypes());
            BytePointer convertedString = null;
            Object[] nativeArgs = args;
            if (nativeMethod == null && args != null && args.length == 1
                    && args[0] instanceof String) {
                nativeMethod = findNativeMethod(nativeName, new Class<?>[]{BytePointer.class});
                if (nativeMethod != null) {
                    convertedString = new BytePointer((String) args[0]);
                    nativeArgs = new Object[]{convertedString};
                }
            }
            if (nativeMethod == null) {
                if (method.isDefault()) {
                    MethodHandles.Lookup lookup = MethodHandles.privateLookupIn(
                            method.getDeclaringClass(), MethodHandles.lookup());
                    MethodHandle handle = lookup.unreflectSpecial(
                            method, method.getDeclaringClass()).bindTo(proxy);
                    return handle.invokeWithArguments(args == null ? new Object[0] : args);
                }
                throw new UnsupportedOperationException(
                        "TPU native Environment does not implement " + method);
            }

            try {
                Object value = nativeMethod.invoke(delegate, nativeArgs);
                if (method.getReturnType() == String.class && value instanceof BytePointer) {
                    return ((BytePointer) value).getString();
                }
                return value;
            } catch (InvocationTargetException failure) {
                throw failure.getCause();
            } finally {
                if (convertedString != null) {
                    convertedString.close();
                }
            }
        }

        private Method findNativeMethod(String name, Class<?>[] parameterTypes) {
            try {
                return delegate.getClass().getMethod(name, parameterTypes);
            } catch (NoSuchMethodException ignored) {
                for (Method candidate : delegate.getClass().getMethods()) {
                    if (!candidate.getName().equals(name) ||
                            candidate.getParameterCount() != parameterTypes.length) {
                        continue;
                    }
                    Class<?>[] nativeParameters = candidate.getParameterTypes();
                    boolean compatible = true;
                    for (int i = 0; i < parameterTypes.length; ++i) {
                        if (!wrap(nativeParameters[i]).isAssignableFrom(wrap(parameterTypes[i]))) {
                            compatible = false;
                            break;
                        }
                    }
                    if (compatible) {
                        return candidate;
                    }
                }
                return null;
            }
        }

        private static Class<?> wrap(Class<?> type) {
            if (!type.isPrimitive()) return type;
            if (type == boolean.class) return Boolean.class;
            if (type == byte.class) return Byte.class;
            if (type == short.class) return Short.class;
            if (type == int.class) return Integer.class;
            if (type == long.class) return Long.class;
            if (type == float.class) return Float.class;
            if (type == double.class) return Double.class;
            if (type == char.class) return Character.class;
            return type;
        }
    }
}
