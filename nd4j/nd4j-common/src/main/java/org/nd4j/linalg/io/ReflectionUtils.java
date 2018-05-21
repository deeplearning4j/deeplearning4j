package org.nd4j.linalg.io;

import java.lang.reflect.*;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.regex.Pattern;

public abstract class ReflectionUtils {
    private static final Pattern CGLIB_RENAMED_METHOD_PATTERN = Pattern.compile("CGLIB\\$(.+)\\$\\d+");
    public static ReflectionUtils.FieldFilter COPYABLE_FIELDS = new ReflectionUtils.FieldFilter() {
        public boolean matches(Field field) {
            return !Modifier.isStatic(field.getModifiers()) && !Modifier.isFinal(field.getModifiers());
        }
    };
    public static ReflectionUtils.MethodFilter NON_BRIDGED_METHODS = new ReflectionUtils.MethodFilter() {
        public boolean matches(Method method) {
            return !method.isBridge();
        }
    };
    public static ReflectionUtils.MethodFilter USER_DECLARED_METHODS = new ReflectionUtils.MethodFilter() {
        public boolean matches(Method method) {
            return !method.isBridge() && method.getDeclaringClass() != Object.class;
        }
    };

    public ReflectionUtils() {}

    public static Field findField(Class<?> clazz, String name) {
        return findField(clazz, name, null);
    }

    public static Field findField(Class<?> clazz, String name, Class<?> type) {
        Assert.notNull(clazz, "Class must not be null");
        Assert.isTrue(name != null || type != null, "Either name or opType of the field must be specified");

        for (Class searchType = clazz; !Object.class.equals(searchType) && searchType != null; searchType =
                        searchType.getSuperclass()) {
            Field[] fields = searchType.getDeclaredFields();
            Field[] arr$ = fields;
            int len$ = fields.length;

            for (int i$ = 0; i$ < len$; ++i$) {
                Field field = arr$[i$];
                if ((name == null || name.equals(field.getName())) && (type == null || type.equals(field.getType()))) {
                    return field;
                }
            }
        }

        return null;
    }

    public static void setField(Field field, Object target, Object value) {
        try {
            field.set(target, value);
        } catch (IllegalAccessException var4) {
            handleReflectionException(var4);
            throw new IllegalStateException("Unexpected reflection exception - " + var4.getClass().getName() + ": "
                            + var4.getMessage());
        }
    }

    public static Object getField(Field field, Object target) {
        try {
            return field.get(target);
        } catch (IllegalAccessException var3) {
            handleReflectionException(var3);
            throw new IllegalStateException("Unexpected reflection exception - " + var3.getClass().getName() + ": "
                            + var3.getMessage());
        }
    }

    public static Method findMethod(Class<?> clazz, String name) {
        return findMethod(clazz, name, new Class[0]);
    }

    public static Method findMethod(Class<?> clazz, String name, Class<?>... paramTypes) {
        Assert.notNull(clazz, "Class must not be null");
        Assert.notNull(name, "Method name must not be null");

        for (Class searchType = clazz; searchType != null; searchType = searchType.getSuperclass()) {
            Method[] methods = searchType.isInterface() ? searchType.getMethods() : searchType.getDeclaredMethods();
            Method[] arr$ = methods;
            int len$ = methods.length;

            for (int i$ = 0; i$ < len$; ++i$) {
                Method method = arr$[i$];
                if (name.equals(method.getName())
                                && (paramTypes == null || Arrays.equals(paramTypes, method.getParameterTypes()))) {
                    return method;
                }
            }
        }

        return null;
    }

    public static Object invokeMethod(Method method, Object target) {
        return invokeMethod(method, target, new Object[0]);
    }

    public static Object invokeMethod(Method method, Object target, Object... args) {
        try {
            return method.invoke(target, args);
        } catch (Exception var4) {
            handleReflectionException(var4);
            throw new IllegalStateException("Should never get here");
        }
    }

    public static Object invokeJdbcMethod(Method method, Object target) throws SQLException {
        return invokeJdbcMethod(method, target, new Object[0]);
    }

    public static Object invokeJdbcMethod(Method method, Object target, Object... args) throws SQLException {
        try {
            return method.invoke(target, args);
        } catch (IllegalAccessException var4) {
            handleReflectionException(var4);
        } catch (InvocationTargetException var5) {
            if (var5.getTargetException() instanceof SQLException) {
                throw (SQLException) var5.getTargetException();
            }

            handleInvocationTargetException(var5);
        }

        throw new IllegalStateException("Should never get here");
    }

    public static void handleReflectionException(Exception ex) {
        if (ex instanceof NoSuchMethodException) {
            throw new IllegalStateException("Method not found: " + ex.getMessage());
        } else if (ex instanceof IllegalAccessException) {
            throw new IllegalStateException("Could not access method: " + ex.getMessage());
        } else {
            if (ex instanceof InvocationTargetException) {
                handleInvocationTargetException((InvocationTargetException) ex);
            }

            if (ex instanceof RuntimeException) {
                throw (RuntimeException) ex;
            } else {
                throw new UndeclaredThrowableException(ex);
            }
        }
    }

    public static void handleInvocationTargetException(InvocationTargetException ex) {
        rethrowRuntimeException(ex.getTargetException());
    }

    public static void rethrowRuntimeException(Throwable ex) {
        if (ex instanceof RuntimeException) {
            throw (RuntimeException) ex;
        } else if (ex instanceof Error) {
            throw (Error) ex;
        } else {
            throw new UndeclaredThrowableException(ex);
        }
    }

    public static void rethrowException(Throwable ex) throws Exception {
        if (ex instanceof Exception) {
            throw (Exception) ex;
        } else if (ex instanceof Error) {
            throw (Error) ex;
        } else {
            throw new UndeclaredThrowableException(ex);
        }
    }

    public static boolean declaresException(Method method, Class<?> exceptionType) {
        Assert.notNull(method, "Method must not be null");
        Class[] declaredExceptions = method.getExceptionTypes();
        Class[] arr$ = declaredExceptions;
        int len$ = declaredExceptions.length;

        for (int i$ = 0; i$ < len$; ++i$) {
            Class declaredException = arr$[i$];
            if (declaredException.isAssignableFrom(exceptionType)) {
                return true;
            }
        }

        return false;
    }

    public static boolean isPublicStaticFinal(Field field) {
        int modifiers = field.getModifiers();
        return Modifier.isPublic(modifiers) && Modifier.isStatic(modifiers) && Modifier.isFinal(modifiers);
    }

    public static boolean isEqualsMethod(Method method) {
        if (method != null && method.getName().equals("equals")) {
            Class[] paramTypes = method.getParameterTypes();
            return paramTypes.length == 1 && paramTypes[0] == Object.class;
        } else {
            return false;
        }
    }

    public static boolean isHashCodeMethod(Method method) {
        return method != null && method.getName().equals("hashCode") && method.getParameterTypes().length == 0;
    }

    public static boolean isToStringMethod(Method method) {
        return method != null && method.getName().equals("toString") && method.getParameterTypes().length == 0;
    }

    public static boolean isObjectMethod(Method method) {
        try {
            Object.class.getDeclaredMethod(method.getName(), method.getParameterTypes());
            return true;
        } catch (SecurityException var2) {
            return false;
        } catch (NoSuchMethodException var3) {
            return false;
        }
    }

    public static boolean isCglibRenamedMethod(Method renamedMethod) {
        return CGLIB_RENAMED_METHOD_PATTERN.matcher(renamedMethod.getName()).matches();
    }

    public static void makeAccessible(Field field) {
        if ((!Modifier.isPublic(field.getModifiers()) || !Modifier.isPublic(field.getDeclaringClass().getModifiers())
                        || Modifier.isFinal(field.getModifiers())) && !field.isAccessible()) {
            field.setAccessible(true);
        }

    }

    public static void makeAccessible(Method method) {
        if ((!Modifier.isPublic(method.getModifiers()) || !Modifier.isPublic(method.getDeclaringClass().getModifiers()))
                        && !method.isAccessible()) {
            method.setAccessible(true);
        }

    }

    public static void makeAccessible(Constructor<?> ctor) {
        if ((!Modifier.isPublic(ctor.getModifiers()) || !Modifier.isPublic(ctor.getDeclaringClass().getModifiers()))
                        && !ctor.isAccessible()) {
            ctor.setAccessible(true);
        }

    }

    public static void doWithMethods(Class<?> clazz, ReflectionUtils.MethodCallback mc)
                    throws IllegalArgumentException {
        doWithMethods(clazz, mc, null);
    }

    public static void doWithMethods(Class<?> clazz, ReflectionUtils.MethodCallback mc, ReflectionUtils.MethodFilter mf)
                    throws IllegalArgumentException {
        Method[] methods = clazz.getDeclaredMethods();
        Method[] arr$ = methods;
        int len$ = methods.length;

        int i$;
        for (i$ = 0; i$ < len$; ++i$) {
            Method superIfc = arr$[i$];
            if (mf == null || mf.matches(superIfc)) {
                try {
                    mc.doWith(superIfc);
                } catch (IllegalAccessException var9) {
                    throw new IllegalStateException(
                                    "Shouldn\'t be illegal to access method \'" + superIfc.getName() + "\': " + var9);
                }
            }
        }

        if (clazz.getSuperclass() != null) {
            doWithMethods(clazz.getSuperclass(), mc, mf);
        } else if (clazz.isInterface()) {
            Class[] var10 = clazz.getInterfaces();
            len$ = var10.length;

            for (i$ = 0; i$ < len$; ++i$) {
                Class var11 = var10[i$];
                doWithMethods(var11, mc, mf);
            }
        }

    }

    public static Method[] getAllDeclaredMethods(Class<?> leafClass) throws IllegalArgumentException {
        final ArrayList methods = new ArrayList(32);
        doWithMethods(leafClass, new ReflectionUtils.MethodCallback() {
            public void doWith(Method method) {
                methods.add(method);
            }
        });
        return (Method[]) methods.toArray(new Method[methods.size()]);
    }

    public static Method[] getUniqueDeclaredMethods(Class<?> leafClass) throws IllegalArgumentException {
        final ArrayList methods = new ArrayList(32);
        doWithMethods(leafClass, new ReflectionUtils.MethodCallback() {
            public void doWith(Method method) {
                boolean knownSignature = false;
                Method methodBeingOverriddenWithCovariantReturnType = null;
                Iterator i$ = methods.iterator();

                while (i$.hasNext()) {
                    Method existingMethod = (Method) i$.next();
                    if (method.getName().equals(existingMethod.getName())
                                    && Arrays.equals(method.getParameterTypes(), existingMethod.getParameterTypes())) {
                        if (existingMethod.getReturnType() != method.getReturnType()
                                        && existingMethod.getReturnType().isAssignableFrom(method.getReturnType())) {
                            methodBeingOverriddenWithCovariantReturnType = existingMethod;
                            break;
                        }

                        knownSignature = true;
                        break;
                    }
                }

                if (methodBeingOverriddenWithCovariantReturnType != null) {
                    methods.remove(methodBeingOverriddenWithCovariantReturnType);
                }

                if (!knownSignature && !ReflectionUtils.isCglibRenamedMethod(method)) {
                    methods.add(method);
                }

            }
        });
        return (Method[]) methods.toArray(new Method[methods.size()]);
    }

    public static void doWithFields(Class<?> clazz, ReflectionUtils.FieldCallback fc) throws IllegalArgumentException {
        doWithFields(clazz, fc, null);
    }

    public static void doWithFields(Class<?> clazz, ReflectionUtils.FieldCallback fc, ReflectionUtils.FieldFilter ff)
                    throws IllegalArgumentException {
        Class targetClass = clazz;

        do {
            Field[] fields = targetClass.getDeclaredFields();
            Field[] arr$ = fields;
            int len$ = fields.length;

            for (int i$ = 0; i$ < len$; ++i$) {
                Field field = arr$[i$];
                if (ff == null || ff.matches(field)) {
                    try {
                        fc.doWith(field);
                    } catch (IllegalAccessException var10) {
                        throw new IllegalStateException(
                                        "Shouldn\'t be illegal to access field \'" + field.getName() + "\': " + var10);
                    }
                }
            }

            targetClass = targetClass.getSuperclass();
        } while (targetClass != null && targetClass != Object.class);

    }

    public static void shallowCopyFieldState(final Object src, final Object dest) throws IllegalArgumentException {
        if (src == null) {
            throw new IllegalArgumentException("Source for field copy cannot be null");
        } else if (dest == null) {
            throw new IllegalArgumentException("Destination for field copy cannot be null");
        } else if (!src.getClass().isAssignableFrom(dest.getClass())) {
            throw new IllegalArgumentException("Destination class [" + dest.getClass().getName()
                            + "] must be same or subclass as source class [" + src.getClass().getName() + "]");
        } else {
            doWithFields(src.getClass(), new ReflectionUtils.FieldCallback() {
                public void doWith(Field field) throws IllegalArgumentException, IllegalAccessException {
                    ReflectionUtils.makeAccessible(field);
                    Object srcValue = field.get(src);
                    field.set(dest, srcValue);
                }
            }, COPYABLE_FIELDS);
        }
    }

    public interface FieldFilter {
        boolean matches(Field var1);
    }

    public interface FieldCallback {
        void doWith(Field var1) throws IllegalArgumentException, IllegalAccessException;
    }

    public interface MethodFilter {
        boolean matches(Method var1);
    }

    public interface MethodCallback {
        void doWith(Method var1) throws IllegalArgumentException, IllegalAccessException;
    }
}
