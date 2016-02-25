package org.nd4j.linalg.io;

import java.lang.reflect.Array;
import java.util.Arrays;

public abstract class ObjectUtils {
    private static final int INITIAL_HASH = 7;
    private static final int MULTIPLIER = 31;
    private static final String EMPTY_STRING = "";
    private static final String NULL_STRING = "null";
    private static final String ARRAY_START = "{";
    private static final String ARRAY_END = "}";
    private static final String EMPTY_ARRAY = "{}";
    private static final String ARRAY_ELEMENT_SEPARATOR = ", ";

    public ObjectUtils() {
    }

    public static boolean isCheckedException(Throwable ex) {
        return !(ex instanceof RuntimeException) && !(ex instanceof Error);
    }

    public static boolean isCompatibleWithThrowsClause(Throwable ex, Class[] declaredExceptions) {
        if(!isCheckedException(ex)) {
            return true;
        } else {
            if(declaredExceptions != null) {
                for(int i = 0; i < declaredExceptions.length; ++i) {
                    if(declaredExceptions[i].isAssignableFrom(ex.getClass())) {
                        return true;
                    }
                }
            }

            return false;
        }
    }

    public static boolean isArray(Object obj) {
        return obj != null && obj.getClass().isArray();
    }

    public static boolean isEmpty(Object[] array) {
        return array == null || array.length == 0;
    }

    public static boolean containsElement(Object[] array, Object element) {
        if(array == null) {
            return false;
        } else {
            Object[] arr$ = array;
            int len$ = array.length;

            for(int i$ = 0; i$ < len$; ++i$) {
                Object arrayEle = arr$[i$];
                if(nullSafeEquals(arrayEle, element)) {
                    return true;
                }
            }

            return false;
        }
    }

    public static boolean containsConstant(Enum<?>[] enumValues, String constant) {
        return containsConstant(enumValues, constant, false);
    }

    public static boolean containsConstant(Enum<?>[] enumValues, String constant, boolean caseSensitive) {
        Enum[] arr$ = enumValues;
        int len$ = enumValues.length;
        int i$ = 0;

        while(true) {
            if(i$ >= len$) {
                return false;
            }

            Enum candidate = arr$[i$];
            if(caseSensitive) {
                if(candidate.toString().equals(constant)) {
                    break;
                }
            } else if(candidate.toString().equalsIgnoreCase(constant)) {
                break;
            }

            ++i$;
        }

        return true;
    }

    public static <E extends Enum<?>> E caseInsensitiveValueOf(E[] enumValues, String constant) {
        Enum[] arr$ = enumValues;
        int len$ = enumValues.length;

        for(int i$ = 0; i$ < len$; ++i$) {
            Enum candidate = arr$[i$];
            if(candidate.toString().equalsIgnoreCase(constant)) {
                return (E) candidate;
            }
        }

        throw new IllegalArgumentException(String.format("constant [%s] does not exist in enum type %s", new Object[]{constant, enumValues.getClass().getComponentType().getName()}));
    }

    public static <A, O extends A> A[] addObjectToArray(A[] array, O obj) {
        Class compType = Object.class;
        if(array != null) {
            compType = array.getClass().getComponentType();
        } else if(obj != null) {
            compType = obj.getClass();
        }

        int newArrLength = array != null?array.length + 1:1;
        Object[] newArr = (Object[])((Object[])Array.newInstance(compType, newArrLength));
        if(array != null) {
            System.arraycopy(array, 0, newArr, 0, array.length);
        }

        newArr[newArr.length - 1] = obj;
        return (A[]) newArr;
    }

    public static Object[] toObjectArray(Object source) {
        if(source instanceof Object[]) {
            return (Object[])((Object[])source);
        } else if(source == null) {
            return new Object[0];
        } else if(!source.getClass().isArray()) {
            throw new IllegalArgumentException("Source is not an array: " + source);
        } else {
            int length = Array.getLength(source);
            if(length == 0) {
                return new Object[0];
            } else {
                Class wrapperType = Array.get(source, 0).getClass();
                Object[] newArray = (Object[])((Object[])Array.newInstance(wrapperType, length));

                for(int i = 0; i < length; ++i) {
                    newArray[i] = Array.get(source, i);
                }

                return newArray;
            }
        }
    }

    public static boolean nullSafeEquals(Object o1, Object o2) {
        if(o1 == o2) {
            return true;
        } else if(o1 != null && o2 != null) {
            if(o1.equals(o2)) {
                return true;
            } else {
                if(o1.getClass().isArray() && o2.getClass().isArray()) {
                    if(o1 instanceof Object[] && o2 instanceof Object[]) {
                        return Arrays.equals((Object[])((Object[])o1), (Object[])((Object[])o2));
                    }

                    if(o1 instanceof boolean[] && o2 instanceof boolean[]) {
                        return Arrays.equals((boolean[])((boolean[])o1), (boolean[])((boolean[])o2));
                    }

                    if(o1 instanceof byte[] && o2 instanceof byte[]) {
                        return Arrays.equals((byte[])((byte[])o1), (byte[])((byte[])o2));
                    }

                    if(o1 instanceof char[] && o2 instanceof char[]) {
                        return Arrays.equals((char[])((char[])o1), (char[])((char[])o2));
                    }

                    if(o1 instanceof double[] && o2 instanceof double[]) {
                        return Arrays.equals((double[])((double[])o1), (double[])((double[])o2));
                    }

                    if(o1 instanceof float[] && o2 instanceof float[]) {
                        return Arrays.equals((float[])((float[])o1), (float[])((float[])o2));
                    }

                    if(o1 instanceof int[] && o2 instanceof int[]) {
                        return Arrays.equals((int[])((int[])o1), (int[])((int[])o2));
                    }

                    if(o1 instanceof long[] && o2 instanceof long[]) {
                        return Arrays.equals((long[])((long[])o1), (long[])((long[])o2));
                    }

                    if(o1 instanceof short[] && o2 instanceof short[]) {
                        return Arrays.equals((short[])((short[])o1), (short[])((short[])o2));
                    }
                }

                return false;
            }
        } else {
            return false;
        }
    }

    public static int nullSafeHashCode(Object obj) {
        if(obj == null) {
            return 0;
        } else {
            if(obj.getClass().isArray()) {
                if(obj instanceof Object[]) {
                    return nullSafeHashCode((Object[])((Object[])((Object[])obj)));
                }

                if(obj instanceof boolean[]) {
                    return nullSafeHashCode((boolean[])((boolean[])((boolean[])obj)));
                }

                if(obj instanceof byte[]) {
                    return nullSafeHashCode((byte[])((byte[])((byte[])obj)));
                }

                if(obj instanceof char[]) {
                    return nullSafeHashCode((char[])((char[])((char[])obj)));
                }

                if(obj instanceof double[]) {
                    return nullSafeHashCode((double[])((double[])((double[])obj)));
                }

                if(obj instanceof float[]) {
                    return nullSafeHashCode((float[])((float[])((float[])obj)));
                }

                if(obj instanceof int[]) {
                    return nullSafeHashCode((int[])((int[])((int[])obj)));
                }

                if(obj instanceof long[]) {
                    return nullSafeHashCode((long[])((long[])((long[])obj)));
                }

                if(obj instanceof short[]) {
                    return nullSafeHashCode((short[])((short[])((short[])obj)));
                }
            }

            return obj.hashCode();
        }
    }

    public static int nullSafeHashCode(Object[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + nullSafeHashCode((Object)array[i]);
            }

            return hash;
        }
    }

    public static int nullSafeHashCode(boolean[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + hashCode(array[i]);
            }

            return hash;
        }
    }

    public static int nullSafeHashCode(byte[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + array[i];
            }

            return hash;
        }
    }

    public static int nullSafeHashCode(char[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + array[i];
            }

            return hash;
        }
    }

    public static int nullSafeHashCode(double[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + hashCode(array[i]);
            }

            return hash;
        }
    }

    public static int nullSafeHashCode(float[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + hashCode(array[i]);
            }

            return hash;
        }
    }

    public static int nullSafeHashCode(int[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + array[i];
            }

            return hash;
        }
    }

    public static int nullSafeHashCode(long[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + hashCode(array[i]);
            }

            return hash;
        }
    }

    public static int nullSafeHashCode(short[] array) {
        if(array == null) {
            return 0;
        } else {
            int hash = 7;
            int arraySize = array.length;

            for(int i = 0; i < arraySize; ++i) {
                hash = 31 * hash + array[i];
            }

            return hash;
        }
    }

    public static int hashCode(boolean bool) {
        return bool?1231:1237;
    }

    public static int hashCode(double dbl) {
        long bits = Double.doubleToLongBits(dbl);
        return hashCode(bits);
    }

    public static int hashCode(float flt) {
        return Float.floatToIntBits(flt);
    }

    public static int hashCode(long lng) {
        return (int)(lng ^ lng >>> 32);
    }

    public static String identityToString(Object obj) {
        return obj == null?"":obj.getClass().getName() + "@" + getIdentityHexString(obj);
    }

    public static String getIdentityHexString(Object obj) {
        return Integer.toHexString(System.identityHashCode(obj));
    }

    public static String getDisplayString(Object obj) {
        return obj == null?"":nullSafeToString((Object)obj);
    }

    public static String nullSafeClassName(Object obj) {
        return obj != null?obj.getClass().getName():"null";
    }

    public static String nullSafeToString(Object obj) {
        if(obj == null) {
            return "null";
        } else if(obj instanceof String) {
            return (String)obj;
        } else if(obj instanceof Object[]) {
            return nullSafeToString((Object[])((Object[])((Object[])obj)));
        } else if(obj instanceof boolean[]) {
            return nullSafeToString((boolean[])((boolean[])((boolean[])obj)));
        } else if(obj instanceof byte[]) {
            return nullSafeToString((byte[])((byte[])((byte[])obj)));
        } else if(obj instanceof char[]) {
            return nullSafeToString((char[])((char[])((char[])obj)));
        } else if(obj instanceof double[]) {
            return nullSafeToString((double[])((double[])((double[])obj)));
        } else if(obj instanceof float[]) {
            return nullSafeToString((float[])((float[])((float[])obj)));
        } else if(obj instanceof int[]) {
            return nullSafeToString((int[])((int[])((int[])obj)));
        } else if(obj instanceof long[]) {
            return nullSafeToString((long[])((long[])((long[])obj)));
        } else if(obj instanceof short[]) {
            return nullSafeToString((short[])((short[])((short[])obj)));
        } else {
            String str = obj.toString();
            return str != null?str:"";
        }
    }

    public static String nullSafeToString(Object[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append(String.valueOf(array[i]));
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }

    public static String nullSafeToString(boolean[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append(array[i]);
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }

    public static String nullSafeToString(byte[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append(array[i]);
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }

    public static String nullSafeToString(char[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append("\'").append(array[i]).append("\'");
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }

    public static String nullSafeToString(double[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append(array[i]);
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }

    public static String nullSafeToString(float[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append(array[i]);
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }

    public static String nullSafeToString(int[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append(array[i]);
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }

    public static String nullSafeToString(long[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append(array[i]);
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }

    public static String nullSafeToString(short[] array) {
        if(array == null) {
            return "null";
        } else {
            int length = array.length;
            if(length == 0) {
                return "{}";
            } else {
                StringBuilder sb = new StringBuilder();

                for(int i = 0; i < length; ++i) {
                    if(i == 0) {
                        sb.append("{");
                    } else {
                        sb.append(", ");
                    }

                    sb.append(array[i]);
                }

                sb.append("}");
                return sb.toString();
            }
        }
    }
}
