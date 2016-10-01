/* Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.optimize.listeners.stats.sbe.init;

import org.agrona.MutableDirectBuffer;
import org.agrona.DirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.optimize.listeners.stats.sbe.init.StaticInfoEncoder"})
@SuppressWarnings("all")
public class StaticInfoEncoder
{
    public static final int BLOCK_LENGTH = 24;
    public static final int TEMPLATE_ID = 1;
    public static final int SCHEMA_ID = 1;
    public static final int SCHEMA_VERSION = 0;

    private final StaticInfoEncoder parentMessage = this;
    private MutableDirectBuffer buffer;
    protected int offset;
    protected int limit;
    protected int actingBlockLength;
    protected int actingVersion;

    public int sbeBlockLength()
    {
        return BLOCK_LENGTH;
    }

    public int sbeTemplateId()
    {
        return TEMPLATE_ID;
    }

    public int sbeSchemaId()
    {
        return SCHEMA_ID;
    }

    public int sbeSchemaVersion()
    {
        return SCHEMA_VERSION;
    }

    public String sbeSemanticType()
    {
        return "";
    }

    public int offset()
    {
        return offset;
    }

    public StaticInfoEncoder wrap(final MutableDirectBuffer buffer, final int offset)
    {
        this.buffer = buffer;
        this.offset = offset;
        limit(offset + BLOCK_LENGTH);

        return this;
    }

    public int encodedLength()
    {
        return limit - offset;
    }

    public int limit()
    {
        return limit;
    }

    public void limit(final int limit)
    {
        this.limit = limit;
    }

    public static long timeNullValue()
    {
        return -9223372036854775808L;
    }

    public static long timeMinValue()
    {
        return -9223372036854775807L;
    }

    public static long timeMaxValue()
    {
        return 9223372036854775807L;
    }

    public StaticInfoEncoder time(final long value)
    {
        buffer.putLong(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    private final InitFieldsPresentEncoder fieldsPresent = new InitFieldsPresentEncoder();

    public InitFieldsPresentEncoder fieldsPresent()
    {
        fieldsPresent.wrap(buffer, offset + 8);
        return fieldsPresent;
    }

    public static int hwJvmProcessorsNullValue()
    {
        return 65535;
    }

    public static int hwJvmProcessorsMinValue()
    {
        return 0;
    }

    public static int hwJvmProcessorsMaxValue()
    {
        return 65534;
    }

    public StaticInfoEncoder hwJvmProcessors(final int value)
    {
        buffer.putShort(offset + 9, (short)value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static short hwNumDevicesNullValue()
    {
        return (short)255;
    }

    public static short hwNumDevicesMinValue()
    {
        return (short)0;
    }

    public static short hwNumDevicesMaxValue()
    {
        return (short)254;
    }

    public StaticInfoEncoder hwNumDevices(final short value)
    {
        buffer.putByte(offset + 11, (byte)value);
        return this;
    }


    public static int modelNumLayersNullValue()
    {
        return -2147483648;
    }

    public static int modelNumLayersMinValue()
    {
        return -2147483647;
    }

    public static int modelNumLayersMaxValue()
    {
        return 2147483647;
    }

    public StaticInfoEncoder modelNumLayers(final int value)
    {
        buffer.putInt(offset + 12, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    public static long modelNumParamsNullValue()
    {
        return -9223372036854775808L;
    }

    public static long modelNumParamsMinValue()
    {
        return -9223372036854775807L;
    }

    public static long modelNumParamsMaxValue()
    {
        return 9223372036854775807L;
    }

    public StaticInfoEncoder modelNumParams(final long value)
    {
        buffer.putLong(offset + 16, value, java.nio.ByteOrder.LITTLE_ENDIAN);
        return this;
    }


    private final HwDeviceInfoGroupEncoder hwDeviceInfoGroup = new HwDeviceInfoGroupEncoder();

    public static long hwDeviceInfoGroupId()
    {
        return 7;
    }

    public HwDeviceInfoGroupEncoder hwDeviceInfoGroupCount(final int count)
    {
        hwDeviceInfoGroup.wrap(parentMessage, buffer, count);
        return hwDeviceInfoGroup;
    }

    public static class HwDeviceInfoGroupEncoder
    {
        private static final int HEADER_SIZE = 4;
        private final InitGroupSizeEncodingEncoder dimensions = new InitGroupSizeEncodingEncoder();
        private StaticInfoEncoder parentMessage;
        private MutableDirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(
            final StaticInfoEncoder parentMessage, final MutableDirectBuffer buffer, final int count)
        {
            if (count < 0 || count > 65534)
            {
                throw new IllegalArgumentException("count outside allowed range: count=" + count);
            }

            this.parentMessage = parentMessage;
            this.buffer = buffer;
            actingVersion = SCHEMA_VERSION;
            dimensions.wrap(buffer, parentMessage.limit());
            dimensions.blockLength((int)8);
            dimensions.numInGroup((int)count);
            index = -1;
            this.count = count;
            blockLength = 8;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize()
        {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength()
        {
            return 8;
        }

        public HwDeviceInfoGroupEncoder next()
        {
            if (index + 1 >= count)
            {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static long deviceMemoryMaxNullValue()
        {
            return -9223372036854775808L;
        }

        public static long deviceMemoryMaxMinValue()
        {
            return -9223372036854775807L;
        }

        public static long deviceMemoryMaxMaxValue()
        {
            return 9223372036854775807L;
        }

        public HwDeviceInfoGroupEncoder deviceMemoryMax(final long value)
        {
            buffer.putLong(offset + 0, value, java.nio.ByteOrder.LITTLE_ENDIAN);
            return this;
        }


        public static int deviceDescriptionId()
        {
            return 50;
        }

        public static String deviceDescriptionCharacterEncoding()
        {
            return "UTF-8";
        }

        public static String deviceDescriptionMetaAttribute(final MetaAttribute metaAttribute)
        {
            switch (metaAttribute)
            {
                case EPOCH: return "unix";
                case TIME_UNIT: return "nanosecond";
                case SEMANTIC_TYPE: return "";
            }

            return "";
        }

        public static int deviceDescriptionHeaderLength()
        {
            return 4;
        }

        public HwDeviceInfoGroupEncoder putDeviceDescription(final DirectBuffer src, final int srcOffset, final int length)
        {
            if (length > 1073741824)
            {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public HwDeviceInfoGroupEncoder putDeviceDescription(final byte[] src, final int srcOffset, final int length)
        {
            if (length > 1073741824)
            {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, src, srcOffset, length);

            return this;
        }

        public HwDeviceInfoGroupEncoder deviceDescription(final String value)
        {
            final byte[] bytes;
            try
            {
                bytes = value.getBytes("UTF-8");
            }
            catch (final java.io.UnsupportedEncodingException ex)
            {
                throw new RuntimeException(ex);
            }

            final int length = bytes.length;
            if (length > 1073741824)
            {
                throw new IllegalArgumentException("length > max value for type: " + length);
            }

            final int headerLength = 4;
            final int limit = parentMessage.limit();
            parentMessage.limit(limit + headerLength + length);
            buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
            buffer.putBytes(limit + headerLength, bytes, 0, length);

            return this;
        }
    }

    public static int swArchId()
    {
        return 100;
    }

    public static String swArchCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String swArchMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int swArchHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putSwArch(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putSwArch(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder swArch(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int swOsNameId()
    {
        return 101;
    }

    public static String swOsNameCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String swOsNameMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int swOsNameHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putSwOsName(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putSwOsName(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder swOsName(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int swJvmNameId()
    {
        return 102;
    }

    public static String swJvmNameCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String swJvmNameMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int swJvmNameHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putSwJvmName(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putSwJvmName(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder swJvmName(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int swJvmVersionId()
    {
        return 103;
    }

    public static String swJvmVersionCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String swJvmVersionMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int swJvmVersionHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putSwJvmVersion(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putSwJvmVersion(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder swJvmVersion(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int swJvmSpecVersionId()
    {
        return 104;
    }

    public static String swJvmSpecVersionCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String swJvmSpecVersionMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int swJvmSpecVersionHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putSwJvmSpecVersion(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putSwJvmSpecVersion(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder swJvmSpecVersion(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int swNd4jBackendClassId()
    {
        return 105;
    }

    public static String swNd4jBackendClassCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String swNd4jBackendClassMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int swNd4jBackendClassHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putSwNd4jBackendClass(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putSwNd4jBackendClass(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder swNd4jBackendClass(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int swNd4jDataTypeNameId()
    {
        return 106;
    }

    public static String swNd4jDataTypeNameCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String swNd4jDataTypeNameMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int swNd4jDataTypeNameHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putSwNd4jDataTypeName(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putSwNd4jDataTypeName(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder swNd4jDataTypeName(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int modelConfigClassNameId()
    {
        return 200;
    }

    public static String modelConfigClassNameCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String modelConfigClassNameMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int modelConfigClassNameHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putModelConfigClassName(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putModelConfigClassName(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder modelConfigClassName(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }

    public static int modelConfigJsonId()
    {
        return 201;
    }

    public static String modelConfigJsonCharacterEncoding()
    {
        return "UTF-8";
    }

    public static String modelConfigJsonMetaAttribute(final MetaAttribute metaAttribute)
    {
        switch (metaAttribute)
        {
            case EPOCH: return "unix";
            case TIME_UNIT: return "nanosecond";
            case SEMANTIC_TYPE: return "";
        }

        return "";
    }

    public static int modelConfigJsonHeaderLength()
    {
        return 4;
    }

    public StaticInfoEncoder putModelConfigJson(final DirectBuffer src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder putModelConfigJson(final byte[] src, final int srcOffset, final int length)
    {
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, src, srcOffset, length);

        return this;
    }

    public StaticInfoEncoder modelConfigJson(final String value)
    {
        final byte[] bytes;
        try
        {
            bytes = value.getBytes("UTF-8");
        }
        catch (final java.io.UnsupportedEncodingException ex)
        {
            throw new RuntimeException(ex);
        }

        final int length = bytes.length;
        if (length > 1073741824)
        {
            throw new IllegalArgumentException("length > max value for type: " + length);
        }

        final int headerLength = 4;
        final int limit = parentMessage.limit();
        parentMessage.limit(limit + headerLength + length);
        buffer.putInt(limit, (int)length, java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.putBytes(limit + headerLength, bytes, 0, length);

        return this;
    }
    public String toString()
    {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder)
    {
        StaticInfoDecoder writer = new StaticInfoDecoder();
        writer.wrap(buffer, offset, BLOCK_LENGTH, SCHEMA_VERSION);

        return writer.appendTo(builder);
    }
}
