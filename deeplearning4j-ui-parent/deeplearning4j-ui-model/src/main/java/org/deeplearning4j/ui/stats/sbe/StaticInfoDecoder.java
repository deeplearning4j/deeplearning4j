/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.StaticInfoDecoder"})
@SuppressWarnings("all")
public class StaticInfoDecoder {
    public static final int BLOCK_LENGTH = 40;
    public static final int TEMPLATE_ID = 1;
    public static final int SCHEMA_ID = 1;
    public static final int SCHEMA_VERSION = 0;

    private final StaticInfoDecoder parentMessage = this;
    private DirectBuffer buffer;
    protected int offset;
    protected int limit;
    protected int actingBlockLength;
    protected int actingVersion;

    public int sbeBlockLength() {
        return BLOCK_LENGTH;
    }

    public int sbeTemplateId() {
        return TEMPLATE_ID;
    }

    public int sbeSchemaId() {
        return SCHEMA_ID;
    }

    public int sbeSchemaVersion() {
        return SCHEMA_VERSION;
    }

    public String sbeSemanticType() {
        return "";
    }

    public int offset() {
        return offset;
    }

    public StaticInfoDecoder wrap(final DirectBuffer buffer, final int offset, final int actingBlockLength,
                    final int actingVersion) {
        this.buffer = buffer;
        this.offset = offset;
        this.actingBlockLength = actingBlockLength;
        this.actingVersion = actingVersion;
        limit(offset + actingBlockLength);

        return this;
    }

    public int encodedLength() {
        return limit - offset;
    }

    public int limit() {
        return limit;
    }

    public void limit(final int limit) {
        this.limit = limit;
    }

    public static int timeId() {
        return 1;
    }

    public static String timeMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static long timeNullValue() {
        return -9223372036854775808L;
    }

    public static long timeMinValue() {
        return -9223372036854775807L;
    }

    public static long timeMaxValue() {
        return 9223372036854775807L;
    }

    public long time() {
        return buffer.getLong(offset + 0, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    public static int fieldsPresentId() {
        return 2;
    }

    public static String fieldsPresentMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    private final InitFieldsPresentDecoder fieldsPresent = new InitFieldsPresentDecoder();

    public InitFieldsPresentDecoder fieldsPresent() {
        fieldsPresent.wrap(buffer, offset + 8);
        return fieldsPresent;
    }

    public static int hwJvmProcessorsId() {
        return 3;
    }

    public static String hwJvmProcessorsMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int hwJvmProcessorsNullValue() {
        return 65535;
    }

    public static int hwJvmProcessorsMinValue() {
        return 0;
    }

    public static int hwJvmProcessorsMaxValue() {
        return 65534;
    }

    public int hwJvmProcessors() {
        return (buffer.getShort(offset + 9, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF);
    }


    public static int hwNumDevicesId() {
        return 4;
    }

    public static String hwNumDevicesMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static short hwNumDevicesNullValue() {
        return (short) 255;
    }

    public static short hwNumDevicesMinValue() {
        return (short) 0;
    }

    public static short hwNumDevicesMaxValue() {
        return (short) 254;
    }

    public short hwNumDevices() {
        return ((short) (buffer.getByte(offset + 11) & 0xFF));
    }


    public static int hwJvmMaxMemoryId() {
        return 5;
    }

    public static String hwJvmMaxMemoryMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static long hwJvmMaxMemoryNullValue() {
        return -9223372036854775808L;
    }

    public static long hwJvmMaxMemoryMinValue() {
        return -9223372036854775807L;
    }

    public static long hwJvmMaxMemoryMaxValue() {
        return 9223372036854775807L;
    }

    public long hwJvmMaxMemory() {
        return buffer.getLong(offset + 12, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    public static int hwOffheapMaxMemoryId() {
        return 6;
    }

    public static String hwOffheapMaxMemoryMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static long hwOffheapMaxMemoryNullValue() {
        return -9223372036854775808L;
    }

    public static long hwOffheapMaxMemoryMinValue() {
        return -9223372036854775807L;
    }

    public static long hwOffheapMaxMemoryMaxValue() {
        return 9223372036854775807L;
    }

    public long hwOffheapMaxMemory() {
        return buffer.getLong(offset + 20, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    public static int modelNumLayersId() {
        return 7;
    }

    public static String modelNumLayersMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int modelNumLayersNullValue() {
        return -2147483648;
    }

    public static int modelNumLayersMinValue() {
        return -2147483647;
    }

    public static int modelNumLayersMaxValue() {
        return 2147483647;
    }

    public int modelNumLayers() {
        return buffer.getInt(offset + 28, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    public static int modelNumParamsId() {
        return 8;
    }

    public static String modelNumParamsMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static long modelNumParamsNullValue() {
        return -9223372036854775808L;
    }

    public static long modelNumParamsMinValue() {
        return -9223372036854775807L;
    }

    public static long modelNumParamsMaxValue() {
        return 9223372036854775807L;
    }

    public long modelNumParams() {
        return buffer.getLong(offset + 32, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    private final HwDeviceInfoGroupDecoder hwDeviceInfoGroup = new HwDeviceInfoGroupDecoder();

    public static long hwDeviceInfoGroupDecoderId() {
        return 9;
    }

    public HwDeviceInfoGroupDecoder hwDeviceInfoGroup() {
        hwDeviceInfoGroup.wrap(parentMessage, buffer);
        return hwDeviceInfoGroup;
    }

    public static class HwDeviceInfoGroupDecoder
                    implements Iterable<HwDeviceInfoGroupDecoder>, java.util.Iterator<HwDeviceInfoGroupDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private StaticInfoDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final StaticInfoDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 8;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<HwDeviceInfoGroupDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public HwDeviceInfoGroupDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int deviceMemoryMaxId() {
            return 10;
        }

        public static String deviceMemoryMaxMetaAttribute(final MetaAttribute metaAttribute) {
            switch (metaAttribute) {
                case EPOCH:
                    return "unix";
                case TIME_UNIT:
                    return "nanosecond";
                case SEMANTIC_TYPE:
                    return "";
            }

            return "";
        }

        public static long deviceMemoryMaxNullValue() {
            return -9223372036854775808L;
        }

        public static long deviceMemoryMaxMinValue() {
            return -9223372036854775807L;
        }

        public static long deviceMemoryMaxMaxValue() {
            return 9223372036854775807L;
        }

        public long deviceMemoryMax() {
            return buffer.getLong(offset + 0, java.nio.ByteOrder.LITTLE_ENDIAN);
        }


        public static int deviceDescriptionId() {
            return 50;
        }

        public static String deviceDescriptionCharacterEncoding() {
            return "UTF-8";
        }

        public static String deviceDescriptionMetaAttribute(final MetaAttribute metaAttribute) {
            switch (metaAttribute) {
                case EPOCH:
                    return "unix";
                case TIME_UNIT:
                    return "nanosecond";
                case SEMANTIC_TYPE:
                    return "";
            }

            return "";
        }

        public static int deviceDescriptionHeaderLength() {
            return 4;
        }

        public int deviceDescriptionLength() {
            final int limit = parentMessage.limit();
            return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        }

        public int getDeviceDescription(final MutableDirectBuffer dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public int getDeviceDescription(final byte[] dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public String deviceDescription() {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            parentMessage.limit(limit + headerLength + dataLength);
            final byte[] tmp = new byte[dataLength];
            buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

            final String value;
            try {
                value = new String(tmp, "UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            return value;
        }

        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_FIELD, name='deviceMemoryMax', description='null', id=10, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("deviceMemoryMax=");
            builder.append(deviceMemoryMax());
            builder.append('|');
            //Token{signal=BEGIN_VAR_DATA, name='deviceDescription', description='null', id=50, version=0, encodedLength=0, offset=8, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("deviceDescription=");
            builder.append(deviceDescription());
            builder.append(')');
            return builder;
        }
    }

    private final SwEnvironmentInfoDecoder swEnvironmentInfo = new SwEnvironmentInfoDecoder();

    public static long swEnvironmentInfoDecoderId() {
        return 12;
    }

    public SwEnvironmentInfoDecoder swEnvironmentInfo() {
        swEnvironmentInfo.wrap(parentMessage, buffer);
        return swEnvironmentInfo;
    }

    public static class SwEnvironmentInfoDecoder
                    implements Iterable<SwEnvironmentInfoDecoder>, java.util.Iterator<SwEnvironmentInfoDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private StaticInfoDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final StaticInfoDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 0;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<SwEnvironmentInfoDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public SwEnvironmentInfoDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int envKeyId() {
            return 51;
        }

        public static String envKeyCharacterEncoding() {
            return "UTF-8";
        }

        public static String envKeyMetaAttribute(final MetaAttribute metaAttribute) {
            switch (metaAttribute) {
                case EPOCH:
                    return "unix";
                case TIME_UNIT:
                    return "nanosecond";
                case SEMANTIC_TYPE:
                    return "";
            }

            return "";
        }

        public static int envKeyHeaderLength() {
            return 4;
        }

        public int envKeyLength() {
            final int limit = parentMessage.limit();
            return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        }

        public int getEnvKey(final MutableDirectBuffer dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public int getEnvKey(final byte[] dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public String envKey() {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            parentMessage.limit(limit + headerLength + dataLength);
            final byte[] tmp = new byte[dataLength];
            buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

            final String value;
            try {
                value = new String(tmp, "UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            return value;
        }

        public static int envValueId() {
            return 52;
        }

        public static String envValueCharacterEncoding() {
            return "UTF-8";
        }

        public static String envValueMetaAttribute(final MetaAttribute metaAttribute) {
            switch (metaAttribute) {
                case EPOCH:
                    return "unix";
                case TIME_UNIT:
                    return "nanosecond";
                case SEMANTIC_TYPE:
                    return "";
            }

            return "";
        }

        public static int envValueHeaderLength() {
            return 4;
        }

        public int envValueLength() {
            final int limit = parentMessage.limit();
            return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        }

        public int getEnvValue(final MutableDirectBuffer dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public int getEnvValue(final byte[] dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public String envValue() {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            parentMessage.limit(limit + headerLength + dataLength);
            final byte[] tmp = new byte[dataLength];
            buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

            final String value;
            try {
                value = new String(tmp, "UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            return value;
        }

        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_VAR_DATA, name='envKey', description='null', id=51, version=0, encodedLength=0, offset=0, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("envKey=");
            builder.append(envKey());
            builder.append('|');
            //Token{signal=BEGIN_VAR_DATA, name='envValue', description='null', id=52, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("envValue=");
            builder.append(envValue());
            builder.append(')');
            return builder;
        }
    }

    private final ModelParamNamesDecoder modelParamNames = new ModelParamNamesDecoder();

    public static long modelParamNamesDecoderId() {
        return 11;
    }

    public ModelParamNamesDecoder modelParamNames() {
        modelParamNames.wrap(parentMessage, buffer);
        return modelParamNames;
    }

    public static class ModelParamNamesDecoder
                    implements Iterable<ModelParamNamesDecoder>, java.util.Iterator<ModelParamNamesDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private StaticInfoDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final StaticInfoDecoder parentMessage, final DirectBuffer buffer) {
            this.parentMessage = parentMessage;
            this.buffer = buffer;
            dimensions.wrap(buffer, parentMessage.limit());
            blockLength = dimensions.blockLength();
            count = dimensions.numInGroup();
            index = -1;
            parentMessage.limit(parentMessage.limit() + HEADER_SIZE);
        }

        public static int sbeHeaderSize() {
            return HEADER_SIZE;
        }

        public static int sbeBlockLength() {
            return 0;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<ModelParamNamesDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public ModelParamNamesDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int modelParamNamesId() {
            return 53;
        }

        public static String modelParamNamesCharacterEncoding() {
            return "UTF-8";
        }

        public static String modelParamNamesMetaAttribute(final MetaAttribute metaAttribute) {
            switch (metaAttribute) {
                case EPOCH:
                    return "unix";
                case TIME_UNIT:
                    return "nanosecond";
                case SEMANTIC_TYPE:
                    return "";
            }

            return "";
        }

        public static int modelParamNamesHeaderLength() {
            return 4;
        }

        public int modelParamNamesLength() {
            final int limit = parentMessage.limit();
            return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        }

        public int getModelParamNames(final MutableDirectBuffer dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public int getModelParamNames(final byte[] dst, final int dstOffset, final int length) {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            final int bytesCopied = Math.min(length, dataLength);
            parentMessage.limit(limit + headerLength + dataLength);
            buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

            return bytesCopied;
        }

        public String modelParamNames() {
            final int headerLength = 4;
            final int limit = parentMessage.limit();
            final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
            parentMessage.limit(limit + headerLength + dataLength);
            final byte[] tmp = new byte[dataLength];
            buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

            final String value;
            try {
                value = new String(tmp, "UTF-8");
            } catch (final java.io.UnsupportedEncodingException ex) {
                throw new RuntimeException(ex);
            }

            return value;
        }

        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_VAR_DATA, name='modelParamNames', description='null', id=53, version=0, encodedLength=0, offset=0, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("modelParamNames=");
            builder.append(modelParamNames());
            builder.append(')');
            return builder;
        }
    }

    public static int sessionIDId() {
        return 100;
    }

    public static String sessionIDCharacterEncoding() {
        return "UTF-8";
    }

    public static String sessionIDMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int sessionIDHeaderLength() {
        return 4;
    }

    public int sessionIDLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSessionID(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSessionID(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String sessionID() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int typeIDId() {
        return 101;
    }

    public static String typeIDCharacterEncoding() {
        return "UTF-8";
    }

    public static String typeIDMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int typeIDHeaderLength() {
        return 4;
    }

    public int typeIDLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getTypeID(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getTypeID(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String typeID() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int workerIDId() {
        return 102;
    }

    public static String workerIDCharacterEncoding() {
        return "UTF-8";
    }

    public static String workerIDMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int workerIDHeaderLength() {
        return 4;
    }

    public int workerIDLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getWorkerID(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getWorkerID(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String workerID() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swArchId() {
        return 201;
    }

    public static String swArchCharacterEncoding() {
        return "UTF-8";
    }

    public static String swArchMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swArchHeaderLength() {
        return 4;
    }

    public int swArchLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwArch(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwArch(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swArch() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swOsNameId() {
        return 202;
    }

    public static String swOsNameCharacterEncoding() {
        return "UTF-8";
    }

    public static String swOsNameMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swOsNameHeaderLength() {
        return 4;
    }

    public int swOsNameLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwOsName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwOsName(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swOsName() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swJvmNameId() {
        return 203;
    }

    public static String swJvmNameCharacterEncoding() {
        return "UTF-8";
    }

    public static String swJvmNameMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swJvmNameHeaderLength() {
        return 4;
    }

    public int swJvmNameLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwJvmName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwJvmName(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swJvmName() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swJvmVersionId() {
        return 204;
    }

    public static String swJvmVersionCharacterEncoding() {
        return "UTF-8";
    }

    public static String swJvmVersionMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swJvmVersionHeaderLength() {
        return 4;
    }

    public int swJvmVersionLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwJvmVersion(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwJvmVersion(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swJvmVersion() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swJvmSpecVersionId() {
        return 205;
    }

    public static String swJvmSpecVersionCharacterEncoding() {
        return "UTF-8";
    }

    public static String swJvmSpecVersionMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swJvmSpecVersionHeaderLength() {
        return 4;
    }

    public int swJvmSpecVersionLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwJvmSpecVersion(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwJvmSpecVersion(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swJvmSpecVersion() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swNd4jBackendClassId() {
        return 206;
    }

    public static String swNd4jBackendClassCharacterEncoding() {
        return "UTF-8";
    }

    public static String swNd4jBackendClassMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swNd4jBackendClassHeaderLength() {
        return 4;
    }

    public int swNd4jBackendClassLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwNd4jBackendClass(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwNd4jBackendClass(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swNd4jBackendClass() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swNd4jDataTypeNameId() {
        return 207;
    }

    public static String swNd4jDataTypeNameCharacterEncoding() {
        return "UTF-8";
    }

    public static String swNd4jDataTypeNameMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swNd4jDataTypeNameHeaderLength() {
        return 4;
    }

    public int swNd4jDataTypeNameLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwNd4jDataTypeName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwNd4jDataTypeName(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swNd4jDataTypeName() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swHostNameId() {
        return 208;
    }

    public static String swHostNameCharacterEncoding() {
        return "UTF-8";
    }

    public static String swHostNameMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swHostNameHeaderLength() {
        return 4;
    }

    public int swHostNameLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwHostName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwHostName(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swHostName() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int swJvmUIDId() {
        return 209;
    }

    public static String swJvmUIDCharacterEncoding() {
        return "UTF-8";
    }

    public static String swJvmUIDMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int swJvmUIDHeaderLength() {
        return 4;
    }

    public int swJvmUIDLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getSwJvmUID(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getSwJvmUID(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String swJvmUID() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int hwHardwareUIDId() {
        return 300;
    }

    public static String hwHardwareUIDCharacterEncoding() {
        return "UTF-8";
    }

    public static String hwHardwareUIDMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int hwHardwareUIDHeaderLength() {
        return 4;
    }

    public int hwHardwareUIDLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getHwHardwareUID(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getHwHardwareUID(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String hwHardwareUID() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int modelConfigClassNameId() {
        return 400;
    }

    public static String modelConfigClassNameCharacterEncoding() {
        return "UTF-8";
    }

    public static String modelConfigClassNameMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int modelConfigClassNameHeaderLength() {
        return 4;
    }

    public int modelConfigClassNameLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getModelConfigClassName(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getModelConfigClassName(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String modelConfigClassName() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public static int modelConfigJsonId() {
        return 401;
    }

    public static String modelConfigJsonCharacterEncoding() {
        return "UTF-8";
    }

    public static String modelConfigJsonMetaAttribute(final MetaAttribute metaAttribute) {
        switch (metaAttribute) {
            case EPOCH:
                return "unix";
            case TIME_UNIT:
                return "nanosecond";
            case SEMANTIC_TYPE:
                return "";
        }

        return "";
    }

    public static int modelConfigJsonHeaderLength() {
        return 4;
    }

    public int modelConfigJsonLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getModelConfigJson(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getModelConfigJson(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String modelConfigJson() {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        parentMessage.limit(limit + headerLength + dataLength);
        final byte[] tmp = new byte[dataLength];
        buffer.getBytes(limit + headerLength, tmp, 0, dataLength);

        final String value;
        try {
            value = new String(tmp, "UTF-8");
        } catch (final java.io.UnsupportedEncodingException ex) {
            throw new RuntimeException(ex);
        }

        return value;
    }

    public String toString() {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder) {
        final int originalLimit = limit();
        limit(offset + actingBlockLength);
        builder.append("[StaticInfo](sbeTemplateId=");
        builder.append(TEMPLATE_ID);
        builder.append("|sbeSchemaId=");
        builder.append(SCHEMA_ID);
        builder.append("|sbeSchemaVersion=");
        if (actingVersion != SCHEMA_VERSION) {
            builder.append(actingVersion);
            builder.append('/');
        }
        builder.append(SCHEMA_VERSION);
        builder.append("|sbeBlockLength=");
        if (actingBlockLength != BLOCK_LENGTH) {
            builder.append(actingBlockLength);
            builder.append('/');
        }
        builder.append(BLOCK_LENGTH);
        builder.append("):");
        //Token{signal=BEGIN_FIELD, name='time', description='null', id=1, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("time=");
        builder.append(time());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='fieldsPresent', description='null', id=2, version=0, encodedLength=0, offset=8, componentTokenCount=7, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=BEGIN_SET, name='InitFieldsPresent', description='null', id=-1, version=0, encodedLength=1, offset=8, componentTokenCount=5, encoding=Encoding{presence=REQUIRED, primitiveType=UINT8, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='FieldsPresent'}}
        builder.append("fieldsPresent=");
        builder.append(fieldsPresent());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='hwJvmProcessors', description='null', id=3, version=0, encodedLength=0, offset=9, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='uint16', description='null', id=-1, version=0, encodedLength=2, offset=9, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=UINT16, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("hwJvmProcessors=");
        builder.append(hwJvmProcessors());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='hwNumDevices', description='null', id=4, version=0, encodedLength=0, offset=11, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='uint8', description='null', id=-1, version=0, encodedLength=1, offset=11, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=UINT8, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("hwNumDevices=");
        builder.append(hwNumDevices());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='hwJvmMaxMemory', description='null', id=5, version=0, encodedLength=0, offset=12, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=12, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("hwJvmMaxMemory=");
        builder.append(hwJvmMaxMemory());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='hwOffheapMaxMemory', description='null', id=6, version=0, encodedLength=0, offset=20, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=20, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("hwOffheapMaxMemory=");
        builder.append(hwOffheapMaxMemory());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='modelNumLayers', description='null', id=7, version=0, encodedLength=0, offset=28, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int32', description='null', id=-1, version=0, encodedLength=4, offset=28, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT32, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("modelNumLayers=");
        builder.append(modelNumLayers());
        builder.append('|');
        //Token{signal=BEGIN_FIELD, name='modelNumParams', description='null', id=8, version=0, encodedLength=0, offset=32, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=32, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("modelNumParams=");
        builder.append(modelNumParams());
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='hwDeviceInfoGroup', description='null', id=9, version=0, encodedLength=8, offset=40, componentTokenCount=15, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("hwDeviceInfoGroup=[");
        HwDeviceInfoGroupDecoder hwDeviceInfoGroup = hwDeviceInfoGroup();
        if (hwDeviceInfoGroup.count() > 0) {
            while (hwDeviceInfoGroup.hasNext()) {
                hwDeviceInfoGroup.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='swEnvironmentInfo', description='null', id=12, version=0, encodedLength=0, offset=-1, componentTokenCount=18, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("swEnvironmentInfo=[");
        SwEnvironmentInfoDecoder swEnvironmentInfo = swEnvironmentInfo();
        if (swEnvironmentInfo.count() > 0) {
            while (swEnvironmentInfo.hasNext()) {
                swEnvironmentInfo.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='modelParamNames', description='null', id=11, version=0, encodedLength=0, offset=-1, componentTokenCount=12, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("modelParamNames=[");
        ModelParamNamesDecoder modelParamNames = modelParamNames();
        if (modelParamNames.count() > 0) {
            while (modelParamNames.hasNext()) {
                modelParamNames.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='sessionID', description='null', id=100, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("sessionID=");
        builder.append(sessionID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='typeID', description='null', id=101, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("typeID=");
        builder.append(typeID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='workerID', description='null', id=102, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("workerID=");
        builder.append(workerID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swArch', description='null', id=201, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swArch=");
        builder.append(swArch());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swOsName', description='null', id=202, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swOsName=");
        builder.append(swOsName());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swJvmName', description='null', id=203, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swJvmName=");
        builder.append(swJvmName());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swJvmVersion', description='null', id=204, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swJvmVersion=");
        builder.append(swJvmVersion());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swJvmSpecVersion', description='null', id=205, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swJvmSpecVersion=");
        builder.append(swJvmSpecVersion());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swNd4jBackendClass', description='null', id=206, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swNd4jBackendClass=");
        builder.append(swNd4jBackendClass());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swNd4jDataTypeName', description='null', id=207, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swNd4jDataTypeName=");
        builder.append(swNd4jDataTypeName());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swHostName', description='null', id=208, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swHostName=");
        builder.append(swHostName());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='swJvmUID', description='null', id=209, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("swJvmUID=");
        builder.append(swJvmUID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='hwHardwareUID', description='null', id=300, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("hwHardwareUID=");
        builder.append(hwHardwareUID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='modelConfigClassName', description='null', id=400, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("modelConfigClassName=");
        builder.append(modelConfigClassName());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='modelConfigJson', description='null', id=401, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("modelConfigJson=");
        builder.append(modelConfigJson());

        limit(originalLimit);

        return builder;
    }
}
