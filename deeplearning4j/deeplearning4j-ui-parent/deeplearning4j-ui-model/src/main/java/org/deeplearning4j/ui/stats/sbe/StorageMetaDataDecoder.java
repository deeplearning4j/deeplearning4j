/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.StorageMetaDataDecoder"})
@SuppressWarnings("all")
public class StorageMetaDataDecoder {
    public static final int BLOCK_LENGTH = 8;
    public static final int TEMPLATE_ID = 3;
    public static final int SCHEMA_ID = 1;
    public static final int SCHEMA_VERSION = 0;

    private final StorageMetaDataDecoder parentMessage = this;
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

    public StorageMetaDataDecoder wrap(final DirectBuffer buffer, final int offset, final int actingBlockLength,
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

    public static int timeStampId() {
        return 1;
    }

    public static String timeStampMetaAttribute(final MetaAttribute metaAttribute) {
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

    public static long timeStampNullValue() {
        return -9223372036854775808L;
    }

    public static long timeStampMinValue() {
        return -9223372036854775807L;
    }

    public static long timeStampMaxValue() {
        return 9223372036854775807L;
    }

    public long timeStamp() {
        return buffer.getLong(offset + 0, java.nio.ByteOrder.LITTLE_ENDIAN);
    }


    private final ExtraMetaDataBytesDecoder extraMetaDataBytes = new ExtraMetaDataBytesDecoder();

    public static long extraMetaDataBytesDecoderId() {
        return 2;
    }

    public ExtraMetaDataBytesDecoder extraMetaDataBytes() {
        extraMetaDataBytes.wrap(parentMessage, buffer);
        return extraMetaDataBytes;
    }

    public static class ExtraMetaDataBytesDecoder
                    implements Iterable<ExtraMetaDataBytesDecoder>, java.util.Iterator<ExtraMetaDataBytesDecoder> {
        private static final int HEADER_SIZE = 4;
        private final GroupSizeEncodingDecoder dimensions = new GroupSizeEncodingDecoder();
        private StorageMetaDataDecoder parentMessage;
        private DirectBuffer buffer;
        private int blockLength;
        private int actingVersion;
        private int count;
        private int index;
        private int offset;

        public void wrap(final StorageMetaDataDecoder parentMessage, final DirectBuffer buffer) {
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
            return 1;
        }

        public int actingBlockLength() {
            return blockLength;
        }

        public int count() {
            return count;
        }

        public java.util.Iterator<ExtraMetaDataBytesDecoder> iterator() {
            return this;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public boolean hasNext() {
            return (index + 1) < count;
        }

        public ExtraMetaDataBytesDecoder next() {
            if (index + 1 >= count) {
                throw new java.util.NoSuchElementException();
            }

            offset = parentMessage.limit();
            parentMessage.limit(offset + blockLength);
            ++index;

            return this;
        }

        public static int bytesId() {
            return 3;
        }

        public static String bytesMetaAttribute(final MetaAttribute metaAttribute) {
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

        public static byte bytesNullValue() {
            return (byte) -128;
        }

        public static byte bytesMinValue() {
            return (byte) -127;
        }

        public static byte bytesMaxValue() {
            return (byte) 127;
        }

        public byte bytes() {
            return buffer.getByte(offset + 0);
        }


        public String toString() {
            return appendTo(new StringBuilder(100)).toString();
        }

        public StringBuilder appendTo(final StringBuilder builder) {
            builder.append('(');
            //Token{signal=BEGIN_FIELD, name='bytes', description='null', id=3, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            //Token{signal=ENCODING, name='int8', description='null', id=-1, version=0, encodedLength=1, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT8, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
            builder.append("bytes=");
            builder.append(bytes());
            builder.append(')');
            return builder;
        }
    }

    public static int sessionIDId() {
        return 4;
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
        return 5;
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
        return 6;
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

    public static int initTypeClassId() {
        return 7;
    }

    public static String initTypeClassCharacterEncoding() {
        return "UTF-8";
    }

    public static String initTypeClassMetaAttribute(final MetaAttribute metaAttribute) {
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

    public static int initTypeClassHeaderLength() {
        return 4;
    }

    public int initTypeClassLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getInitTypeClass(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getInitTypeClass(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String initTypeClass() {
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

    public static int updateTypeClassId() {
        return 8;
    }

    public static String updateTypeClassCharacterEncoding() {
        return "UTF-8";
    }

    public static String updateTypeClassMetaAttribute(final MetaAttribute metaAttribute) {
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

    public static int updateTypeClassHeaderLength() {
        return 4;
    }

    public int updateTypeClassLength() {
        final int limit = parentMessage.limit();
        return (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
    }

    public int getUpdateTypeClass(final MutableDirectBuffer dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public int getUpdateTypeClass(final byte[] dst, final int dstOffset, final int length) {
        final int headerLength = 4;
        final int limit = parentMessage.limit();
        final int dataLength = (int) (buffer.getInt(limit, java.nio.ByteOrder.LITTLE_ENDIAN) & 0xFFFF_FFFFL);
        final int bytesCopied = Math.min(length, dataLength);
        parentMessage.limit(limit + headerLength + dataLength);
        buffer.getBytes(limit + headerLength, dst, dstOffset, bytesCopied);

        return bytesCopied;
    }

    public String updateTypeClass() {
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
        builder.append("[StorageMetaData](sbeTemplateId=");
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
        //Token{signal=BEGIN_FIELD, name='timeStamp', description='null', id=1, version=0, encodedLength=0, offset=0, componentTokenCount=3, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        //Token{signal=ENCODING, name='int64', description='null', id=-1, version=0, encodedLength=8, offset=0, componentTokenCount=1, encoding=Encoding{presence=REQUIRED, primitiveType=INT64, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("timeStamp=");
        builder.append(timeStamp());
        builder.append('|');
        //Token{signal=BEGIN_GROUP, name='extraMetaDataBytes', description='Extra metadata bytes', id=2, version=0, encodedLength=1, offset=8, componentTokenCount=9, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='null', timeUnit=null, semanticType='null'}}
        builder.append("extraMetaDataBytes=[");
        ExtraMetaDataBytesDecoder extraMetaDataBytes = extraMetaDataBytes();
        if (extraMetaDataBytes.count() > 0) {
            while (extraMetaDataBytes.hasNext()) {
                extraMetaDataBytes.next().appendTo(builder);
                builder.append(',');
            }
            builder.setLength(builder.length() - 1);
        }
        builder.append(']');
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='sessionID', description='null', id=4, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("sessionID=");
        builder.append(sessionID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='typeID', description='null', id=5, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("typeID=");
        builder.append(typeID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='workerID', description='null', id=6, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("workerID=");
        builder.append(workerID());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='initTypeClass', description='null', id=7, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("initTypeClass=");
        builder.append(initTypeClass());
        builder.append('|');
        //Token{signal=BEGIN_VAR_DATA, name='updateTypeClass', description='null', id=8, version=0, encodedLength=0, offset=-1, componentTokenCount=6, encoding=Encoding{presence=REQUIRED, primitiveType=null, byteOrder=LITTLE_ENDIAN, minValue=null, maxValue=null, nullValue=null, constValue=null, characterEncoding='null', epoch='unix', timeUnit=nanosecond, semanticType='null'}}
        builder.append("updateTypeClass=");
        builder.append(updateTypeClass());

        limit(originalLimit);

        return builder;
    }
}
