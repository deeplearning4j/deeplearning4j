package org.datavec.arrow.recordreader;

/**
 * Copyright 2014 Google Inc. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */


import com.google.common.base.Preconditions;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ClosedChannelException;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.SeekableByteChannel;

/**
 * InputStreamSeekableReadableByteChannel is an adaptor from any InputStream which returns true
 * for markSupported() to expose the SeekableReadableByteChannel interface. Each call to
 * position(long) my require skipping over the entire contents of the InputStream, so this
 * adaptor will not be efficient if the underlying InputStream does not support efficient
 * random access.
 */
public class InputStreamSeekableReadableByteChannel implements SeekableByteChannel {
    // Underlying InputStream provided at construction time; must return true for markSupported().
    private final InputStream readStream;

    // Count of total bytes this channel can read.
    private final int maxBytesToRead;

    // Delegate which implements the base ReadableByteChannel (non-seekable) portion of the interface.
    // We require the delegate instead of extending from it because the library class suitable for
    // this use case is a non-public inner class.
    private final ReadableByteChannel readChannelDelegate;

    // Internal accounting of position in the stream.
    private int position;

    /**
     * @param readStream The underlying InputStream providing this channel's data. It must return
     *     true for markSupported(). The position of this channel is defined relative to the
     *     current position of the InputStream; any bytes previously read out of the InputStream
     *     already will be invisible to this channel.
     * @param maxBytesToRead The total number of bytes this channel can read out of the InputStream.
     */
    public InputStreamSeekableReadableByteChannel(InputStream readStream, int maxBytesToRead) {
        Preconditions.checkArgument(readStream != null);
        Preconditions.checkArgument(maxBytesToRead >= 0);

        // We require markSupported() in order to implement seeking; InputStream.mark() allows
        // returning to a position in the stream using reset(). If this is not supported by the
        // InputStream, the user can only obtain a non-seekable ReadableByteChannel through the
        // standard Channels library.
        Preconditions.checkArgument(readStream.markSupported());

        this.readStream = readStream;
        this.maxBytesToRead = maxBytesToRead;

        // Seeking will be relative to the current position in the stream;
        readStream.mark(this.maxBytesToRead);
        this.readChannelDelegate = Channels.newChannel(this.readStream);
        this.position = 0;
    }

    @Override
    public long position()
            throws IOException {
        throwIfNotOpen();
        return position;
    }

    @Override
    public SeekableByteChannel position(long newPosition)
            throws IOException {
        throwIfNotOpen();

        // Validate: 0 <= newPosition < size.
        if ((newPosition != 0) && ((newPosition < 0) || (newPosition >= size()))) {
            throw new IllegalArgumentException(
                    String.format(
                            "Invalid seek offset: position value (%d) must be between 0 and %d",
                            newPosition, size()));
        }
        readStream.reset();
        readStream.skip(newPosition);
        position = (int) newPosition;
        return this;
    }

    @Override
    public long size()
            throws IOException {
        throwIfNotOpen();
        return maxBytesToRead;
    }

    @Override
    public int read(ByteBuffer dst) throws IOException {
        throwIfNotOpen();
        if (dst.remaining() > 0 && position == maxBytesToRead) {
            // Reached our definition of end-of-stream, even if the underlying InputStream has more
            // available.
            return -1;
        } else if (dst.remaining() > maxBytesToRead - position) {
            // Make sure we don't grab more than (maxBytesToRead - position) bytes out of the InputStream
            // if the provided buffer is has larger 'remaning' than maxBytesToRead - position.
            int oldLimit = dst.limit();
            dst.limit(dst.position() + maxBytesToRead - position);
            int numRead = readChannelDelegate.read(dst);
            dst.limit(oldLimit);
            position += numRead;
            return numRead;
        }

        // Normal case: provided buffer 'remaining' is less than or equal to the number of bytes we
        // are allowed to read out of the stream.
        int numRead = readChannelDelegate.read(dst);
        position += numRead;
        return numRead;
    }

    @Override
    public SeekableByteChannel truncate(long size) throws IOException {
        throw new UnsupportedOperationException("Cannot mutate read-only channel");
    }

    @Override
    public int write(ByteBuffer src) throws IOException {
        throw new UnsupportedOperationException("Cannot mutate read-only channel");
    }

    @Override
    public void close() throws IOException {
        readChannelDelegate.close();
    }

    @Override
    public boolean isOpen() {
        return readChannelDelegate.isOpen();
    }

    /**
     * Throws if this channel is not currently open.
     */
    private void throwIfNotOpen()
            throws IOException {
        if (!isOpen()) {
            throw new ClosedChannelException();
        }
    }
}