/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.codec.reader;

import org.apache.commons.compress.utils.IOUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.ImageLoader;
import org.jcodec.api.FrameGrab;
import org.jcodec.api.JCodecException;
import org.jcodec.common.ByteBufferSeekableByteChannel;
import org.jcodec.common.NIOUtils;
import org.jcodec.common.SeekableByteChannel;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Codec record reader for parsing:
 * H.264 ( AVC ) Main profile decoder	MP3 decoder/encoder
 Apple ProRes decoder and encoder	AAC encoder
 H264 Baseline profile encoder
 Matroska ( MKV ) demuxer and muxer
 MP4 ( ISO BMF, QuickTime ) demuxer/muxer and tools
 MPEG 1/2 decoder ( supports interlace )
 MPEG PS/TS demuxer
 Java player applet
 VP8 encoder
 MXF demuxer

 Credit to jcodec for the underlying parser
 *
 * @author Adam Gibson
 */
public class CodecRecordReader extends BaseCodecRecordReader {

    private ImageLoader imageLoader;

    @Override
    public void setConf(Configuration conf) {
        super.setConf(conf);
        imageLoader = new ImageLoader(rows, cols);
    }

    @Override
    protected List<List<Writable>> loadData(File file, InputStream inputStream) throws IOException {
        SeekableByteChannel seekableByteChannel;
        if (inputStream != null) {
            //Reading video from DataInputStream: Need data from this stream in a SeekableByteChannel
            //Approach used here: load entire video into memory -> ByteBufferSeekableByteChanel
            byte[] data = IOUtils.toByteArray(inputStream);
            ByteBuffer bb = ByteBuffer.wrap(data);
            seekableByteChannel = new FixedByteBufferSeekableByteChannel(bb);
        } else {
            seekableByteChannel = NIOUtils.readableFileChannel(file);
        }

        List<List<Writable>> record = new ArrayList<>();

        if (numFrames >= 1) {
            FrameGrab fg;
            try {
                fg = new FrameGrab(seekableByteChannel);
                if (startFrame != 0)
                    fg.seekToFramePrecise(startFrame);
            } catch (JCodecException e) {
                throw new RuntimeException(e);
            }

            for (int i = startFrame; i < startFrame + numFrames; i++) {
                try {
                    BufferedImage grab = fg.getFrame();
                    if (ravel)
                        record.add(RecordConverter.toRecord(imageLoader.toRaveledTensor(grab)));
                    else
                        record.add(RecordConverter.toRecord(imageLoader.asRowVector(grab)));

                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        } else {
            if (framesPerSecond < 1)
                throw new IllegalStateException("No frames or frame time intervals specified");


            else {
                for (double i = 0; i < videoLength; i += framesPerSecond) {
                    try {
                        BufferedImage grab = FrameGrab.getFrame(seekableByteChannel, i);
                        if (ravel)
                            record.add(RecordConverter.toRecord(imageLoader.toRaveledTensor(grab)));
                        else
                            record.add(RecordConverter.toRecord(imageLoader.asRowVector(grab)));

                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }

        return record;
    }

    /** Ugly workaround to a bug in JCodec: https://github.com/jcodec/jcodec/issues/24 */
    private static class FixedByteBufferSeekableByteChannel extends ByteBufferSeekableByteChannel {
        private ByteBuffer backing;

        public FixedByteBufferSeekableByteChannel(ByteBuffer backing) {
            super(backing);
            try {
                Field f = this.getClass().getSuperclass().getDeclaredField("maxPos");
                f.setAccessible(true);
                f.set(this, backing.limit());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            this.backing = backing;
        }

        @Override
        public int read(ByteBuffer dst) throws IOException {
            if (!backing.hasRemaining())
                return -1;
            return super.read(dst);
        }
    }

}
