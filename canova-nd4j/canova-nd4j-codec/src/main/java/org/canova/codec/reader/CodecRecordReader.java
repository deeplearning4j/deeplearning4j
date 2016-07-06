/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.codec.reader;

import org.apache.commons.compress.utils.IOUtils;
import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.canova.common.RecordConverter;
import org.canova.image.loader.ImageLoader;
import org.jcodec.api.FrameGrab;
import org.jcodec.api.JCodecException;
import org.jcodec.common.ByteBufferSeekableByteChannel;
import org.jcodec.common.NIOUtils;
import org.jcodec.common.SeekableByteChannel;
import org.jcodec.scale.AWTUtil;


import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;

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
public class CodecRecordReader extends FileRecordReader implements SequenceRecordReader {
    private int startFrame = 0;
    private int numFrames = -1;
    private int totalFrames = -1;
    private double framesPerSecond = -1;
    private double videoLength = -1;
    private ImageLoader imageLoader;
    private boolean ravel = false;

    public final static String NAME_SPACE = "org.canova.codec.reader";
    public final static String ROWS = NAME_SPACE + ".rows";
    public final static String COLUMNS = NAME_SPACE + ".columns";
    public final static String START_FRAME = NAME_SPACE + ".startframe";
    public final static String TOTAL_FRAMES = NAME_SPACE + ".frames";
    public final static String TIME_SLICE = NAME_SPACE + ".time";
    public final static String RAVEL = NAME_SPACE + ".ravel";
    public final static String VIDEO_DURATION = NAME_SPACE + ".duration";


    @Override
    public Collection<Collection<Writable>> sequenceRecord() {
        File next = iter.next();

        try{
            return loadData(NIOUtils.readableFileChannel(next));
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public Collection<Collection<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        //Reading video from DataInputStream: Need data from this stream in a SeekableByteChannel
        //Approach used here: load entire video into memory -> ByteBufferSeekableByteChanel
        byte[] data = IOUtils.toByteArray(dataInputStream);
        ByteBuffer bb = ByteBuffer.wrap(data);
        SeekableByteChannel sbc = new FixedByteBufferSeekableByteChannel(bb);
        return loadData(sbc);
    }

    private Collection<Collection<Writable>> loadData( SeekableByteChannel seekableByteChannel ) throws IOException {
        Collection<Collection<Writable>> record = new ArrayList<>();

        if(numFrames >= 1) {
            FrameGrab fg;
            try{
                fg = new FrameGrab(seekableByteChannel);
                if(startFrame != 0) fg.seekToFramePrecise(startFrame);
            } catch(JCodecException e){
                throw new RuntimeException(e);
            }

            for(int i = startFrame; i < startFrame+numFrames; i++) {
                try {
                    BufferedImage grab = fg.getFrame();
                    if(ravel)
                        record.add(RecordConverter.toRecord(imageLoader.toRaveledTensor(grab)));
                    else
                        record.add(RecordConverter.toRecord(imageLoader.asRowVector(grab)));

                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        } else {
            if(framesPerSecond < 1)
                throw new IllegalStateException("No frames or frame time intervals specified");


            else {
                for(double i = 0; i < videoLength; i += framesPerSecond) {
                    try {
                        BufferedImage grab = FrameGrab.getFrame(seekableByteChannel, i);
                        if(ravel)
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


    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        setConf(conf);
        initialize(split);
    }

    @Override
    public Collection<Writable> next(){
        throw new UnsupportedOperationException("next() not supported for CodecRecordReader (use: sequenceRecord)");
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("record(URI,DataInputStream) not supported for CodecRecordReader");
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public void setConf(Configuration conf) {
        super.setConf(conf);
        startFrame = conf.getInt(START_FRAME,0);
        numFrames = conf.getInt(TOTAL_FRAMES,-1);
        int rows = conf.getInt(ROWS,28);
        int cols = conf.getInt(COLUMNS,28);
        imageLoader = new ImageLoader(rows,cols);
        framesPerSecond = conf.getFloat(TIME_SLICE,-1);
        videoLength = conf.getFloat(VIDEO_DURATION,-1);
        ravel = conf.getBoolean(RAVEL, false);
        totalFrames = conf.getInt(TOTAL_FRAMES, -1);
    }

    @Override
    public Configuration getConf() {
        return super.getConf();
    }


    /** Ugly workaround to a bug in JCodec: https://github.com/jcodec/jcodec/issues/24 */
    private static class FixedByteBufferSeekableByteChannel extends ByteBufferSeekableByteChannel {
        private ByteBuffer backing;
        public FixedByteBufferSeekableByteChannel(ByteBuffer backing) {
            super(backing);
            try{
                Field f = this.getClass().getSuperclass().getDeclaredField("maxPos");
                f.setAccessible(true);
                f.set(this,backing.limit());
            }catch(Exception e){
                throw new RuntimeException(e);
            }
            this.backing = backing;
        }

        @Override
        public int read(ByteBuffer dst) throws IOException {
            if(!backing.hasRemaining()) return -1;
            return super.read(dst);
        }
    }

}
