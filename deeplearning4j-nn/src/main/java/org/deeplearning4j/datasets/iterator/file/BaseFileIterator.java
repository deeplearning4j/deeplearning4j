package org.deeplearning4j.datasets.iterator.file;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.collection.CompactHeapStringList;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;

public abstract class BaseFileIterator<T,P> implements Iterator<T> {

    protected final List<String> list;
    protected final int batchSize;
    protected final Random rng;

    protected int[] order;
    protected int position;

    private T partialStored;
    @Getter @Setter private P preProcessor;


    protected BaseFileIterator(File rootDir, int batchSize, String... validExtensions){
        this(rootDir, true, new Random(), -1, validExtensions);
    }

    protected BaseFileIterator(File rootDir, boolean recursive, Random rng, int batchSize, String... validExtensions){
        this.batchSize = batchSize;
        this.rng = rng;

        list = new CompactHeapStringList();
        Collection<File> c = FileUtils.listFiles(rootDir, validExtensions, recursive);
        if(c.isEmpty()){
            throw new IllegalStateException("Root directory is null " + (validExtensions != null ? " (or all files rejected by extension filter)" : ""));
        }
        for(File f : c){
            list.add(f.getPath());
        }

        if(rng != null){
            order = new int[list.size()];
            for( int i=0; i<order.length; i++ ){
                order[i] = i;
            }
            MathUtils.shuffleArray(order, rng);
        }
    }

    @Override
    public boolean hasNext(){
        return position < list.size();
    }

    @Override
    public T next(){
        if(!hasNext()){
            throw new NoSuchElementException("No next element");
        }
        int nextIdx = (order == null ? order[position++] : position++);

        T next = load(new File(list.get(nextIdx)));
        if(batchSize < 0){
            //Don't recombine, return as-is
            return next;
        }

        if(partialStored == null && sizeOf(next) == batchSize){
            return next;
        }

        int exampleCount = 0;
        List<T> toMerge = new ArrayList<>();
        if(partialStored != null){
            toMerge.add(partialStored);
            exampleCount += sizeOf(partialStored);
            partialStored = null;
        }

        while(exampleCount < batchSize && hasNext()){
            nextIdx = (order == null ? order[position++] : position++);
            next = load(new File(list.get(nextIdx)));
            exampleCount += sizeOf(next);
            toMerge.add(next);
        }

        return mergeAndStoreRemainder(toMerge);
    }

    @Override
    public void remove(){
        throw new UnsupportedOperationException("Not supported");
    }

    protected T mergeAndStoreRemainder(List<T> toMerge){
        //Could be smaller or larger
        List<T> correctNum = new ArrayList<>();
        List<T> remainder = new ArrayList<>();
        int soFar = 0;
        for(T t : toMerge){
            int size = sizeOf(t);

            if(soFar + size <= batchSize ) {
                correctNum.add(t);
                soFar += size;
            } else if(soFar < batchSize){
                //Split and add some
                List<T> split = split(t);
                for(T t2 : split){
                    if(soFar < batchSize){
                        correctNum.add(t2);
                        soFar += sizeOf(t2);
                    } else {
                        remainder.add(t2);
                    }
                }
            } else {
                //Don't need any of this
                remainder.add(t);
            }
        }

        T ret = merge(correctNum);
        try(MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()){
            this.partialStored = merge(remainder);
        }

        return ret;
    }


    public void reset(){
        position = 0;
        if(rng != null){
            MathUtils.shuffleArray(order, rng);
        }
    }

    public boolean resetSupported(){
        return true;
    }

    public boolean asyncSupported(){
        return true;
    }


    protected abstract T load(File f);

    protected abstract int sizeOf(T of);

    protected abstract List<T> split(T toSplit);

    protected abstract T merge(List<T> toMerge);
}
