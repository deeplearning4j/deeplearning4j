package org.apdplat.word.dictionary.impl;

import org.apdplat.word.dictionary.Dictionary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicInteger;
/**
 * An implemention of Aho Corasick algorithm based on Double Array Trie
 *
 */
public class AhoCorasickDoubleArrayTrie<V> implements Serializable, Dictionary {
    private static final Logger LOGGER = LoggerFactory.getLogger(AhoCorasickDoubleArrayTrie.class);
    private AtomicInteger maxLength = new AtomicInteger();
    public AhoCorasickDoubleArrayTrie(){
        LOGGER.info("初始化词典：" + this.getClass().getName());
    }
    /**
     * check array of the Double Array Trie structure
     */
    protected int check[];
    /**
     * base array of the Double Array Trie structure
     */
    protected int base[];
    /**
     * fail table of the Aho Corasick automata
     */
    protected int fail[];
    /**
     * output table of the Aho Corasick automata
     */
    protected int[][] output;
    /**
     * outer value array
     */
    protected V[] v;

    /**
     * the length of every key
     */
    protected int[] l;

    /**
     * the size of base and check array
     */
    protected int size;

    public List<Hit<V>> parseText(String text){
        return parseText(text, 0, text.length());
    }
    /**
     * Parse text
     * @param text The text
     * @return a list of outputs
     */
    public List<Hit<V>> parseText(String text, int start, int length)
    {
        int position = 1;
        int currentState = 0;
        List<Hit<V>> collectedEmits = new LinkedList<Hit<V>>();
        int limit = start + length;
        for (int i = start; i < limit; i++)
        {
            currentState = getState(currentState, text.charAt(i));
            storeEmits(position, currentState, collectedEmits);
            ++position;
        }

        return collectedEmits;
    }

    /**
     * Parse text
     * @param text The text
     * @param processor A processor which handles the output
     */
    public void parseText(String text, IHit<V> processor)
    {
        int position = 1;
        int currentState = 0;
        for (int i = 0; i < text.length(); ++i)
        {
            currentState = getState(currentState, text.charAt(i));
            int[] hitArray = output[currentState];
            if (hitArray != null)
            {
                for (int hit : hitArray)
                {
                    processor.hit(position - l[hit], position, v[hit]);
                }
            }
            ++position;
        }
    }

    /**
     * Parse text
     * @param text The text
     * @param processor A processor which handles the output
     */
    public void parseText(char[] text, IHit<V> processor)
    {
        int position = 1;
        int currentState = 0;
        for (char c : text)
        {
            currentState = getState(currentState, c);
            int[] hitArray = output[currentState];
            if (hitArray != null)
            {
                for (int hit : hitArray)
                {
                    processor.hit(position - l[hit], position, v[hit]);
                }
            }
            ++position;
        }
    }

    /**
     * Parse text
     * @param text The text
     * @param processor A processor which handles the output
     */
    public void parseText(char[] text, IHitFull<V> processor)
    {
        int position = 1;
        int currentState = 0;
        for (char c : text)
        {
            currentState = getState(currentState, c);
            int[] hitArray = output[currentState];
            if (hitArray != null)
            {
                for (int hit : hitArray)
                {
                    processor.hit(position - l[hit], position, v[hit], hit);
                }
            }
            ++position;
        }
    }


    /**
     * Save
     * @param out An ObjectOutputStream object
     * @throws IOException Some IOException
     */
    public void save(ObjectOutputStream out) throws IOException
    {
        out.writeObject(base);
        out.writeObject(check);
        out.writeObject(fail);
        out.writeObject(output);
        out.writeObject(l);
        out.writeObject(v);
    }

    /**
     * Load
     * @param in An ObjectInputStream object
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public void load(ObjectInputStream in) throws IOException, ClassNotFoundException
    {
        base = (int[]) in.readObject();
        check = (int[]) in.readObject();
        fail = (int[]) in.readObject();
        output = (int[][]) in.readObject();
        l = (int[]) in.readObject();
        v = (V[]) in.readObject();
    }

    /**
     * Get value by a String key, just like a map.get() method
     * @param key The key
     * @return
     */
    public V get(String key)
    {
        int index = exactMatchSearch(key);
        if (index >= 0)
        {
            return v[index];
        }

        return null;
    }

    /**
     * Pick the value by index in value array <br>
     * Notice that to be more efficiently, this method DONOT check the parameter
     * @param index The index
     * @return The value
     */
    public V get(int index)
    {
        return v[index];
    }


    @Override
    public int getMaxLength() {
        return maxLength.get();
    }

    @Override
    public boolean contains(String item, int start, int length) {
        if(base==null){
            return false;
        }
        List<Hit<V>> hits = parseText(item, start, length);
        for(Hit<V> hit : hits){
            //System.out.println(hit.begin+":"+hit.end+":"+hit.value);
            if((hit.end-hit.begin)==length){
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean contains(String item) {
        return contains(item, 0, item.length());
    }

    @Override
    public void addAll(List<String> items) {
        if(check!=null){
            throw new RuntimeException("addAll method can just be used once after clear method!");
        }

        Map<String, V> map = new HashMap<>();
        items
            .stream()
            .map(item -> item.trim())
            .filter(item -> {
                //统计最大词长
                int len = item.length();
                if (len > maxLength.get()) {
                    maxLength.set(len);
                }
                return len > 0;
            })
            .forEach(item -> map.put(item, (V)item));
        build(map);
    }

    @Override
    public void add(String item) {
        throw new RuntimeException("not yet support, please use addAll method!");
    }

    @Override
    public void removeAll(List<String> items) {
        throw new RuntimeException("not yet support menthod!");
    }

    @Override
    public void remove(String item) {
        throw new RuntimeException("not yet support menthod!");
    }

    @Override
    public void clear() {
        check = null;
        base = null;
        fail = null;
        output = null;
        v = null;
        l = null;
        size = 0;
        maxLength.set(0);
    }

    /**
     * Processor handles the output when hit a keyword
     */
    public interface IHit<V>
    {
        /**
         * Hit a keyword, you can use some code like text.substring(begin, end) to get the keyword
         * @param begin the beginning index, inclusive.
         * @param end   the ending index, exclusive.
         * @param value the value assigned to the keyword
         */
        void hit(int begin, int end, V value);
    }

    /**
     * Processor handles the output when hit a keyword, with more detail
     */
    public interface IHitFull<V>
    {
        /**
         * Hit a keyword, you can use some code like text.substring(begin, end) to get the keyword
         * @param begin the beginning index, inclusive.
         * @param end   the ending index, exclusive.
         * @param value the value assigned to the keyword
         * @param index the index of the value assigned to the keyword, you can use the integer as a perfect hash value
         */
        void hit(int begin, int end, V value, int index);
    }

    /**
     * A result output
     *
     * @param <V> the value type
     */
    public class Hit<V>
    {
        /**
         * the beginning index, inclusive.
         */
        public final int begin;
        /**
         * the ending index, exclusive.
         */
        public final int end;
        /**
         * the value assigned to the keyword
         */
        public final V value;

        public Hit(int begin, int end, V value)
        {
            this.begin = begin;
            this.end = end;
            this.value = value;
        }

        @Override
        public String toString()
        {
            return String.format("[%d:%d]=%s", begin, end, value);
        }
    }

    /**
     * transmit state, supports failure function
     *
     * @param currentState
     * @param character
     * @return
     */
    private int getState(int currentState, char character)
    {
        int newCurrentState = transitionWithRoot(currentState, character);  // 先按success跳转
        while (newCurrentState == -1) // 跳转失败的话，按failure跳转
        {
            currentState = fail[currentState];
            newCurrentState = transitionWithRoot(currentState, character);
        }
        return newCurrentState;
    }

    /**
     * store output
     *
     * @param position
     * @param currentState
     * @param collectedEmits
     */
    private void storeEmits(int position, int currentState, List<Hit<V>> collectedEmits)
    {
        int[] hitArray = output[currentState];
        if (hitArray != null)
        {
            for (int hit : hitArray)
            {
                collectedEmits.add(new Hit<V>(position - l[hit], position, v[hit]));
            }
        }
    }

    /**
     * transition of a state
     *
     * @param current
     * @param c
     * @return
     */
    protected int transition(int current, char c)
    {
        int b = current;
        int p;

        p = b + c + 1;
        if (b == check[p])
            b = base[p];
        else
            return -1;

        p = b;
        return p;
    }

    /**
     * transition of a state, if the state is root and it failed, then returns the root
     *
     * @param nodePos
     * @param c
     * @return
     */
    protected int transitionWithRoot(int nodePos, char c)
    {
        int b = base[nodePos];
        int p;

        p = b + c + 1;
        if (b != check[p])
        {
            if (nodePos == 0) return 0;
            return -1;
        }

        return p;
    }


    /**
     * Build a AhoCorasickDoubleArrayTrie from a map
     * @param map a map containing key-value pairs
     */
    public void build(Map<String, V> map)
    {
        new Builder().build(map);
    }


    /**
     * match exactly by a key
     *
     * @param key the key
     * @return the index of the key, you can use it as a perfect hash function
     */
    public int exactMatchSearch(String key)
    {
        return exactMatchSearch(key, 0, 0, 0);
    }

    /**
     * match exactly by a key
     *
     * @param key
     * @param pos
     * @param len
     * @param nodePos
     * @return
     */
    private int exactMatchSearch(String key, int pos, int len, int nodePos)
    {
        if (len <= 0)
            len = key.length();
        if (nodePos <= 0)
            nodePos = 0;

        int result = -1;

        char[] keyChars = key.toCharArray();

        int b = base[nodePos];
        int p;

        for (int i = pos; i < len; i++)
        {
            p = b + (int) (keyChars[i]) + 1;
            if (b == check[p])
                b = base[p];
            else
                return result;
        }

        p = b;
        int n = base[p];
        if (b == check[p] && n < 0)
        {
            result = -n - 1;
        }
        return result;
    }

    /**
     * match exactly by a key
     *
     * @param keyChars the char array of the key
     * @param pos      the begin index of char array
     * @param len      the length of the key
     * @param nodePos  the starting position of the node for searching
     * @return the value index of the key, minus indicates null
     */
    private int exactMatchSearch(char[] keyChars, int pos, int len, int nodePos)
    {
        int result = -1;

        int b = base[nodePos];
        int p;

        for (int i = pos; i < len; i++)
        {
            p = b + (int) (keyChars[i]) + 1;
            if (b == check[p])
                b = base[p];
            else
                return result;
        }

        p = b;
        int n = base[p];
        if (b == check[p] && n < 0)
        {
            result = -n - 1;
        }
        return result;
    }

//    /**
//     * Just for debug when I wrote it
//     */
//    public void debug()
//    {
//        System.out.println("base:");
//        for (int i = 0; i < base.length; i++)
//        {
//            if (base[i] < 0)
//            {
//                System.out.println(i + " : " + -base[i]);
//            }
//        }
//
//        System.out.println("output:");
//        for (int i = 0; i < output.length; i++)
//        {
//            if (output[i] != null)
//            {
//                System.out.println(i + " : " + Arrays.toString(output[i]));
//            }
//        }
//
//        System.out.println("fail:");
//        for (int i = 0; i < fail.length; i++)
//        {
//            if (fail[i] != 0)
//            {
//                System.out.println(i + " : " + fail[i]);
//            }
//        }
//
//        System.out.println(this);
//    }
//
//    @Override
//    public String toString()
//    {
//        String infoIndex = "i    = ";
//        String infoChar = "char = ";
//        String infoBase = "base = ";
//        String infoCheck = "check= ";
//        for (int i = 0; i < Math.min(base.length, 200); ++i)
//        {
//            if (base[i] != 0 || check[i] != 0)
//            {
//                infoChar += "    " + (i == check[i] ? " ×" : (char) (i - check[i] - 1));
//                infoIndex += " " + String.format("%5d", i);
//                infoBase += " " + String.format("%5d", base[i]);
//                infoCheck += " " + String.format("%5d", check[i]);
//            }
//        }
//        return "DoubleArrayTrie：" +
//                "\n" + infoChar +
//                "\n" + infoIndex +
//                "\n" + infoBase +
//                "\n" + infoCheck + "\n" +
////                "check=" + Arrays.toString(check) +
////                ", base=" + Arrays.toString(base) +
////                ", used=" + Arrays.toString(used) +
//                "size=" + size
////                ", length=" + Arrays.toString(length) +
////                ", value=" + Arrays.toString(value) +
//                ;
//    }
//
//    /**
//     * 一个顺序输出变量名与变量值的调试类
//     */
//    private static class DebugArray
//    {
//        Map<String, String> nameValueMap = new LinkedHashMap<String, String>();
//
//        public void add(String name, int value)
//        {
//            String valueInMap = nameValueMap.get(name);
//            if (valueInMap == null)
//            {
//                valueInMap = "";
//            }
//
//            valueInMap += " " + String.format("%5d", value);
//
//            nameValueMap.put(name, valueInMap);
//        }
//
//        @Override
//        public String toString()
//        {
//            String text = "";
//            for (Map.Entry<String, String> entry : nameValueMap.entrySet())
//            {
//                String name = entry.getKey();
//                String value = entry.getValue();
//                text += String.format("%-5s", name) + "= " + value + '\n';
//            }
//
//            return text;
//        }
//
//        public void println()
//        {
//            System.out.print(this);
//        }
//    }

    /**
     * Get the size of the keywords
     * @return
     */
    public int size()
    {
        return v.length;
    }

    /**
     * A builder to build the AhoCorasickDoubleArrayTrie
     */
    private class Builder
    {
        /**
         * the root state of trie
         */
        private State rootState = new State();
        /**
         * whether the position has been used
         */
        private boolean used[];
        /**
         * the allocSize of the dynamic array
         */
        private int allocSize;
        /**
         * a parameter controls the memory growth speed of the dynamic array
         */
        private int progress;
        /**
         * the next position to check unused memory
         */
        private int nextCheckPos;
        /**
         * the size of the key-pair sets
         */
        private int keySize;

        /**
         * Build from a map
         * @param map a map containing key-value pairs
         */
        @SuppressWarnings("unchecked")
        public void build(Map<String, V> map)
        {
            // 把值保存下来
            v = (V[]) map.values().toArray();
            l = new int[v.length];
            Set<String> keySet = map.keySet();
            // 构建二分trie树
            addAllKeyword(keySet);
            // 在二分trie树的基础上构建双数组trie树
            buildDoubleArrayTrie(keySet.size());
            used = null;
            // 构建failure表并且合并output表
            constructFailureStates();
            rootState = null;
            loseWeight();
        }

        /**
         * fetch siblings of a parent node
         *
         * @param parent   parent node
         * @param siblings parent node's child nodes, i . e . the siblings
         * @return the amount of the siblings
         */
        private int fetch(State parent, List<Map.Entry<Integer, State>> siblings)
        {
            if (parent.isAcceptable())
            {
                State fakeNode = new State(-(parent.getDepth() + 1));  // 此节点是parent的子节点，同时具备parent的输出
                fakeNode.addEmit(parent.getLargestValueId());
                siblings.add(new AbstractMap.SimpleEntry<Integer, State>(0, fakeNode));
            }
            for (Map.Entry<Character, State> entry : parent.getSuccess().entrySet())
            {
                siblings.add(new AbstractMap.SimpleEntry<Integer, State>(entry.getKey() + 1, entry.getValue()));
            }
            return siblings.size();
        }

        /**
         * add a keyword
         *
         * @param keyword a keyword
         * @param index   the index of the keyword
         */
        private void addKeyword(String keyword, int index)
        {
            State currentState = this.rootState;
            for (Character character : keyword.toCharArray())
            {
                currentState = currentState.addState(character);
            }
            currentState.addEmit(index);
            l[index] = keyword.length();
        }

        /**
         * add a collection of keywords
         *
         * @param keywordSet the collection holding keywords
         */
        private void addAllKeyword(Collection<String> keywordSet)
        {
            int i = 0;
            for (String keyword : keywordSet)
            {
                addKeyword(keyword, i++);
            }
        }

        /**
         * construct failure table
         */
        private void constructFailureStates()
        {
            fail = new int[size + 1];
            fail[1] = base[0];
            output = new int[size + 1][];
            Queue<State> queue = new LinkedBlockingDeque<State>();

            // 第一步，将深度为1的节点的failure设为根节点
            for (State depthOneState : this.rootState.getStates())
            {
                depthOneState.setFailure(this.rootState, fail);
                queue.add(depthOneState);
                constructOutput(depthOneState);
            }

            // 第二步，为深度 > 1 的节点建立failure表，这是一个bfs
            while (!queue.isEmpty())
            {
                State currentState = queue.remove();

                for (Character transition : currentState.getTransitions())
                {
                    State targetState = currentState.nextState(transition);
                    queue.add(targetState);

                    State traceFailureState = currentState.failure();
                    while (traceFailureState.nextState(transition) == null)
                    {
                        traceFailureState = traceFailureState.failure();
                    }
                    State newFailureState = traceFailureState.nextState(transition);
                    targetState.setFailure(newFailureState, fail);
                    targetState.addEmit(newFailureState.emit());
                    constructOutput(targetState);
                }
            }
        }

        /**
         * construct output table
         */
        private void constructOutput(State targetState)
        {
            Collection<Integer> emit = targetState.emit();
            if (emit == null || emit.size() == 0) return;
            int output[] = new int[emit.size()];
            Iterator<Integer> it = emit.iterator();
            for (int i = 0; i < output.length; ++i)
            {
                output[i] = it.next();
            }
            AhoCorasickDoubleArrayTrie.this.output[targetState.getIndex()] = output;
        }

        private void buildDoubleArrayTrie(int keySize)
        {
            progress = 0;
            this.keySize = keySize;
            resize(65536 * 32); // 32个双字节

            base[0] = 1;
            nextCheckPos = 0;

            State root_node = this.rootState;

            List<Map.Entry<Integer, State>> siblings = new ArrayList<Map.Entry<Integer, State>>(root_node.getSuccess().entrySet().size());
            fetch(root_node, siblings);
            insert(siblings);
        }

        /**
         * allocate the memory of the dynamic array
         *
         * @param newSize
         * @return
         */
        private int resize(int newSize)
        {
            int[] base2 = new int[newSize];
            int[] check2 = new int[newSize];
            boolean used2[] = new boolean[newSize];
            if (allocSize > 0)
            {
                System.arraycopy(base, 0, base2, 0, allocSize);
                System.arraycopy(check, 0, check2, 0, allocSize);
                System.arraycopy(used, 0, used2, 0, allocSize);
            }

            base = base2;
            check = check2;
            used = used2;

            return allocSize = newSize;
        }

        /**
         * insert the siblings to double array trie
         *
         * @param siblings the siblings being inserted
         * @return the position to insert them
         */
        private int insert(List<Map.Entry<Integer, State>> siblings)
        {
            int begin = 0;
            int pos = Math.max(siblings.get(0).getKey() + 1, nextCheckPos) - 1;
            int nonzero_num = 0;
            int first = 0;

            if (allocSize <= pos)
                resize(pos + 1);

            outer:
            // 此循环体的目标是找出满足base[begin + a1...an]  == 0的n个空闲空间,a1...an是siblings中的n个节点
            while (true)
            {
                pos++;

                if (allocSize <= pos)
                    resize(pos + 1);

                if (check[pos] != 0)
                {
                    nonzero_num++;
                    continue;
                }
                else if (first == 0)
                {
                    nextCheckPos = pos;
                    first = 1;
                }

                begin = pos - siblings.get(0).getKey(); // 当前位置离第一个兄弟节点的距离
                if (allocSize <= (begin + siblings.get(siblings.size() - 1).getKey()))
                {
                    // progress can be zero // 防止progress产生除零错误
                    double l = (1.05 > 1.0 * keySize / (progress + 1)) ? 1.05 : 1.0 * keySize / (progress + 1);
                    resize((int) (allocSize * l));
                }

                if (used[begin])
                    continue;

                for (int i = 1; i < siblings.size(); i++)
                    if (check[begin + siblings.get(i).getKey()] != 0)
                        continue outer;

                break;
            }

            // -- Simple heuristics --
            // if the percentage of non-empty contents in check between the
            // index
            // 'next_check_pos' and 'check' is greater than some constant value
            // (e.g. 0.9),
            // new 'next_check_pos' index is written by 'check'.
            if (1.0 * nonzero_num / (pos - nextCheckPos + 1) >= 0.95)
                nextCheckPos = pos; // 从位置 next_check_pos 开始到 pos 间，如果已占用的空间在95%以上，下次插入节点时，直接从 pos 位置处开始查找
            used[begin] = true;

            size = (size > begin + siblings.get(siblings.size() - 1).getKey() + 1) ? size : begin + siblings.get(siblings.size() - 1).getKey() + 1;

            for (Map.Entry<Integer, State> sibling : siblings)
            {
                check[begin + sibling.getKey()] = begin;
            }

            for (Map.Entry<Integer, State> sibling : siblings)
            {
                List<Map.Entry<Integer, State>> new_siblings = new ArrayList<Map.Entry<Integer, State>>(sibling.getValue().getSuccess().entrySet().size() + 1);

                if (fetch(sibling.getValue(), new_siblings) == 0)  // 一个词的终止且不为其他词的前缀，其实就是叶子节点
                {
                    base[begin + sibling.getKey()] = (-sibling.getValue().getLargestValueId() - 1);
                    progress++;
                }
                else
                {
                    int h = insert(new_siblings);   // dfs
                    base[begin + sibling.getKey()] = h;
                }
                sibling.getValue().setIndex(begin + sibling.getKey());
            }
            return begin;
        }

        /**
         * free the unnecessary memory
         */
        private void loseWeight()
        {
            int nbase[] = new int[size + 65535];
            System.arraycopy(base, 0, nbase, 0, size);
            base = nbase;

            int ncheck[] = new int[size + 65535];
            System.arraycopy(check, 0, ncheck, 0, size);
            check = ncheck;
        }
    }
    /**
     * <p>
     * 一个状态有如下几个功能
     * </p>
     * <p/>
     * <ul>
     * <li>success; 成功转移到另一个状态</li>
     * <li>failure; 不可顺着字符串跳转的话，则跳转到一个浅一点的节点</li>
     * <li>emits; 命中一个模式串</li>
     * </ul>
     * <p/>
     * <p>
     * 根节点稍有不同，根节点没有 failure 功能，它的“failure”指的是按照字符串路径转移到下一个状态。其他节点则都有failure状态。
     * </p>
     *
     */
    private static class State {

        /**
         * 模式串的长度，也是这个状态的深度
         */
        protected final int depth;

        /**
         * fail 函数，如果没有匹配到，则跳转到此状态。
         */
        private State failure = null;

        /**
         * 只要这个状态可达，则记录模式串
         */
        private Set<Integer> emits = null;
        /**
         * goto 表，也称转移函数。根据字符串的下一个字符转移到下一个状态
         */
        private Map<Character, State> success = new TreeMap<Character, State>();

        /**
         * 在双数组中的对应下标
         */
        private int index;

        /**
         * 构造深度为0的节点
         */
        public State()
        {
            this(0);
        }

        /**
         * 构造深度为depth的节点
         * @param depth
         */
        public State(int depth)
        {
            this.depth = depth;
        }

        /**
         * 获取节点深度
         * @return
         */
        public int getDepth()
        {
            return this.depth;
        }

        /**
         * 添加一个匹配到的模式串（这个状态对应着这个模式串)
         * @param keyword
         */
        public void addEmit(int keyword)
        {
            if (this.emits == null)
            {
                this.emits = new TreeSet<Integer>(Collections.reverseOrder());
            }
            this.emits.add(keyword);
        }

        /**
         * 获取最大的值
         * @return
         */
        public Integer getLargestValueId()
        {
            if (emits == null || emits.size() == 0) return null;

            return emits.iterator().next();
        }

        /**
         * 添加一些匹配到的模式串
         * @param emits
         */
        public void addEmit(Collection<Integer> emits)
        {
            for (int emit : emits)
            {
                addEmit(emit);
            }
        }

        /**
         * 获取这个节点代表的模式串（们）
         * @return
         */
        public Collection<Integer> emit()
        {
            return this.emits == null ? Collections.<Integer>emptyList() : this.emits;
        }

        /**
         * 是否是终止状态
         * @return
         */
        public boolean isAcceptable()
        {
            return this.depth > 0 && this.emits != null;
        }

        /**
         * 获取failure状态
         * @return
         */
        public State failure()
        {
            return this.failure;
        }

        /**
         * 设置failure状态
         * @param failState
         */
        public void setFailure(State failState, int fail[])
        {
            this.failure = failState;
            fail[index] = failState.index;
        }

        /**
         * 转移到下一个状态
         * @param character 希望按此字符转移
         * @param ignoreRootState 是否忽略根节点，如果是根节点自己调用则应该是true，否则为false
         * @return 转移结果
         */
        private State nextState(Character character, boolean ignoreRootState)
        {
            State nextState = this.success.get(character);
            if (!ignoreRootState && nextState == null && this.depth == 0)
            {
                nextState = this;
            }
            return nextState;
        }

        /**
         * 按照character转移，根节点转移失败会返回自己（永远不会返回null）
         * @param character
         * @return
         */
        public State nextState(Character character)
        {
            return nextState(character, false);
        }

        /**
         * 按照character转移，任何节点转移失败会返回null
         * @param character
         * @return
         */
        public State nextStateIgnoreRootState(Character character)
        {
            return nextState(character, true);
        }

        public State addState(Character character)
        {
            State nextState = nextStateIgnoreRootState(character);
            if (nextState == null)
            {
                nextState = new State(this.depth + 1);
                this.success.put(character, nextState);
            }
            return nextState;
        }

        public Collection<State> getStates()
        {
            return this.success.values();
        }

        public Collection<Character> getTransitions()
        {
            return this.success.keySet();
        }

        @Override
        public String toString()
        {
            final StringBuilder sb = new StringBuilder("State{");
            sb.append("depth=").append(depth);
            sb.append(", ID=").append(index);
            sb.append(", emits=").append(emits);
            sb.append(", success=").append(success.keySet());
            sb.append(", failureID=").append(failure == null ? "-1" : failure.index);
            sb.append(", failure=").append(failure);
            sb.append('}');
            return sb.toString();
        }

        /**
         * 获取goto表
         * @return
         */
        public Map<Character, State> getSuccess()
        {
            return success;
        }

        public int getIndex()
        {
            return index;
        }

        public void setIndex(int index)
        {
            this.index = index;
        }
    }

    public static void main(String[] args) {
        AhoCorasickDoubleArrayTrie<String> dictionary = new AhoCorasickDoubleArrayTrie<>();

        List<String> words = Arrays.asList("杨尚川", "章子怡", "刘亦菲", "刘", "刘诗诗", "巩俐", "中国", "主演");

        //构造词典
        dictionary.addAll(words);
        System.out.println("增加数据：" + words);

        System.out.println("最大词长：" + dictionary.getMaxLength());
        System.out.println("查找 杨尚川的梦中情人是刘亦菲，曾经也爱过章子怡，刘诗诗以及巩俐：" + dictionary.contains("杨尚川的梦中情人是刘亦菲，曾经也爱过章子怡，刘诗诗以及巩俐"));
        System.out.println("查找 杨尚川：" + dictionary.contains("杨尚川"));
        System.out.println("查找 章子怡：" + dictionary.contains("章子怡"));
        System.out.println("查找 刘："+dictionary.contains("刘"));
        System.out.println("查找 刘亦菲：" + dictionary.contains("刘亦菲"));
        System.out.println("查找 刘诗诗：" + dictionary.contains("刘诗诗"));
        System.out.println("查找 巩俐："+dictionary.contains("巩俐"));
        System.out.println("查找 中国的巩俐是红高粱的主演 3 2：" + dictionary.contains("中国的巩俐是红高粱的主演", 3, 2));
        System.out.println("查找 中国的巩俐是红高粱的主演 0 2：" + dictionary.contains("中国的巩俐是红高粱的主演", 0, 2));
        System.out.println("查找 中国的巩俐是红高粱的主演 10 2：" + dictionary.contains("中国的巩俐是红高粱的主演", 10, 2));
        System.out.println("查找 复仇者联盟2：" + dictionary.contains("复仇者联盟2"));
        System.out.println("查找 白掌：" + dictionary.contains("白掌"));
        System.out.println("查找 红掌：" + dictionary.contains("红掌"));

        dictionary.clear();
        System.out.println("清除所有数据");

        System.out.println("查找 杨尚川：" + dictionary.contains("杨尚川"));
        System.out.println("查找 章子怡：" + dictionary.contains("章子怡"));

        List<String> data = new ArrayList<>();
        data.add("白掌");
        data.add("红掌");
        data.add("复仇者联盟2");
        data.addAll(words);

        dictionary.addAll(data);
        System.out.println("增加数据：" + data);

        System.out.println("查找 杨尚川：" + dictionary.contains("杨尚川"));
        System.out.println("查找 章子怡：" + dictionary.contains("章子怡"));
        System.out.println("最大词长：" + dictionary.getMaxLength());
        System.out.println("查找 复仇者联盟2："+dictionary.contains("复仇者联盟2"));
        System.out.println("查找 白掌：" + dictionary.contains("白掌"));
        System.out.println("查找 红掌："+dictionary.contains("红掌"));
        System.out.println("查找 刘亦菲："+dictionary.contains("刘亦菲"));
        System.out.println("查找 刘诗诗："+dictionary.contains("刘诗诗"));
        System.out.println("查找 巩俐：" + dictionary.contains("巩俐"));
        System.out.println("查找 金钱树："+dictionary.contains("金钱树"));
    }
}
