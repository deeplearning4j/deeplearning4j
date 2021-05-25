
/*******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

//
// @author AbdelRauf
//

#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <cmath>
#include <execution/Threads.h>
#include <execution/ThreadPool.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/ctc.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
struct BeamProb
{
    T total = negative_infinity<T>();
    T non_blank = negative_infinity<T>();
    T blank = negative_infinity<T>(); //log(1)
};


template <typename T, typename T2 = void>
struct DefaultInvalid
{
    static constexpr T value = T();
};


template <typename T>
struct DefaultInvalid<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
    static constexpr T value = static_cast<T>(-1);
};

template <typename T>
struct SequenceNode
{
    //intrusive double links
    SequenceNode<T>* prev = nullptr;
    SequenceNode<T>* next = nullptr;

    //sequence prefix/parent
    SequenceNode<T>* prefix = nullptr;

    T value = DefaultInvalid<T>::value;

    int state = 0;

    void markAsFullyExtended()
    {
        state |= 1;
    }

    void increaseRef()
    {
        //we will have just two copies in bad case. so just or
        state = state | 2;
    }

    void decreaseRef()
    {
        //we will have just two cases in bad case, so just remove that
        state = state & (-2);
    }

    bool safeToRemove()
    {

        if (state & 1) return false;

        decreaseRef();
        //we do not want to remove parent nodes in our case. otherwise just returning state<=1 is ok
        return state == 0;
    }

    bool isFullyExtended() const { return state & 1; }
};

/***
 * Sequence container.
 *
 * NOTE: it is not thread-safe
 *
 * Extend path - O(1)
 * Remove path - O(1)
 * Generating Sequence with backtracking prefix:  O(n)
 *
 * Note: Sequence container is implemented primitively and only usable within this task.
 * As it does not behave as a fully capable tree. some cases should be handled manually
 *
 * Here is special cases that should be handled manually to exploit tree/graph behaviour:
 *
 *   Extending new path value:
 *
 *        To extend the path one need to give path and value and in return get new_path:
 *            new_path = container.extendPath ( path, new_value );
 *
 *        Also note that:
 *        SequenceContainer has already default empty path as a beginning point for paths.
 *        So as an initial node one should use it.
 *           initial_path = container.getEmptyPath();
 *
 *   Adding new path that could be already in container:
 *
 *      Assume we have two paths that can overlap in next step
 *      1st path: node#0() -> node#1(1)                   => generated sequence {},{1}
 *      2nd path: node#0() -> node#1(1) -> node#2(2)      => generated sequence {},{1}, {2}
 *
 *      While extending the first path with value (2). it will be:
 *
 *      node#0() -> node#0(1) -> node#( either new or old)(2)       => generated sequence {},{1}, {2}
 *
 *      For some tasks its not desired to have additional node that will generate the same sequence.
 *      For example:
 *        Assume you wanted to use it as sequence entry in map with just (entry->prefix, entry->value).
 *        so in that case having different paths is not correct and will not be unique in map.
 *
 *      there is not direct way to handle that in our container other than searching.
 *      So one should look for the node with prefix node#1(1) and value(2) and return that node instead of adding new one

 *      Fortunately, for our beam search case:
 *
 *      we need only look for such overlapped cases within the candidates list.
 *      which makes it easy to determine them beforehand while finding and marking overlapped cases. instead of looking for it in SequenceContainer
 *
 *   Removing the same nodes multiple times:
 *        It is fast to remove nodes. As nodes can be stored externally One should follow this rule:
 *
 *        One should not remove the same node twice as it will lead to double free. as Nodes are pointers the same applies to removing a copy
 *
 *        There could be cases where you would like to store copy of nodes. in that cases you can use below method to be able to safely remove:
 *           node should have mutable method named safeToRemove().
 *           Basic implementation will be decreasing reference/copy counts and returning true if it is safe to delete
 *
 *
 */
template <typename T>
class SequenceContainer
{
public:
    SequenceContainer() : count_(1)
    {
        empty_path = new SequenceNode<T>();
        current_ = empty_path;
    }

    SequenceContainer(const SequenceContainer& s) = delete;

    SequenceContainer(SequenceContainer&& other) noexcept
    {
        this->current_ = other.current_;
        other.current_ = nullptr;
    }

    SequenceContainer& operator=(const SequenceContainer& other) = delete;

    SequenceContainer& operator=(SequenceContainer&& other) noexcept
    {
        if (this != other)
        {
            clear();
            this->current_ = other.current_;
            this->count_ = other.count_;
            other.current_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    SequenceNode<T>* getEmptyPath()
    {
        return current_;
    }

    SequenceNode<T>* extendPath(SequenceNode<T>* prefix, T value)
    {
        auto new_node = new SequenceNode<T>();

        new_node->value = value;
        new_node->prefix = prefix;
        //add in the holder
        new_node->next = nullptr;
        new_node->prev = current_;
        /*std::cout << "add " << (long long)new_node << std::endl;
        print_seq1(new_node);*/
        if (current_) current_->next = new_node;

        current_ = new_node;
        count_++;
        return new_node;
    }

    void remove(SequenceNode<T>* seq)
    {
        if (seq == nullptr) return;

        if (!seq->safeToRemove()) return;

        SequenceNode<T>* previous = seq->prev;
        SequenceNode<T>* next = seq->next;
        if (previous) previous->next = next;
        if (next) next->prev = previous;

        if (current_ == seq)
        {
            current_ = previous;
        }
        //std::cout << "remove " << (long long)seq << " " << std::endl;
        //print_seq1(seq);
        delete seq;
        count_--;
    }

    static std::vector<T> getSequence(SequenceNode<T>* seq, size_t reserve_size = 1024)
    {
        std::vector<T> ret;
        ret.reserve(reserve_size);
        SequenceNode<T>* backtrack = seq;
        while (backtrack)
        {
            ret.push_back(backtrack->value);
            backtrack = backtrack->prefix;
        }
        if (ret.size() > 1)
        {
            //remove last default node
            ret.pop_back();
            //reverse
            std::reverse(std::begin(ret), std::end(ret));
            return ret;
        }
        return {};
    }

    void clear()
    {
        //destruct all nodes
        SequenceNode<T>* del = current_;
        //int i = 0;
        while (del)
        {
            //++i;
            SequenceNode<T>* temp = del->prev;
            delete del;
            del = temp;
        }
        current_ = nullptr;
        //assert(count_==i);
    }

    ~SequenceContainer()
    {
        clear();
    }

private:
    SequenceNode<T>* current_ = nullptr;

    SequenceNode<T>* empty_path = nullptr;

    int count_ = 0;
};

template <typename T, typename U>
struct BeamEntry
{
    SequenceNode<U>* sequence{};
    BeamProb<T> prob;
};


template <typename T, typename U>
struct BeamEntryEx
{
    BeamEntry<T, U> entry;
    //keep indices for lookUp
    int index_as_child = -1;
    int index_as_parent = -1;
    int children_count = 0;
};

template <typename T, typename U>
struct LookUpEntry
{
    U last_c;  //this is is the same as node->value. just we added for the speed
    SequenceNode<U>* node = nullptr;
    int next_beam_index = -1; //index inside next_beam array
};

template <typename T, typename U>
bool compare_beam_prob(const BeamEntry<T, U>& i1, const BeamEntry<T, U>& i2)
{
    return (i1.prob.total > i2.prob.total);
}


template <typename T, typename U>
T pr(const int c, const BeamProb<T>& beam_prob, const SequenceNode<U>* seq, const T prob)
{
    return seq->value == c ? beam_prob.blank + prob : beam_prob.total + prob;
}

template<bool HasElementStride = false, typename Type, typename IndexType>
void inner_beam_search(const Type* log_p, const uint64_t inc_p, IndexType* result_sequence, const uint64_t inc_res_seq, 
                       const uint64_t max_len_t, Type* result_prob, IndexType* result_seq_length, uint64_t len_t,
                       const uint64_t len_c, const int blank_index, int beam_width, int nbest_len, bool normalize_logits, const uint64_t element_stride = 1L)
{

    using BeamEntryType = BeamEntry<Type, IndexType>;
    using BeamEntryTypeEx = BeamEntryEx<Type, IndexType>;

    if (beam_width < 1) beam_width = 1;
    if (nbest_len > beam_width) nbest_len = beam_width;
    //if len_t is greater than max_len_t truncate it
    len_t = len_t > max_len_t ? max_len_t : len_t;

    SequenceContainer<IndexType> sequence_container;
    BeamEntryType empty;
    empty.prob.blank = 0;
    empty.prob.total = log_sum_exp(empty.prob.blank, empty.prob.non_blank);
    empty.sequence = sequence_container.getEmptyPath();

    //vectors: we will use it as array, here
    std::vector<BeamEntryTypeEx> last_beams;
    std::vector<BeamEntryType> next_beams;
    last_beams.resize(beam_width);
    //as we skip blank indexes the count is beam_width * len_c 
    next_beams.resize(beam_width * len_c);
    last_beams[0].entry = empty;
    last_beams[0].index_as_child = -1;
    last_beams[0].index_as_parent = -1;
    last_beams[0].children_count = 0;
    auto last_beam_size = 1;

    // lookupContainer:
    // it will keep sorted entries. so we will just move and compare the entry
    // in each step there will be overlapped cases
    // the size of overlapped cases in last_beam[0:beam_width]:
    //    as we have beam_width size in each step after sort and pruning
    //    there is at least one item who will not have any parent
    //    and for the rest (beam_width-1) it will check  has_parent_in_container() ? 1 : 0
    //    so maximum size of overlapped pairs is  beam_width-1 

    std::vector<LookUpEntry<Type, IndexType>> lookUp;
    lookUp.resize(beam_width - 1);

    //additional storage to sort overlapped case by classes
    std::vector<std::pair<IndexType, int >> child_class_sorter_help;
    child_class_sorter_help.resize(beam_width - 1);
    Type norm_offset = 0;

    for (uint64_t t = 0; t < len_t; t++)
    {
        auto next_beam_size = 0;
        if (normalize_logits){
            norm_offset = softmax_normalization_term<HasElementStride, Type, IndexType>(log_p, len_c, element_stride);
        }
        for (auto j = 0; j < last_beam_size; j++)
        {
            SequenceNode<IndexType>* seq = last_beams[j].entry.sequence;
            auto& cur_prob = last_beams[j].entry.prob;
            //if len(seq) > 0 then
            const auto log_p_blank = element<HasElementStride>(log_p, blank_index, element_stride);
            Type blank_prob, non_blank_prob;
            //log_p[seq->value] 
            non_blank_prob = seq->value != -1 ? (element<HasElementStride>(log_p, seq->value, element_stride) + cur_prob.non_blank) : negative_infinity<Type>();
            blank_prob = log_p_blank + cur_prob.total;

            if (normalize_logits){
                non_blank_prob = non_blank_prob - norm_offset;
                blank_prob = blank_prob - norm_offset;
            }

            auto look_up_beam_index = -1;

            if (last_beams[j].index_as_child != -1)
            {
                //check entry
                look_up_beam_index = lookUp[last_beams[j].index_as_child].next_beam_index;
            }

            if (look_up_beam_index == -1)
            {
                BeamEntryType entry;
                entry.sequence = seq;
                entry.prob.blank = blank_prob;
                entry.prob.non_blank = non_blank_prob;
                entry.prob.total = log_sum_exp(blank_prob, non_blank_prob);
                next_beams[next_beam_size] = entry;
                //map if its overlapped one. in this case just being child is enough
                if (last_beams[j].index_as_child != -1)
                {
                    lookUp[last_beams[j].index_as_child].next_beam_index = next_beam_size;
                }
                ++next_beam_size;
            }
            else
            {
                //note: here we took as ref &
                auto& entry_prob = next_beams[look_up_beam_index].prob;
                entry_prob.blank = log_sum_exp(entry_prob.blank, blank_prob);
                entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
                entry_prob.total = log_sum_exp(entry_prob.blank, entry_prob.non_blank);
            }
            //check to see if it is overlapped parent
            auto start_index = last_beams[j].index_as_parent;
            auto end_index = last_beams[j].index_as_parent + last_beams[j].children_count;

            for (int c = 0; c < len_c; c++)
            {
                if (c == blank_index) continue;

                const auto prob = element<HasElementStride>(log_p, c, element_stride);//log_p[c];

                non_blank_prob = pr(c, cur_prob, seq, prob);
                if(normalize_logits) non_blank_prob = non_blank_prob - norm_offset;
                //extend by new character 
                auto look_up_beam_index_ex = -1; 
                int found_index = -1;

                //get index within array if its that class index
                if (start_index < end_index && lookUp[start_index].last_c == c){
                        look_up_beam_index_ex = lookUp[start_index].next_beam_index;
                        
                        found_index = start_index;
                        ++start_index; 
                }

                if (look_up_beam_index_ex == -1)
                {
                    BeamEntryType entry;
                    SequenceNode<IndexType>* extended_sequence;
                    if (found_index!=-1)
                    { 
                        extended_sequence = lookUp[found_index].node;
                        //assing next_beam_index for lookup
                        lookUp[found_index].next_beam_index = next_beam_size;
                        extended_sequence->increaseRef();
                    }
                    else {
                        extended_sequence = sequence_container.extendPath(seq, c);
                    }
                    entry.prob.non_blank = non_blank_prob;
                    entry.prob.total = non_blank_prob;
                    entry.sequence = extended_sequence;
                    next_beams[next_beam_size] = entry;

                    ++next_beam_size;
                }
                else
                {
                    auto& entry_prob = next_beams[look_up_beam_index_ex].prob;
                    entry_prob.non_blank = log_sum_exp(entry_prob.non_blank, non_blank_prob);
                    entry_prob.total = log_sum_exp(entry_prob.total, non_blank_prob);
                }
            } //iteration over classes

            //mark it as extended
            seq->markAsFullyExtended();

        } //iteration over  beams

        log_p += inc_p;

        last_beam_size = std::min(next_beam_size, beam_width);
#if !defined(NTH_ELEMENT)
        //sort next beams to get candidates
        std::partial_sort(std::begin(next_beams),
            std::begin(next_beams) + last_beam_size,
            std::begin(next_beams) + next_beam_size, compare_beam_prob<Type, IndexType>);

#else
        std::nth_element(std::begin(next_beams),
            std::begin(next_beams) + last_beam_size,
            std::begin(next_beams) + next_beam_size, compare_beam_prob<Type, IndexType>);

#endif    

        if (t < len_t)
        {
            //copy top beams
            for (int j = 0; j < last_beam_size; j++)
            {
                last_beams[j].entry = next_beams[j];
                last_beams[j].index_as_child = -1;
                last_beams[j].index_as_parent = -1;
                last_beams[j].children_count = 0;
            }

            //delete sequences from the sequence_holder to decrease memory
            for (auto j = beam_width; j < next_beam_size; j++)
            {
                sequence_container.remove(next_beams[j].sequence);
            }

            //check overlapping cases and create lookUp with sorted classes as well
            int look_up_index = 0; 
            for (auto j = 0; j < last_beam_size; j++)
            {
                //if it is not parent node then there is not any need to check
                if (last_beams[j].entry.sequence->isFullyExtended())
                {
                    auto parent_seq=last_beams[j].entry.sequence;
                    int children_count = 0;
                    for (int k = 0; k < last_beam_size; k++)
                    {
                        auto current = last_beams[k].entry.sequence;
                        if (current->prefix == parent_seq)
                        { 
                            child_class_sorter_help[children_count].first = current->value;
                            child_class_sorter_help[children_count].second = k ;
                            ++children_count ;
                        }
                    }

                    if (children_count > 0)
                    {
                
                        //sort by class
                        if(children_count<2){
                            // 
                            if (children_count > 1 && child_class_sorter_help[0].first > child_class_sorter_help[1].first)
                            {
                                std::swap(child_class_sorter_help[0], child_class_sorter_help[1]);
                            }
                        }
                        else
                        {
                            std::sort(std::begin(child_class_sorter_help), std::begin(child_class_sorter_help) + children_count,
                                [](const std::pair<int, int>& left, const std::pair<int, int>& right) {
                                    return left.first < right.first;
                                });
                        }
                        last_beams[j].index_as_parent = look_up_index;
                        last_beams[j].children_count = children_count;

                        for (int l = 0; l < children_count; l++)
                        {
                            
                            int c = child_class_sorter_help[l].first;
                            int k = child_class_sorter_help[l].second;
                            //std::cout << c <<" , " << k << std::endl;
                            last_beams[k].index_as_child = look_up_index;
                            auto seq = last_beams[k].entry.sequence;
                            lookUp[look_up_index].last_c = c;
                            lookUp[look_up_index].node = seq;
                            lookUp[look_up_index].next_beam_index = -1;
                            //next one
                            ++look_up_index;
                        }
                    }//add sorted lookUps

                }
            } //overlap_direction identified to speed up lookUp
            
        }
        
    }//iterate over t
#if defined(NTH_ELEMENT)
    //use sort  for n elements as only nth_element was used
    std::sort(std::begin(next_beams), std::begin(next_beams) + last_beam_size, compare_beam_prob<Type, IndexType>);
#endif
    //store nbest results
    if (nbest_len <= last_beam_size) {
        for (int j = 0; j < nbest_len; j++)
        {
            auto top = next_beams[j];
            auto result_vector = SequenceContainer<IndexType>::getSequence(top.sequence, len_t);
            const auto seq_size = result_vector.size();

            result_prob[j] = top.prob.total;
            result_seq_length[j] = seq_size;
            //copy sequence
            for (auto s = 0; s < seq_size; s++)
            {
                result_sequence[s] = result_vector[s];
            }

            result_sequence += inc_res_seq;

        }
    }
    else
    {
        for (int j = 0; j < nbest_len; j++)
        {
            result_prob[j] = negative_infinity<Type>();
            result_seq_length[j] = 0;;
        }
    }
    return;
}

template<typename Type, typename IndexType = int>
void
beamSearch_(const NDArray& logit, const NDArray& sequence_length, NDArray& result_sequences, NDArray& result_probs, NDArray& result_sequences_length, int blank_index, int beam_width, int nbest_len, bool normalize_logits )
{

    const auto shapes = logit.shapeOf();
    const auto strides = logit.stridesOf();
    const auto rank = logit.rankOf();

    const IndexType* len_t_ptr = nullptr;
    uint64_t element_stride_t = 1;

    //checks before
    if (rank < 2) return;
    auto batch_len = rank > 2 ? shapes[0] : 1;
    auto max_len_t = shapes[rank - 2];
    auto len_c = shapes[rank - 1];

    if (len_c < 1 || max_len_t < 1) return;
    //defaulting blankIndex to the last class if its incorrect or -1
    if (blank_index > len_c || blank_index < 0) blank_index = static_cast<int>(len_c) - 1;
    if (sequence_length.rankOf() == 1 && sequence_length.shapeOf()[0] == batch_len)
    {
        len_t_ptr = sequence_length.bufferAsT<IndexType>();
        element_stride_t = sequence_length.stridesOf()[0];
    }

    //strides
    auto batch_stride = rank > 2 ? strides[0] : 0;
    auto inc_p = strides[rank - 2];
    auto element_stride = logit.stridesOf()[rank - 1];

    auto logits_ptr = logit.bufferAsT<Type>();

#if defined(ASSERT_INNER)
    //result_probs should be [batch_len, nbest_len]
    assert(result_probs.ews() == 1 && result_probs.rankOf() == 2 && result_probs.shapeOf()[0] == batch_len && result_probs.shapeOf()[1] == nbest_len);
    //result sequence should be [batch_len, nbest_len,  max_len_t]
    assert(result_sequences.ews() == 1 && result_sequences.rankOf() == 3 && result_sequences.shapeOf()[0] == batch_len && result_sequences.shapeOf()[1] == nbest_len
            && result_sequences.shapeOf()[2] == max_len_t);
#endif
    auto result_seq_ptr = result_sequences.bufferAsT<IndexType>();
    auto result_probs_ptr = result_probs.bufferAsT<Type>();
    auto result_seq_length_ptr = result_sequences_length.bufferAsT<IndexType>();

    const auto  batch_stride_res = result_sequences.stridesOf()[0];
    const auto  inc_res = result_sequences.stridesOf()[1];
    const auto  batch_stride_res_prob = result_probs.stridesOf()[0];
    const auto  batch_stride_res_seq_length = result_sequences_length.stridesOf()[0];
    auto func = [max_len_t, len_c, batch_stride, inc_p, element_stride, element_stride_t, logits_ptr, len_t_ptr, blank_index, beam_width, normalize_logits,
        nbest_len, result_seq_ptr, result_seq_length_ptr, result_probs_ptr, batch_stride_res, inc_res, batch_stride_res_prob, batch_stride_res_seq_length]
        (uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void
    {

        auto ptr = logits_ptr + start * batch_stride;

        if (element_stride == 1)
        {
            //choose ews one
            for (auto b = start; b < stop; b += increment)
            {
                auto prob_ptr = &(result_probs_ptr[b * batch_stride_res_prob]);
                auto seq_length_ptr = &(result_seq_length_ptr[b * batch_stride_res_seq_length]);
                auto seq_ptr = &(result_seq_ptr[b * batch_stride_res]);

                auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
                inner_beam_search<false, Type, IndexType>(ptr, inc_p, seq_ptr, inc_res, max_len_t, prob_ptr, seq_length_ptr, len_t, len_c, blank_index, beam_width, nbest_len, normalize_logits);

                ptr += batch_stride;

            }
        }
        else
        {
            // element with stride case 
            for (auto b = start; b < stop; b += increment)
            {
                auto prob_ptr = &(result_probs_ptr[b * batch_stride_res_prob]);
                auto seq_length_ptr = &(result_seq_length_ptr[b * batch_stride_res_seq_length]);
                auto seq_ptr = &(result_seq_ptr[b * batch_stride_res]);

                auto len_t = len_t_ptr ? len_t_ptr[b * element_stride_t] : max_len_t;
                inner_beam_search<false, Type, IndexType>(ptr, inc_p, seq_ptr, inc_res, max_len_t, prob_ptr, seq_length_ptr, len_t, len_c, blank_index, beam_width, nbest_len, normalize_logits, element_stride);

                ptr += batch_stride;
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, batch_len, 1);
    return;
}

void beamSearch(const NDArray& logit, const NDArray& sequence_length, NDArray& result_sequences, NDArray& result_probs, NDArray& result_sequences_length, int blank_index, int beam_width , int nbest_len, bool normalize_logits = true){

    BUILD_DOUBLE_SELECTOR(logit.dataType(), result_sequences.dataType(), beamSearch_, (logit, sequence_length, result_sequences, result_probs, result_sequences_length, blank_index, beam_width , nbest_len, normalize_logits), FLOAT_TYPES, INDEXING_TYPES);
}


BUILD_DOUBLE_TEMPLATE(template void beamSearch_, (const NDArray& logit, const NDArray& sequence_length, NDArray& result_sequences, NDArray& result_probs, NDArray& result_sequences_length, int blank_index, int beam_width , int nbest_len, bool normalize_logits), FLOAT_TYPES, INDEXING_TYPES);

}}}
