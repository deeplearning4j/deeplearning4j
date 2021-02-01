'''
@author : Abdelrauf rauf@konduit.ai
'''

#  /* ******************************************************************************
#   * Copyright (c) 2021 Deeplearning4j Contributors
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

import json


def get_compressed_indices_list(set_a):
    '''Get compressed list from set '''
    new_list = sorted(list(set_a)) 
    for i in range(len(new_list)-1,0,-1):
        new_list[i] = new_list[i] - new_list[i-1] 
    return new_list

def intersect_compressed_sorted_list(list1, list2):
    len_1 = len(list1)
    len_2 = len(list2)
    intersected = []
    last_1 ,last_2,i,j = 0,0,0,0
    while i<len_1 and j<len_2:
        real_i = last_1 + list1[i]
        real_j = last_2 + list2[j]
       
        #move first if it is smaller
        if real_i<real_j:
            last_1 = real_i
            i+=1
        elif real_i>real_j:
            last_2 = real_j
            j+=1
        else:
            #intersected
            i+=1
            j+=1
            intersected.append(real_i)
            last_1 = real_i
            last_2 = real_j
    return intersected


class InvertedIndex:
    ''' InvertedIndex for the auto_vect generated invert_index json format '''

    def __init__(self, file_name):
        with open(file_name,"r") as ifx:
            self.index_obj = json.load(ifx)


    def get_all_index(self, entry_name, predicate ):
        '''
         Parameters:
         entry_name  {messages,files,function}
         predicate  function
         Returns:
         list:   list of indexes
        '''
        return [idx for idx,x in enumerate(self.index_obj[entry_name]) if predicate(x)]

    def get_all_index_value(self, entry_name, predicate ):
        '''
         Parameters:
         entry_name  {messages,files,function}
         predicate  function
         Returns:
         list:   list of indexes  ,values
        '''
        return [(idx, x) for idx,x in enumerate(self.index_obj[entry_name]) if predicate(x)]

    def get_function_index(self, predicate = lambda x: True):
        return self.get_all_index('functions', predicate)

    def get_msg_index(self, predicate = lambda x: True):
        return self.get_all_index('messages', predicate)

    def get_file_index(self, predicate = lambda x: True):
        return self.get_all_index('files', predicate)



    def get_msg_postings(self, index ):
        '''
         Gets postings for the given message   
         Parameters:
         index   message index
         Returns:
         [[file index , line position , [ functions ]]]:  list of file index  line position and and compressed functions for the given message  
        '''   
        key =  str(index)
        if not key in self.index_obj['msg_entries']:
            return []
        return self.index_obj['msg_entries'][key]



    def intersect_postings(self, posting1, compressed_sorted_functions , sorted_files = None):
        '''
         Intersects postings with the given functions and sorted_files
         Parameters:
         posting1 postings. posting is [[file_id1,line, [compressed_functions]],..]
         compressed_sorted_functions compressed sorted function index to be intersected
         sorted_files  sorted index of files to be Intersected with the result [ default is None]
         Returns:
         filtered uncompressed posting
        '''
        new_postings = []
        if sorted_files is not None:
            i,j = 0,0
            len_1 = len(posting1)
            len_2 = len(sorted_files)
            while i<len_1 and j <len_2:
                file_1 = posting1[i][0]
                file_2 = sorted_files[j]
                if file_1<file_2:
                    i+=1
                elif file_1>file_2:
                    j+=1
                else:
                    new_postings.append(posting1[i])
                    i+=1
                    #dont increase sorted_files in this case
                    #j=+1

        input_p =new_postings if sorted_files is not None else posting1
        #search and intersect all functions
        new_list = []
        for p in input_p:
            px = intersect_compressed_sorted_list(compressed_sorted_functions,p[2])
            if len(px)>0:
                new_list.append([p[0],p[1],px])
        return new_list  

   
    def get_results_for_msg(self,  msg_index, functions, sorted_files = None):
        '''
         Return  filtered posting for the given msgs index
         Parameters:
         msg_index  message index
         functions   function index list
         sorted_files   intersects with sorted_files also (default: None)
         Returns:
         filtered uncompressed posting for msg index  [ [doc, line, [ function index]] ]
        '''  
        result = []
        compressed = get_compressed_indices_list(functions)
        ix = self.intersect_postings(self.get_msg_postings(msg_index) , compressed, sorted_files)
        if len(ix)>0:
            result.append(ix)
        return result

    def get_results_for_msg_grouped_by_func(self,  msg_index, functions, sorted_files = None):
        '''
         Return  {functions: set((doc_id, pos))} for the given msg index
         Parameters:
         msg_index  message index
         functions   function index list
         sorted_files   intersects with sorted_files also (default: None)
         Returns:
         {functions: set((doc_id, pos))}
        '''  
        result = {}
        compressed = get_compressed_indices_list(functions)
        ix = self.intersect_postings(self.get_msg_postings(msg_index) , compressed, sorted_files)
        for t in ix:
                for f in t[2]:
                    if f in result:
                        result[f].add((t[0],t[1]))
                    else:
                        result[f] = set()
                        result[f].add((t[0],t[1]))
        return result

'''
Example:

import re
rfile='vecmiss_fsave_inverted_index.json'
import inverted_index as helper
ind_obj = helper.InvertedIndex(rfile) 
reg_ops = re.compile(r'simdOps')
reg_msg = re.compile(r'success')
simdOps = ind_obj.get_function_index(lambda x : reg_ops.search(x) )
msgs    = ind_obj.get_msg_index(lambda x: reg_msg.search(x))
files   = ind_obj.get_file_index(lambda x:  'cublas' in x)
res     = ind_obj.get_results_for_msg(msgs[0],simdOps)

'''