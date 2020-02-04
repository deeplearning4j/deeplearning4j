'''
@author : Abdelrauf rauf@konduit.ai
Simple object xtractor form very big json files 
'''

import sys
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


cdef char JSON_1 = b':'
cdef char JSON_2 = b','
cdef char JSON_3 = b'{'
cdef char JSON_4 = b'}'
cdef char JSON_5 = b'['
cdef char JSON_6 = b']'
cdef char QUOTE = b'"'
cdef char ESCAPE = b"\\"
cdef char SPACE = b' '
cdef char TAB = b't'
cdef char CR = b'\r'
cdef char NL = b'\n'
cdef char B = b'\b'
cdef char EMPTY = b'\0'


cdef struct Span:
    int b
    int e 

cdef inline Span read_unquoted(char *text, int start,int end):
    cdef Span sp
    cdef int j = start
    while j < end:
        #if text[j].isspace():
        if text[j] == SPACE or text[j] == NL or text[j] == TAB or text[j] == CR or text[j] == B:
            j += 1
            continue
        if text[j] != QUOTE and text[j] != JSON_1 and text[j] != JSON_2 and text[j] != JSON_3 and text[j] != JSON_4 and text[j] != JSON_5 and text[j] != JSON_6:
            start = j
            j += 1
            while j < end:
                # read till JSON or white space
                if text[j] == SPACE or text[j] == NL or text[j] == TAB or text[j] == CR or text[j] == B:
                    sp.b = start
                    sp.e = j
                    return sp
                elif text[j] == JSON_1 or text[j] == JSON_2 or text[j] == JSON_3 or text[j] == JSON_4 or text[j] == JSON_5 or text[j] == JSON_6:
                    sp.b = start
                    sp.e = j                 
                    return sp
                j += 1
            if j == end-1:
                sp.b = start
                sp.e = end            
                return sp
        break
    sp.b = j
    sp.e = j
    return sp


cdef inline Span read_seq_token(char *text,int  start,int end):
    #read quoted
    #skip white_space
    cdef Span sp    
    cdef int j = start
    cdef char last_char
    cdef char char_x
    while j < end:
        if text[j] == SPACE or text[j] == NL or text[j] == TAB or text[j] == CR or text[j] == B:
            j += 1
            continue
        if text[j] == QUOTE:
            last_char = EMPTY
            #read till another quote
            start = j
            j += 1
            while j < end:
                char_x = text[j]
                if char_x == QUOTE and last_char != ESCAPE:
                    # finished reading
                    sp.b =start
                    sp.e = j+1
                    return sp
                last_char = char_x
                j += 1
            if j == end-1:
                sp.b = start
                sp.e = end
                return sp
        else:
            break
    return read_unquoted(text, j, end)


def tokenizer_spans(utext):
    '''
    we will just return tokenize spans
    '''
    token_spans = []
    last_char = b''
    end_i = len(utext)
    cdef char *text = utext
    i = 0
    cdef Span sp
    while i < end_i:
        sp = read_seq_token(text, i, end_i)
        i = sp.e
        if sp.e > sp.b:
            token_spans.append((sp.b, sp.e))
        if i < end_i:
            #if text[i] in JSON:
            if text[i] == JSON_3 or text[i] == JSON_4 or text[i] == JSON_5 or text[i] == JSON_6 or text[i] == JSON_1 or text[i] == JSON_2:
                token_spans.append((i, i+1))
        i += 1
    return token_spans





cdef class JsonObjXtractor:
    '''
     JsonObjXtractor that utilize cython better
    '''

    cdef Span* token_spans
    cdef size_t size      

    def __cinit__(self, size_t count=4096): 
        self.token_spans = <Span*> PyMem_Malloc(count * sizeof(Span))
        self.size = count
        if not self.token_spans:
            raise MemoryError()


    def __tokenizer_spans(self,utext, length):
        '''
        we will just return token  spans length
        '''
         
        last_char = b''
        end_i = length
        cdef char *text = utext
        cdef int i = 0
        cdef size_t j = 0
        cdef Span sp
        while i < end_i:
            sp = read_seq_token(text, i, end_i)
            i = sp.e
            if sp.e > sp.b:
                self.token_spans[j] = sp
                j+=1
                if j>self.size:
                    #we need to reallocate
                    self.__resize(self.size+self.size//2)                
            if i < end_i:
                #if text[i] in JSON:
                if text[i] == JSON_3 or text[i] == JSON_4 or text[i] == JSON_5 or text[i] == JSON_6 or text[i] == JSON_1 or text[i] == JSON_2:
                    sp.b=i
                    sp.e=i+1
                    self.token_spans[j] = sp
                    j+=1
                    if j>self.size:
                        #we need to reallocate
                        self.__resize(self.size+self.size//2)
            i += 1
        return j



    def try_extract_parent_obj(self, json_bytes, property_name, next_contains_value=b'', debug=False):
        '''
        try_extract_parent_obj(json_text, property_name, next_contains_value='', debug=False):
        make sure that passed variables encoded to bytes with encode('utf-8')
        next_contains_value either direct content or followed by '['
        tries to extract the parent object for given named object
        if the left brace of the parent object is outside of the current buffer
        it will be ignored
        if the right brace is outside of the buffer it will be left to be handled by caller
        '''

        look_for_the_left = True
        parent_left = []
        parent_right = []
        parent_objects = []
        len_next = len(next_contains_value) 
        cdef int ind = 0
        cdef int end
        cdef int last_start = 0
        property_name = b'"'+property_name+b'"'
        cdef int lenx  = self.__tokenizer_spans(json_bytes,len(json_bytes))
        cdef char x
        cdef int i = -1
        cdef Span sp
        while i < lenx-1:
            i += 1
            ind = self.token_spans[i].b
            x = json_bytes[ind]
            #print("-----{0} -- {1} -- {2} ".format(x,parent_left,parent_right))
            if look_for_the_left == False:
                if x == JSON_3:
                    parent_right.append(ind)
                elif x == JSON_4:
                    if len(parent_right) == 0:
                        #we found parent closing brace
                        look_for_the_left = True
                        parent_objects.append((parent_left[-1], ind+1))
                        last_start = ind+1
                        #print("=============found {0}".format(parent_objects))
                        parent_left = []
                        parent_right = []
                    else:
                        parent_right.pop()
                continue
            #search obj
            if look_for_the_left:
                if x == JSON_3:
                    parent_left.append(ind)
                    last_start = ind
                elif x == JSON_4:
                    if len(parent_left) >= 1:
                        #ignore
                        parent_left.pop()

            if x == JSON_1:  # ':'
                #check to see if propertyname
                old_property = EMPTY
                if i > 1:
                    sp = self.token_spans[i-1]
                    old_property = json_bytes[sp.b:sp.e]
                if old_property == property_name:
                    #we found
                    if len(parent_left) < 1:
                        #left brace is outside of the buffer
                        #we have to ignore it
                        #try to increase buffer
                        if debug:
                            print('''left brace of the parent is outside of the buffer and parent is big.
                            it will be ignored 
                            try to choose disambiguous property names if you are looking for small objects''', file=sys.stderr)
                        last_start = ind+1
                        parent_left = []
                        parent_right = []
                        continue
                    else:
                        #print("++++++ look for the right brace")
                        if len_next>0 and i+1 < lenx:
                                i += 1
                                ind = self.token_spans[i].b
                                end = self.token_spans[i].e
                                m = json_bytes[ind]
                                
                                if m == JSON_5:
                                    #print ("----{0} {1}".format(m,JSON_5))
                                    if i+1 < lenx:
                                        i += 1
                                        ind = self.token_spans[i].b
                                        end = self.token_spans[i].e
                                        #print ("----{0}  == {1}".format(next_contains_value,json_bytes[ind:end]))
                                        if len_next <= end-ind and next_contains_value in json_bytes[ind:end]:
                                            look_for_the_left = False
                                            continue
                                elif len_next <= end-ind and  next_contains_value in json_bytes[ind:end]:
                                    look_for_the_left = False
                                    continue

                                #ignore as it does not have that value
                                parent_left = []
                                parent_right = []
                                last_start = ind + 1
                        else:
                            look_for_the_left = False

        # lets return last succesful opened brace as the last
        # or  left brace failure case, safe closed brace
        if len(parent_left)>0:
            return (parent_objects, parent_left[-1])

        return (parent_objects, last_start)



    def __resize(self, size_t new_count):
        cdef Span* mem = <Span*> PyMem_Realloc(self.token_spans, new_count * sizeof(Span))
        if not mem:
            raise MemoryError()
        self.token_spans = mem
        self.size = new_count

    def __dealloc__(self):
        PyMem_Free(self.token_spans) 



import json
import gzip
import sys
DEBUG_LOG = False

def json_gzip_extract_objects(filename, property_name, next_contains_value=''):
    strx = b''
    started= False
    b_next_contains_value = next_contains_value.encode('utf-8')
    b_property_name = property_name.encode('utf-8')
    #print(b_property_name)
    objXt = JsonObjXtractor()
    with gzip.open(filename, 'rb') as f:
        if DEBUG_LOG:
            print("opened {0}".format(filename), file=sys.stderr)
        #instead of reading it as line, I will read it as binary bytes
        is_End = False
        #total = 0
        while is_End==False:
            buffer  = f.read(8192*2)
            
            lenx= len(buffer)
            #total +=lenx
            if lenx<1:
                is_End = True
            else:   
                strx = strx + buffer

            objects , last_index = objXt.try_extract_parent_obj(strx,b_property_name,b_next_contains_value)

            # if b_property_name in strx and b_next_contains_value in strx:
            #     print(strx)
            #     print(objects)
            #     print(last_index)
            #     print("===================================================")

            for start,end in objects:
                yield  json.loads(strx[start:end]) #.decode('utf-8'))


            #remove processed 
            if last_index< len(strx):
                strx = strx[last_index:]
                
            else:
                strx = b''
                #print('----+++')
                
            if(len(strx)>16384*3):
                #buffer to big
                #try to avoid big parents
                if DEBUG_LOG:
                    print("parent object is too big. please, look for better property name", file=sys.stderr)

                break 




