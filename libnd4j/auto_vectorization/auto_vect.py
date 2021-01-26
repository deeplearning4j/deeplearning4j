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

import argparse
import sys
import re
import os
import subprocess
import fnmatch
import json
import gzip
import argparse

try:
    from bigGzipJson import json_gzip_extract_objects
except ImportError:
    pass
from pathlib import Path
from multiprocessing import  Pool, Manager ,cpu_count
import traceback
import html

# compiler_name :[ (check_operation, version, name_entry), ..]
# positions playes role as checking will stop if it finds non empty entry
STDIN_COMPILER_ENTRY = { 'gcc' : [('<','9','gcc_old')], 'g++' : [('<','9','gcc_old'), ('t','_','')],'nc++' :[('t','_', 'ncxx')] }

#  FSAVE_SUPPORT compiler_name : (check_operation, version ) True or False
# if you want to make it false for all just put 'f' and 't' for true case
FSAVE_SUPPORT = { 'gcc' : ('>=','9'),  'g++' : ('>=','9'), 'nc++' : ('f','_')}

stdin_parser    = None
HAS_FSAVE       = False

FALLBACK_TO_FSAVE_FILES  = True
FSAVE_INVERTED_INDEX     = False

number_replace        = re.compile(r"(\d+)?\.?(\d+)?_?\d+\.?(\d+)?")
cmake_build_progress  = re.compile(r"\s{0,4}\[\s{0,2}\d+\%\]")

internal_match = 'deeplearning4j'+os.path.sep+'libnd4j'+os.path.sep
internal_match_replace = "./"

BASE_URL = ''

FSAVE_IGNORE_EXTERNALS = True
FSAVE_SHOW_SUCCESSFULS = True

def general_stdin_parser(std_success_msg, std_fail_msg , std_line_regex_str):
    '''
    General Parser from success and error message and line regex extractor
    Parameters:
    std_line_regex_str: it should match group(1) to file, group(2) to line_number and group(3) to message
    '''
    matcher = re.compile(std_line_regex_str)
    def local_parser(line, helper_storage):
        #for generic we will parsing stdin input line by line
        #so we dont need any storage
        parse_info = ParseInfo()
        x = matcher.match(line)
        parse_info.external_source = True
        if x:
            #print(line)
            file_name =x.group(1).strip()
            ppos = file_name.find(internal_match)
            if  ppos>=0:
                file_name = internal_match_replace + file_name[ppos+len(internal_match):]
                parse_info.external_source = False
            parse_info.line_pos = int(x.group(2))
            msg = x.group(3).lower().strip()
            parse_info.file_name = file_name
            if std_fail_msg in msg:
                msg = number_replace.sub("_numb",msg.replace(std_fail_msg,"fail:"))
                parse_info.msg =  msg.strip()
                parse_info.miss  = 1
                parse_info.success = 0
                #print(parse_info.__dict__)
                return parse_info
            elif std_success_msg in msg: 
                parse_info.msg =  msg.strip()
                parse_info.miss  = 0
                parse_info.success = 1
                #print(parse_info.__dict__)
                return parse_info
        return None

    return local_parser


# entry:  parser list for compilers that can parse compilers output and return Parse_info
# the signature of the parser function   is `Parse_info parser_function_name_(line, helper_storage)`
# Please note that Parse_info members should be the same as we defined in `general_stdin_parser local_parser`
# the line is a compiler output. helper_storage is a dict and can be used as a state storage 
# to parse multi-line and et cetera, as parser called for each line.
STDIN_PARSERS = { 'gcc_old' : general_stdin_parser('loop vectorized', 'note: not vectorized:', r"[^/]*([^:]+)\:(\d+)\:\d+\:(.*)" ),
                  'ncxx'    : general_stdin_parser("vectorized loop", "unvectorized loop",     r'[^/]+([^,]+)\,\s*line\s*(\d+)\:(.*)')
}



def version_check( version1, version2, op='>='):
    op_list = {"<": (lambda x,y: x<y), "==": (lambda x,y: x==y),
          "<=": (lambda x,y: x<y), "!=": (lambda x,y: x!=y),
               ">": (lambda x,y: x>y), ">=": (lambda x,y: x>=y),
               'f': (lambda x,y:   False),'t': (lambda x,y: True)
               
          }
    return op_list[op](version1.split('.'),version2.split('.'))


def init_global_options(args):
    global stdin_parser
    global HAS_FSAVE
    global BASE_URL
    global FSAVE_INVERTED_INDEX

    FSAVE_INVERTED_INDEX = args.inverted_index
    BASE_URL  = args.base_url
    if BASE_URL.endswith("/")==False:
        BASE_URL = BASE_URL + "/"

    entry_name = ''

    if args.compiler in STDIN_COMPILER_ENTRY:
        for x in STDIN_COMPILER_ENTRY[args.compiler]:
            ret = version_check(args.compiler_version,x[1],x[0])
            if ret == True:
                entry_name = x[2]
                break
    
    if len(entry_name)>0:          
        stdin_parser    = STDIN_PARSERS[entry_name]
    if args.compiler in FSAVE_SUPPORT:
        x = FSAVE_SUPPORT[args.compiler]
        HAS_FSAVE = version_check(args.compiler_version,x[1],x[0])

class info:
    def __repr__(self):
        return str(self.__dict__) 



def get_cxx_filt_result(strx):
    if len(strx)<1:
        return ""
    res = subprocess.Popen(["c++filt","-i", strx], stdout=subprocess.PIPE).communicate()[0]
    res =res.decode('utf-8')
    #replace some long names to reduce size
    res = res.replace("unsigned long long", "uLL")
    res = res.replace("unsigned long int","uL")
    res = res.replace("unsigned long", "uL")
    res = res.replace("unsigned int", "ui")
    res = res.replace("unsigned char", "uchar")
    res = res.replace("unsigned short", "ushort")    
    res = res.replace("long long", "LL")
    res = res.replace(", ",",")
    return res.strip()

 
def internal_glob(dir, match):
    listx = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, match):
            listx.append(os.path.join(root, filename))
    return listx

def get_obj_json_gz(filename):
    with gzip.GzipFile(filename, 'r') as f:
        return json.loads(f.read().decode('utf-8'))[-1]


class ParseInfo:
    pass

 
class File_Info:
    '''
    Holds information about vectorized and miss vectorized lines for one file
    '''
    
    def __init__(self):
        self.infos = {}
        self.total_opted =0
        self.total_missed = 0
        self.external = False


    def add_line(self, line_pos):
        if line_pos not in self.infos:
            v = info()
            v.optimized = 0
            v.missed = 0
            v.miss_details = set()
            self.infos[line_pos] = v
            return v
        else:
            return self.infos[line_pos]


    def add_line_fsave(self, line_pos):
        if line_pos not in self.infos:
            v = info()
            v.optimized = 0
            v.missed = 0
            v.miss_details2 = dict()
            self.infos[line_pos] = v
            return v
        else:
            return self.infos[line_pos]            
               


    def add_fsave(self, line_pos,success, msg, function ,inline_fns=''):
        v = self.add_line_fsave(line_pos)
        if success and "loop vectorized" in msg:
            v.optimized +=1
            self.total_opted +=1
            if FSAVE_SHOW_SUCCESSFULS==True:
                if "success" in v.miss_details2:
                    ls = v.miss_details2.get("success") 
                    ls.add(function)
                else:
                    ls =set()
                    v.miss_details2["success"]=ls
                    ls.add(function)
        elif success==False and "not vectorized:" in msg:
            #reduce this msg
            msg = msg.replace("not vectorized:","").strip()
            v.missed +=1
            self.total_missed +=1
            msg = sys.intern(msg)
            if msg in v.miss_details2:
                ls = v.miss_details2.get(msg) 
                ls.add(function)
            else:
                ls =set()
                v.miss_details2[msg]=ls
                ls.add(function)
        return self

    def add(self, line_pos, msg, success, missed):
        v = self.add_line(line_pos)
        if msg is not None:
                v.optimized += success
                v.missed += missed
                self.total_opted  += success
                self.total_missed += missed
                if msg is not None:
                    v.miss_details.add(msg)
        return self
    

    def __repr__(self):
        return str(self.__dict__)                    
                    



def process_gzip_json_mp(args):
    process_gzip_json_new(*args)

def process_gzip_json_new(json_gz_fname,list_Queue):
    gz_name = Path(json_gz_fname).stem
    #print("::--open and process {0}".format(gz_name))
    queue_count = len(list_Queue)
    #print(queue_count)
    q = list_Queue[0]
    old_fname = ''
    total_c = 0
    for x in json_gzip_extract_objects(json_gz_fname,'message','vectorized'):
        external_source = True
        if len(x['message'])>0 and 'location' in x:
            line = int(x['location']['line'])
            file_name = x['location']['file'].strip()
            ppos = file_name.find(internal_match)
            if  ppos>=0:
                file_name = internal_match_replace + file_name[ppos+len(internal_match):]
                external_source = False
            msg = x['message'][0]
            success = x['kind'] == 'success'
            func = ''  if 'function' not in x else x['function']

            if file_name!=old_fname:
                #send our info to the right consumer
                queue_ind = hash(file_name) % queue_count
                #print("quen index {0}".format(queue_ind))
                q =list_Queue[queue_ind]
                old_fname = file_name
            total_c +=1
            #print("pp {0} {1}".format(q,(file_name,line,success, msg, func,external_source )))
            if FSAVE_IGNORE_EXTERNALS==True and external_source == True:
                continue
            q.put((file_name,line,success, msg, func,external_source ))
    print("::finished {0:60s} :{1:8d}".format(gz_name,total_c))

def consume_processed_mp(args):
    return consume_processed_new(*args)



def consume_processed_new(list_Queue , c_index):
    
    info_ = dict()
    func_list = dict()
    last_func_index = 0 
    q  = list_Queue[c_index]
    print("::consumer {0}".format(c_index))
    total_c = 0
    r_c = 0
    while True:
        #print("try to get new from {0}".format(index))
        obj = q.get()
        #print("cc {0} {1}".format(q,obj))
        if obj==None:
            break #we received the end
        file_name,line,success, msg, func, external_source = obj
        try:
            #get function index
            func_index = -1
            if func in func_list:
                func_index = func_list[func]
            else:
                func_list[func] = last_func_index
                func_index = last_func_index
                last_func_index +=1
 
            if file_name in info_: 
                info_[file_name].add_fsave(line, success, msg, func_index)
            else: 
                info_[file_name] = File_Info().add_fsave(line, success, msg, func_index)
                info_[file_name].external = external_source
            total_c +=1 
            if total_c - r_c >10000:
                r_c = total_c
                print("::consumer {0:2d} :{1:10d}".format(c_index,total_c))
        except Exception as e:
            print(traceback.format_exc())
            break

    print("::consumer {0:2d} :{1:10d}".format(c_index,total_c))
    #write to temp file 
    wr_fname= "vecmiss_fsave{0}.html".format(str(c_index) if len(list_Queue)>1 else '')
    print("generate report for consumer {0} {1}".format(c_index,len(info_)))
    try:
        uniq_ind = str(c_index)+'_' if len(list_Queue)>1  else ''
        wr = generate_report(wr_fname,info_ ,only_body = False, unique_id_prefix = uniq_ind,fsave_format = True, function_list= func_list)
        print(" consumer {0} saved output into {1}".format(c_index, wr))
    except Exception as e:
        print(traceback.format_exc())

            

def obtain_info_from(input_):
    info_ = dict()
    parser_storage = dict() #can be used for parsing multi-lines
    if HAS_FSAVE ==True or stdin_parser is None:
        #just print progress
        for line in input_:
            if cmake_build_progress.match(line):
                #actually we redirect only, stderr so this should not happen
                print("__"+line.strip())
            elif "error" in line or "Error" in line:
                print("****"+line.strip())
        return info_
    for line in input_:
        x = stdin_parser(line, parser_storage)
        if x is not None:
            if x.file_name in info_:
                #ignore col_number
                info_[x.file_name].add(x.line_pos, x.msg, x.success, x.miss)
                info_[x.file_name].external = x.external_source
            else:
                info_[x.file_name] = File_Info().add(x.line_pos, x.msg, x.success, x.miss)
                info_[x.file_name].external = x.external_source
        elif cmake_build_progress.match(line):
            #actually we redirect only, stderr so this should not happen
            print("__"+line.strip())
        elif "error" in line or "Error" in line:
            print("****"+line.strip())
    return info_


           
def custom_style(fsave):
    st = '''<style>a{color:blue;}
a:link{text-decoration:none}a:visited{text-decoration:none}a:hover{cursor:pointer;text-decoration:underline}
a:active{text-decoration:underline}
.f.ext{display:none} 
.f{color:#000;display:flex;overflow:hidden;justify-content:space-between;flex-wrap:wrap;align-items:baseline;width:100%}
.f>div{min-width:10%}.f>div:first-child{min-width:70%;text-overflow:ellipsis}
.f:nth-of-type(even){background-color:#f5f5f5}
.f>div.g{flex:0 0 100%}.f>div:nth-child(2){font-weight:600;color:green}
.f>div:nth-child(3){font-weight:600;color:red}
.f>div:nth-child(2)::after{content:' ✓';color:green}.f>div:nth-child(3)::after{content:' -';color:red}
.f>div.g>div>div:nth-child(2){font-weight:600;color:green}
.f>div.g>div>div:nth-child(3){font-weight:600;color:red}
.f>div.g>div>div:nth-child(2)::after{content:' ✓';color:green}
.f>div.g>div>div:nth-child(3)::after{content:' -';color:red}
.f>div.g>div{display:flex;justify-content:space-between;flex-wrap:wrap;align-items:baseline}
.f>div.g>div>div{min-width:10%;text-align:left}
.g>div:nth-of-type(even){background-color:#ede6fa}
.f>div.g>div>ul{flex:0 0 100%}input[type=checkbox]{opacity:0;display:none}label{cursor:pointer}
.f>label{color:red}input[type=checkbox]~.g{display:none}input[type=checkbox]:checked~.g{display:block}
input[type=checkbox]~ul{display:none}
input[type=checkbox]:checked~ul{display:block}input[type=checkbox]+label::after{content:"⇲";display:block}
input[type=checkbox]:checked+label::after{content:"⇱";display:block}

'''
    if fsave==True:
        st+='''.modal{display:none;height:100%;background-color:#144F84;color:#fff;opacity:.93;left:0;position:fixed;top:0;width:100%}
        .modal.open{display:flex;flex-direction:column}.modal__header{height:auto;font-size:large;padding:10px;background-color:#000;color:#fff}
        .modal__footer{height:auto;font-size:medium;background-color:#000}
        .modal__content{height:100%;display:flex;flex-direction:column;padding:20px;overflow-y:auto}
        .modal_close{cursor:pointer;float:right}li{cursor:pointer}
        '''
    return st + '''</style>'''

def header(fsave=False):
    strx ='<!DOCTYPE html>\n<html>\n<head>\n<meta charset="UTF-8">\n<title>Auto-Vectorization</title>\n'
    strx +='<base id="base_id" href="{0}" target="_blank" >'.format(BASE_URL)
    strx +=custom_style(fsave)  
    strx +='\n</head>\n<body>\n'
    return strx

def footer():
    return '\n</body></html>'


  
def get_compressed_indices_list(set_a):
    new_list = sorted(list(set_a)) 
    for i in range(len(new_list)-1,0,-1):
        new_list[i] = new_list[i] - new_list[i-1] 
    return new_list

def get_compressed_indices(set_a):
    a_len = len(set_a)
    if a_len<=1:
        if a_len<1:
            return ''
        return str(set_a)[1:-1]
    #we sorted and only saved difference 
    # 1,14,15,19 --> 1,13,1,4  10bytes=>8bytes
    list_sorted = sorted(list(set_a))
    last = list_sorted[0]
    str_x = str(list_sorted[0])
    for i in range(1,a_len):
        str_x += ','+str(list_sorted[i]-last)
        last = list_sorted[i]
    return str_x


    


def get_content(k, v,  unique_id_prefix = '', fsave_format=False):
    inner_str=''
    content = ''
    inc_id = 0
    for fk,fv in sorted(v.infos.items()):
        if fsave_format==True:
            inner_str+='<div><div><a>{0}</a></div><div>{1}</div><div>{2}</div><input type="checkbox" id="{3}c{4}"><label for="{3}c{4}"></label><ul>'.format(
            fk,fv.optimized,fv.missed,unique_id_prefix,inc_id)
        else:    
            inner_str+='<div><div><a href=".{0}#L{1}">{1}</a></div><div>{2}</div><div>{3}</div><input type="checkbox" id="{4}c{5}"><label for="{4}c{5}"></label><ul>'.format(
            k,fk,fv.optimized,fv.missed,unique_id_prefix,inc_id)
        inc_id+=1
        if fsave_format==True:
            #
            for dt,df in fv.miss_details2.items():
                #inner_str  +='<li data-fns="{0}">{1}</li>'.format(str(df).replace(", ",",")[1:-1],dt) 
                inner_str  +='<li data-fns="{0}">{1}</li>'.format(get_compressed_indices(df),dt)           
        else:
            for dt in fv.miss_details:
                inner_str+="<li>"+str(dt)+ "</li>"
        inner_str+="</ul></div>\n"

    content += '<div class="f'
    if v.external:
        content += " ext"
    content += '">\n<div>{0}</div><div>{1}</div><div>{2}</div><input type="checkbox" id="i{3}{4}"><label for="i{3}{4}"></label>'.format(
        k,v.total_opted,v.total_missed,unique_id_prefix,inc_id)  
    content += "<div class='g'>"  
    content += inner_str
    content += "</div> </div>\n"
    return content   


def jscript_head():
    return '''
    window.onload = function () {
    var modal = document.getElementsByClassName("modal")[0];
    var modal_close = document.getElementsByClassName("modal_close")[0];
    var content = document.getElementsByClassName("modal__content")[0];
    a_tags = document.getElementsByTagName("a");
    base_href = document.getElementById("base_id").href;
    for(i=0;i<a_tags.length;i++){
        a_tags[i].addEventListener("click", function () {
            var source = event.target || event.srcElement;
            file_src = source.parentElement.parentElement.parentElement.parentElement.children[0].innerText ;
            link = base_href + file_src+'#L'+ source.innerText;
            window.open(link, '_blank');
            
        });
    }
    modal_close.addEventListener("click", function () {
        content.innerHTML = '';
        modal.className = 'modal';
    });
    
    ''' 
def jscipt_end():
    return '''
    tags = document.getElementsByTagName("li");
    function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
    }
    for (i = 0; i < tags.length; i++) {
        tags[i].addEventListener("click", function () {
            var source = event.target || event.srcElement;
            funcs = source.dataset.fns.split(",")
            strx = ''
            //we saved differences,not real indices
            last_ind = 0;
            for (j = 0; j < funcs.length; j++) {
                ind  = last_ind + parseInt(funcs[j]);
                strx += "<p>" + escapeHtml(func_list[ind]) + "</p>";
                last_ind = ind;
            }
            if (strx.length > 0) {
                content.innerHTML = strx;
                modal.className = 'modal open';
            }

        });
    }
 
    };'''

def additional_tags(fsave):
    if fsave==False:
        return ''
    #
    return '''<script type='text/javascript'> 
    var script = document.createElement('script'); script.src =  window.location.href+".js" ;
    document.head.appendChild(script); 
    </script>
    <div class="modal">
        <div class="modal__header">Functions <span class="modal_close">X</span></div>
        <div class="modal__content"></div>
        <div class="modal__footer">========</div>
    </div>
    '''

class Json_reverse:
    pass

def generate_inverted_index(output_name, info_ , function_list ):
    temp_str =''  
    output_name = output_name.replace(".html","_inverted_index")
    rev_index = Json_reverse()
    rev_index.functions =[get_cxx_filt_result(k) for k,v in sorted(function_list.items(), key=lambda x: x[1])]
    rev_index.msg_entries = {}
    message_list =dict()
    rev_index.files = list()
    doc_i = 0
    for doc_name,v in  info_.items():
        for line_pos,info in v.infos.items():
            for msg,func_indices in info.miss_details2.items():
                    #we index msgs here, as previously it was not done
                    msg_index = len(message_list)
                    if msg in message_list:
                        msg_index = message_list[msg]
                    else:
                        message_list[msg] = msg_index
                    ## postings
                    if not msg_index in rev_index.msg_entries:
                        rev_index.msg_entries[msg_index] = list()
                    rev_index.msg_entries[msg_index].append([doc_i,line_pos, get_compressed_indices_list(func_indices)])          
        doc_i = doc_i + 1
        rev_index.files.append(doc_name)
    rev_index.messages = [k for k,v in sorted(message_list.items(), key=lambda x: x[1])]
    with open(output_name+ ".json","w") as f:    
        json.dump(rev_index.__dict__, f)
    return (output_name+ ".json")


    

def generate_report(output_name,info_ ,only_body = False, unique_id_prefix='',fsave_format = False , function_list = None):
    '''
      Generate Auto-Vectorization Report in html format
    '''
    temp_str =''
    if FSAVE_INVERTED_INDEX == True and fsave_format == True:
        return generate_inverted_index(output_name,info_ ,  function_list )

    if fsave_format ==True:
        # we gonna dump function_list as key list sorted by value
        #and use it as jscript array  
        sorted_funcs_by_index = sorted(function_list.items(), key=lambda x: x[1])
        del function_list
        with open(output_name+ ".js","w") as f:
            #temp_str =jscript_head() +'{ "fmaps":['
            temp_str = jscript_head() + "\n var func_list = ["
            for k,v in sorted_funcs_by_index:
                #json.dumps using for escape
                #print(str(v)+str(k))
                temp_str+=json.dumps(get_cxx_filt_result(k))+","
                #reduce write calls
                if len(temp_str)>8192*2:
                    f.write(temp_str)
                    temp_str= ''
            if len(temp_str)>0:
                f.write(temp_str)
            f.write('"-"];'+jscipt_end())
            

    temp_str = ''
    with open(output_name,"w") as f:
        if only_body==False:
            f.write(header(fsave_format))
            f.write(additional_tags(fsave_format))
        nm=0
        for k,v in sorted(info_.items()): # sorted(info_.items(), key=lambda x: x[1].total_opted, reverse=True):
            temp_str += get_content(k,v,unique_id_prefix+str(nm),fsave_format)
            #reduce io write calls
            if len(temp_str)>8192:
                f.write(temp_str)
                temp_str =''
            nm+=1
        if len(temp_str)>0:
            f.write(temp_str)
        if only_body==False:
            f.write(footer())  

    return (output_name, output_name+".js") if fsave_format ==True else  (output_name)



def fsave_report_launch(json_gz_list):

        cpus = cpu_count()
        if cpus>32:
            cpus = 24

        c_count = 1 # 2 i sufficient # if cpus<=1 else min(4,cpus)
        p_count = 3 if cpus<=1 else max(8, cpus - c_count)

        m = Manager()
        #consumer Queues
        list_Queue = [m.Queue() for index in range(0,c_count)]
        with Pool(processes=c_count) as consumers:
            #start consumers
            cs = consumers.map_async(consume_processed_mp,[(list_Queue, index,) for index in range(0,c_count)])
            with Pool(processes=p_count) as processors:
                processors.map(process_gzip_json_mp, [(fname, list_Queue,) for fname in json_gz_list])
                
            #send ends to inform our consumers
            #send ends
            for q in list_Queue:
                q.put(None)
            
            #wait for consumers
            cs.wait()
 

class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, ' error: {0}\n'.format ( message))

def main():
    parser =  ArgumentParser(description='Auto vectorization report')
    parser.add_argument('--fsave', action='store_true', help='looks for json files generated by -fsave-optimization-record flag instead of waiting for the stdin')
    parser.add_argument('--inverted_index', action='store_true', help='generate inverted_index for -fsave-optimization-record in json format')
    parser.add_argument('--base_url', default='https://github.com/eclipse/deeplearning4j/tree/master/libnd4j/', help='url link for source code line view')
    parser.add_argument('--compiler', choices=['gcc','nc++'], default = 'gcc')
    parser.add_argument('--compiler_version',default='')
    args = parser.parse_args()
    init_global_options(args)
    
    if args.fsave:
        json_gz_list = internal_glob(".","*.json.gz")
        fsave_report_launch(json_gz_list)
        return        
    #initialize globals
    
    file_info = obtain_info_from(sys.stdin)

    if HAS_FSAVE==True:
        json_gz_list = internal_glob(".","*.json.gz")
        fsave_report_launch(json_gz_list)  
        return

    if len(file_info)>0:
        #print(file_info)
        print("---generating vectorization html report--")
        generate_report("vecmiss.html", file_info)
    elif FALLBACK_TO_FSAVE_FILES == True:
        # lets check if we got fsave files
        json_gz_list = internal_glob(".","*.json.gz")
        fsave_report_launch(json_gz_list)




if __name__ == '__main__':
    main()     
