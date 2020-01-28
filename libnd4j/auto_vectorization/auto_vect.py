'''
@author : Abdelrauf rauf@konduit.ai
'''
import re
import sys
import os
import subprocess
import fnmatch
import json
import gzip
try:
    from bigGzipJson import json_gzip_extract_objects
except ImportError:
    pass
from pathlib import Path
from multiprocessing import  Pool, Manager ,cpu_count
import traceback
import html

mtch = re.compile(r"[^/]*([^:]+)\:(\d+)\:(\d+)\:(.*)")
replace_msg = re.compile(r"(\d+)?\.?(\d+)?_?\d+\.?(\d+)?")
progress_msg = re.compile(r"\s{0,4}\[\s{0,2}\d+\%\]")
file_dir_strip = str(Path(os.getcwd()))
pp_index = file_dir_strip.rfind("libnd4j")
if pp_index>=0:
    file_dir_strip =file_dir_strip[:pp_index+len("libnd4j")]
BASE_URL = "https://github.com/eclipse/deeplearning4j/tree/master/libnd4j/"
if BASE_URL.endswith("/")==False:
    BASE_URL = BASE_URL + "/"
#print(file_dir_strip)
class info:
    def __repr__(self):
        return str(self.__dict__) 

FSAVE_IGNORE_EXTERNALS = True

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



def get_msg(msg):
    msg = msg.lower().strip() 
    if "note: not vectorized:" in msg:
        msg = replace_msg.sub("_numb",msg.replace("note: not vectorized:",""))
        return( 0, 1, msg.strip()) 
    elif "loop vectorized" in msg: 
        return (1, 0, None)
    # elif msg.startswith("missed")==False:
    #     msg = replace_msg.sub("_numb",msg)
    #     return( 0, 0, msg.strip())         
    return None


 

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
        elif success==False and "not vectorized:" in msg:
            #reduce this msg
            msg = msg.replace("not vectorized:","")
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

    def add(self, line_pos, msg_x):
        v = self.add_line(line_pos)
        if msg_x is not None:
                v.optimized += msg_x[0]
                v.missed += msg_x[1]
                self.total_opted += msg_x[0]
                self.total_missed += msg_x[1]
                if msg_x[2] is not None:
                    v.miss_details.add(msg_x[2])
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
            if  file_dir_strip in file_name:
                file_name = file_name.replace(file_dir_strip,'./')
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
        generate_report(wr_fname,info_ ,only_body = False, unique_id_prefix = uniq_ind,fsave_format = True, function_list= func_list)
        print(" consumer {0} saved output into {1}".format(c_index,wr_fname))
    except Exception as e:
        print(traceback.format_exc())

            

def obtain_info_from(input_):
    info_ = dict()
    for line in input_:
        x = mtch.match(line)
        external_source = True
        if x:
            file_name =x.group(1).strip()
            if  file_dir_strip in file_name:
                file_name = file_name.replace(file_dir_strip,'')
                external_source = False
            line_number = int(x.group(2))
            msg = x.group(4).lower()
            msg = msg.replace(file_dir_strip,'./')
            msg_x = get_msg(msg)
            if msg_x is None:
                continue
            if file_name in info_:
                #ignore col_number
                info_[file_name].add(line_number,msg_x)
            else:
                #print("{0} {1}".format(file_name,external_source))
                info_[file_name] = File_Info().add(line_number,msg_x)
                info_[file_name].external = external_source
        elif progress_msg.match(line):
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
            inner_str+='<div><div><a>{0}</a></div><div>{1}</div><div>{2}</div><input type="checkbox" id="c{3}{4}"><label for="c{3}{4}"></label><ul>'.format(
            fk,fv.optimized,fv.missed,unique_id_prefix,inc_id)
        else:    
            inner_str+='<div><div><a href=".{0}#L{1}">{1}</a></div><div>{2}</div><div>{3}</div><input type="checkbox" id="c{4}{5}"><label for="c{4}{5}"></label><ul>'.format(
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

def generate_report(output_name,info_ ,only_body = False, unique_id_prefix='',fsave_format = False , function_list = None):
    '''
      Generate Auto-Vectorization Report in html format
    '''

    temp_str =''    
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
 



def main():
    if "--fsave" in sys.argv:
        json_gz_list = internal_glob(".","*.json.gz")
        fsave_report_launch(json_gz_list)
        return        
    
    file_info = obtain_info_from(sys.stdin)
    if len(file_info)>0:
        #print(file_info)
        print("---generating vectorization html report--")
        generate_report("vecmiss.html", file_info)
    else:
        # lets check if we got fsave files
        json_gz_list = internal_glob(".","*.json.gz")
        fsave_report_launch(json_gz_list)




if __name__ == '__main__':
    main()     
