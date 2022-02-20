var_1="hello"
function append_args {
   if [ "$1" != '' ]; then
       var_1="${var_1}_$1"
   fi
}

append_args "hello2"
append_args ""
echo "$var_1"