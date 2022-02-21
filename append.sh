file_name=""

function append_args {
   if [ -n "$1"  ]; then
       file_name="${file_name}-$1"
   fi
}
for var in "$@"
do
  append_args "${var}"
done

# Get rid of the first character
file_name_2="${file_name:1:${#file_name}}"
echo "${file_name_2}"