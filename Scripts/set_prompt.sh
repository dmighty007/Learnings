#!/bin/bash
# declaring array list and index iterator
declare -a array=()
i=0

# reading file in row mode, insert each line into array
while IFS= read -r line; do
    array[i]=$line
    let "i++"
    # reading from file path
done < "/home/dibya/Soft/Scripts/posh_themes"
#n=${#array[@]}
#for ((i = 0; i < n; i++)); do
#  echo "${array[$i]}"
#done
current=${array[$1]}
echo $current
sed  "s/custom_pos_theme/$current/g" /home/dibya/Soft/Scripts/.bashrc > /home/dibya/.bashrc
exec bash

