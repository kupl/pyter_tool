#! /bin/bash

for project_folder in $(find /pyter/BugsInPy/benchmark -mindepth 1 -maxdepth 1 -print)
do
    if [[ ${project_folder} == *"${1}"* ]]; then 
        cd ${project_folder}

        #if [ -f "${project_folder}/pyfix/all.json" ]; then
        #    continue
        #fi

        if [ ! -d "${project_folder}/pyter" ]; then
            mkdir pyter
        fi




        python /pyter/pyter_tool/my_tool/extract_neg.py --bench="/pyter/pyter_tool/bugsinpy" --nopos=""
        python /pyter/pyter_tool/my_tool/extract_pos.py --bench="/pyter/pyter_tool/bugsinpy" --nopos=""
        #if [ ! -f "${project_folder}/pyfix/all.json" ]; then
        #python /home/wonseok/pyfix/my_tool/extract_all.py --project ${1}
        #fi
    fi
done



