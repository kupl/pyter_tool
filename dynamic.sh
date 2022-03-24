#! /bin/bash

for project_folder in $(find /pyter/benchmark -mindepth 1 -maxdepth 1 -print)
do
    if [[ ${project_folder} == *"${1}"* ]]; then 
        cd ${project_folder}

        #if [ -f "${project_folder}/pyfix/all.json" ]; then
        #    continue
        #fi

        if [ ! -d "${project_folder}/pyfix" ]; then
            mkdir pyfix
        fi

        #pip install --upgrade pip

        if [[ ${project_folder} == *"rich"* ]]; then
            poetry run pip install -e /pyter/pyter_tool/pyannotate
            poetry run pip install pytest-timeouts

            poetry run python /pyter/pyter_tool/my_tool/extract_neg.py --bench="/pyter/pyter_tool" --nopos=""
            poetry run python /pyter/pyter_tool/my_tool/extract_pos.py --bench="/pyter/pyter_tool" --nopos=""

            continue
        fi

        if [[ ${project_folder} == *"kivy"* ]]; then
            pip install -e /pyter/pyter_tool/pyannotate
            pip install pytest-timeouts

            python /pyter/pyter_tool/my_tool/extract_neg_kivy.py
            python /pyter/pyter_tool/my_tool/extract_pos_kivy.py

            continue
        fi

        pip install -e /pyter/pyter_tool/pyannotate
        pip install pytest-timeouts

        python /pyter/pyter_tool/my_tool/extract_neg.py --bench="/pyter/pyter_tool" --nopos=""
        python /pyter/pyter_tool/my_tool/extract_pos.py --bench="/pyter/pyter_tool" --nopos=""
        #if [ ! -f "${project_folder}/pyfix/all.json" ]; then
        #python /home/wonseok/pyfix/my_tool/extract_all.py --project ${1}
        #fi
    fi
done


