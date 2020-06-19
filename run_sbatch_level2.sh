#!/bin/bash

# set up icerec environment in Asterix before you run this script
#		sh /home/rkoirala/icerecdist/V05-02-00/build/env-shell.sh

# To Run: 
#       ./run_sbatch_level2.sh 10410

#$1 = 10410

# Copied from: https://www.cyberciti.biz/faq/unix-howto-read-line-by-line-from-file/

file="/data/icet0/rkoirala/CORSIKA/sbatch_level2_"$1".submit"

while IFS= read -r job
do
    # display $line or do somthing with $line
    $job #This does the magic.
    printf '%s\n' "$job"
done <"$file"
