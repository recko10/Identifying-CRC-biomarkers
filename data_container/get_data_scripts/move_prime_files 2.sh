#!/bin/bash
while read line; do    
    echo "Fetching:" $line
    sudo aws s3 cp s3://pd-dev-shotgun-output-ohio/sra/DRA006684/"$line"/pipeline_output/metaphlan2/"$line"_bugs_list_relab.tsv ~/data/adith/DRA006684_bugs    
done < ~/data/adith/japanese_sample_id.txt

