#!/bin/bash
while read line; do    
    echo "Fetching:" $line
    sudo aws s3 cp s3://pd-dev-shotgun-output-ohio/sra/PRJNA531273/"$line"/pipeline_output/metaphlan2/"$line"_bugs_list_relab.tsv ~/data/adith/PRJNA531273    
done < ~/data/adith/indian_crc_sample_id.txt

