#!/Users/adithreddi/opt/anaconda3/envs/r-bio/bin/Rscript --vanilla

if(!suppressPackageStartupMessages(require("docopt"))){
  stop("You must have the docopt R package installed. Assuming you have
             installed R & Bioconductor (www.bioconductor.org/install), type:\n
             BiocManager::install(\"docopt\") \n
             from your R prompt.")
}

doc <- "curatedMetagenomicData: command-line access to the curatedMetagenomicData R/Bioconductor package
Usage:
  curatedMetagenomicData [--pdata] [--counts] [--dryrun] [<NAME>]...
  curatedMetagenomicData [-pcd] [<NAME>]...
  curatedMetagenomicData -l | --list
  curatedMetagenomicData -h | --help
Options:
  -h --help      Show this. Note that arguments may optionally be quoted.
  -p --pdata     Include metadata in the first rows of the tsv file. This
includes participant data such as age, sex, and disease status, and the number
of sequencing reads. See explanations of pdata rows at
https://github.com/waldronlab/curatedMetagenomicData/blob/master/inst/extdata/template.csv.
  -c --counts    Multiply MetaPhlAn2 and HUMAnN2 relative abundances by read
depth, then round to the nearest integer. If it is not set, return MetaPhlAn2
and HUMAnN2 results as-is.
  -d --dryrun    Show which datasets would be downloaded, then exit normally
without downloading.
  -l --list      List all datasets available then exit.
  <NAME>...      One or more names curatedMetagenomicData datasets (See
reference manual for list of all datasets). Standard Unix globbing is
supported (such as * for wildcard), however in this case NAME argument
MUST BE QUOTED.
The script will produce one tab-separated tsv file per dataset requested.
The output files are the name of the dataset name with '.tsv' appended.
Examples:
  curatedMetagenomicData -p -c LomanNJ_2013.metaphlan_bugs_list.stool
Will produce a file LomanNJ_2013.metaphlan_bugs_list.stool.tsv providing MetaPhlAn2
taxonomic abundance multiplied by read depth, with metadata about study
participants and sequencing information in the first several rows of the file.
  curatedMetagenomicData LomanNJ_2013.metaphlan_bugs_list.stool LomanNJ_2013.pathcoverage.stool
Will return two files, without clinical / sequencing metadata.
  curatedMetagenomicData -d \"*\"                 #Will show all available datasets (don't download)
  curatedMetagenomicData -d \"HMP*\"              #Will show all HMP data products (don't download)
  curatedMetagenomicData -d \"HMP*nasalcavity\"   #Will show all HMP hard palate data products (don't download)
  curatedMetagenomicData -d \"HMP*\" \"Loman*\"   #Will show all HMP and Loman data products (don't download)
  curatedMetagenomicData -d \"*metaphlan*stool\"  #Will show all stool bug abundance tables (don't download)
Output: The file names of datasets to be created are returned to standard output.
"

input <- docopt(doc)
if(length(input$NAME) == 0 & !input$list)
    docopt(doc, args="-h")
if(input$list){
    input$NAME <- "*"
    input$dryrun <- TRUE
}
required.packages <- c("curatedMetagenomicData", "Biobase")
for (pkg in required.packages){
  if(!suppressPackageStartupMessages(require(pkg, character.only=TRUE))){
    stop(paste0("Make sure Bioconductor is installed (www.bioconductor.org/install), then type:\n
                BiocManager::install(\"", pkg, "\") \n
                from your R prompt."))
  }
}

requested.datasets <- input$NAME
all.datasets <- ls("package:curatedMetagenomicData")
all.datasets <-  grep("marker|gene|path|metaphlan_bugs", all.datasets, val=TRUE)

regex <- paste(glob2rx(requested.datasets), collapse="|")
matched.datasets <- grep(regex, all.datasets, value=TRUE)

message("Datasets to be downloaded:")
cat(paste0(matched.datasets, ".tsv\n"), file=stdout())

if(!any(matched.datasets %in% all.datasets))
  stop("NAME arguments do not match any available datasets.")

for (i in seq_along(matched.datasets)){
  if(input$dryrun) break
  message(paste0("Working on ", matched.datasets[i]))
  eset <- do.call(get(matched.datasets[i]), list())
  if(input$counts){
    edat <- round(sweep(exprs(eset), 2, eset$number_reads/100, "*"))
  }else{
    edat <- exprs(eset)
  }
  if(input$pdata){
    pdat <- t(as.matrix(pData(eset)))
    edat <- rbind(pdat, edat)
  }
  write.table(edat, file = paste0(matched.datasets[i], ".tsv"),
              col.names=NA, sep = "\t", quote = FALSE)
}