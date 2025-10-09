# for english use specific scripts
# for lang in german dutch french italian polish; do
for lang in spanish portuguese; do
    for split in train dev test; do
        qsub -v MLS_LANGUAGE=$lang,SPLIT=$split scripts/pbs/cleanse/mls/cleanse_language.sh
    done
done