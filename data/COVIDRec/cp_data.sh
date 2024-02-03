datapath=/lus/eagle/clone/g2/projects/datascience/avasan/Data/Data_Docking/1M-flatten
current_path=.
array=($(ls ${currentpath}))
echo $array
for x in "${array[@]}"
do	
    #echo ${datapath}/${x}
    if [ -d ./${x} ]; then
        cp ${datapath}/ml.${x}.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet.xform-smiles.csv.reg ./${x}/
    fi
    #printf "%s\n" "$x"
done
