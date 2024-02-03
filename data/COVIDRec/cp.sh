datapath=/lus/eagle/clone/g2/projects/datascience/avasan/DockingSurrogates/Benchmarks_ST_Publication/ST_Revised_Train_multiReceptors/
array=($(ls ${datapath}))
echo $array
for x in "${array[@]}"
do	
    echo ${datapath}/${x}
    if [ -d ${datapath}/${x} ]; then
        if [ -d ./${x} ]; then
            cp ${datapath}/${x}/model.weights.h5 ./${x}/
        fi
    fi
    #printf "%s\n" "$x"
done
