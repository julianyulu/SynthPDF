#!/bin/bash
img_dir='./outputs/imgs'
out_img='./outputs/raw_imgs'
out_mask='./outputs/masks'
out_annot='./outputs/annotates'


read -p "Copy or Move ? [c/m]: " how
while [ $how != "c" ] && [ $how != "m" ];do
    echo "Please input 'c' or 'm'"
    read -p "Copy or Move ? [c/m]: " how
done

# Create output folders
for x in $out_img $out_mask $out_annot;do
    if [ ! -d $x ]
    then
	mkdir $x
    fi
done


# copy files
if [ $how == 'c' ];then
    cmd="cp"
elif [ $how == 'm' ];then
    cmd="mv"
else
    echo "Method 'how' error!"
fi

for v in $(ls $img_dir);do
    id=$(echo $v |cut -d '_' -f 1)
    $cmd ${img_dir}/${id}_page1.jpg ${out_img}/${id}.jpg
    $cmd ${img_dir}/${id}_mask.jpg ${out_mask}/${id}_mask.jpg
    $cmd ${img_dir}/${id}_ann.jpg ${out_annot}/${id}_ann.jpg
    break
done

if [ $cmd == 'mv' ];then
    rm -r $img_dir
fi

echo "done !"

