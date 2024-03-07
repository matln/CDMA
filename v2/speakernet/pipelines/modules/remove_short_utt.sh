#!/bin/bash
# Copyright 2022 Jianchen Li

limit_dur=0    # The spk whose duration < limit_dur will be removed. <=0 means no removing.

. parse_options.sh

if [[ $# != 2 ]]; then
  echo "[exit] Num of parameters is not equal to 2"
  echo "usage:$0 <data-dir> <length>"
  echo "[note] The utts whose duration less than length will be removed but keep the back-up so that you can recover while running this script with a small enough length."
  exit 1
fi

data=$1
len=$2

[ -f "$data"/spk2utt.backup ] && cp -f "$data"/spk2utt.backup "$data"/spk2utt

if [ $limit_dur -gt 0 ]; then
    [ ! -f "$data"/utt2dur ] && echo "$0: line $LINENO: no utt2dur file." && exit 1
fi
# printf 不会输出换行符
# 所有长度小于min_chunk的句子

if [ $limit_dur -gt 0 ]; then
    utt=$(awk -v len="$len" '{if($2<len) printf $1" "}' "${data}"/utt2dur)
fi

# 加上：去掉长度较短的句子后，所有dur小于limit_dur的spk对应的句子
# 有重复，但没关系
if [ "$limit_dur" -gt 0 ]; then
    utt=$utt$(echo $utt | awk -v limit=$limit_dur '
        ARGIND==1{for(i=1;i<=NF;i++){
            a[$i]=1;}}
        ARGIND==2{
            utt2dur[$1]=$2;
        }
        ARGIND==3{tot=0; for(i=2;i<=NF;i++){
            if(a[$i]!=1){
                tot=tot+utt2dur[$i];
            }}
            if(tot<limit){
                $1=""; print $0}
            }' - "$data"/utt2dur "$data"/spk2utt)
fi

list=$(echo "$utt" | sed 's/ /\n/g' | sed '/^$/d' | sort -u)
num=0
if [ "$list" != "" ]; then
  num=$(echo "$list" | wc -l | awk '{print $1}')
else
  echo "Nothing need to remove. It means that your datadir will be recovered fromm backup if you used this script before."
fi

echo -e "[$(echo $list)] $num utts here will be removed."

# Backup and Recover
for x in wav.scp utt2spk spk2utt utt2dur utt2domain; do
    [ -f "$data"/$x.backup ] && cp -f "$data"/$x.backup "$data"/$x
    [ ! -f "$data"/$x.backup ] && [ -f "$data"/$x ] && cp "$data"/$x "$data"/$x.backup
done

# Remove
# 这些都是以utt为索引的
for x in wav.scp utt2spk utt2dur utt2domain; do
    [ -f "$data"/$x ] && [ "$list" != "" ] && echo "$list" | awk '
        NR==FNR{a[$1]=1}NR>FNR{if(!a[$1]){print $0}}' - "$data"/$x > "$data"/$x.tmp && \
        mv -f "$data"/$x.tmp "$data"/$x
    echo "$data/$x done"
done

[ -f "$data"/utt2domain ] && utt2spk_to_spk2utt.pl $data/utt2domain >$data/domain2utt

fix_data_dir.sh "$data" || exit 1

# fix_data_dir.sh 调用了 utt2spk_to_spk2utt.pl 生成spk2utt
echo "$data/spk2utt done"

echo 'Remove invalid utts done.'
