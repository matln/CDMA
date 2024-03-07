#!/bin/bash

# Copyright xmuspeech (Author:Snowdar 2019-02-17)

# This script creates a subset of data according to the utterance list file, see also kaldi/utils/subset_data_dir.sh

f=1 # field of utt-id in id-file
exclude=false
check=true

. parse_options.sh

if [[ $# != 3 ]]; then
	echo "[exit] Num of parameters is not equal to 3"
	echo "$0 [--check true|false] [--exclude false|true] [--f 1] <in-data-dir> <id-list> <out-data-dir>"
	exit 1
fi

indata=$1
idlist=$2
outdata=$3

[ ! -d "$indata" ] && echo "$0: line $LINENO: No such dir $indata" && exit 1
[ ! -f "$idlist" ] && echo "$0: line $LINENO: No such file $idlist" && exit 1
[ "$check" == "true" ] && [ -d "$outdata" ] && echo "$0: line $LINENO: $outdata is exist." && exit 1
mkdir -p "$outdata"

exclude_string=""
[[ "$exclude" == "true" ]] && exclude_string="--exclude"

for x in wav.scp utt2spk feats.scp utt2num_frames vad.scp utt2dur text utt2domain; do
  [ -f "$indata/$x" ] && awk -v f=$f '{print $f}' $idlist | filter_scp.pl $exclude_string - $indata/$x > $outdata/$x
done

if [ -f "${indata}/utt2domain" ]; then
	utt2spk_to_spk2utt.pl $outdata/utt2domain > $outdata/domain2utt
fi

# include: utt2spk_to_spk2utt.pl
fix_data_dir.sh $outdata || exit 1
validate_data_dir.sh --no-text --no-feats $outdata || exit 1

echo "Filter $outdata done."
