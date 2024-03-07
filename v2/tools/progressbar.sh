#!/bin/bash


bar::start() {
    split_inscp=$logdir/$(basename $inscp)
    total_steps=$(cat $inscp | wc -l)

    mkdir -p $logdir || exit 1
    split_scps=""
    for n in $(seq $nj); do
        split_scps="$split_scps ${split_inscp}.$n"
    done
    split_scp.pl $inscp $split_scps || exit 1;

    tmpfile=$(mktemp /tmp/$(basename $outscp).XXXXXX)
    steps_done=0

    # All log files
    _logs=
    for i in `seq 1 $nj`; do
        _logs="${_logs} ${outscp}.$i"
        touch ${outscp}.$i
    done

    # Get steps_done, write to a temporary file.
    # If you print the progressbar string directly in the while loop,
    # it cannot keep up with the speed of reading the line, as the 
    # floating point division and 'seq ... | sed' operations take a lot of time.
    tail -F -n 0 $_logs 2> /dev/null | stdbuf -oL grep $finished_mark | \
        {
        while read line; do
            (( steps_done += 1 ))
            echo $steps_done > $tmpfile 
        done
        }&
    tail_pid=$(jobs -p)
}

bar::stop() {
    run_pid=$(jobs -p | xargs | awk '{print $2}')

    # progressbar string
    {
    foreground="${foreground:-$(tput setaf 0)}" # Foreground can be set by the caller, defaults to black
    background="${background:-$(tput setab 2)}" # Background can be set by the caller, defaults to green
    reset_color="$(tput sgr0)"

    while [ $steps_done -lt $total_steps ]; do 
        steps_done=$([ -f $tmpfile ] && cat $tmpfile || break)
        while [[ $steps_done == "" ]]; do
            steps_done=$([ -f $tmpfile ] && cat $tmpfile || break)
        done

        percent=$(echo "$steps_done * 100 / ${total_steps}" | bc)
        # a sign '#' accounts for 2%
        sign_num=$(echo "$percent / 2" | bc)
        mark=$(seq -s "#" 0 $sign_num | sed -r 's/[0-9]+//g')
        remain_mark_num=$(echo "50 - $sign_num" | bc)
        mark="$mark""$(seq -s "." 0 $remain_mark_num | sed -r 's/[0-9]+//g')"
        printf -v progress_str "$(basename $outscp): [%3li%%]" $percent
        printf "%s: [%-50s] | %s/%s\r" "${background}${foreground}${progress_str}${reset_color}" \
            "${mark}" "${steps_done}" "${total_steps}"
    done
    }&
    bg_pid=$(jobs -p | xargs | awk '{print $3}')

    # Kill the processes
    # trap "kill $run_pid && kill $tail_pid $$ kill $bg_pid && \rm $tmpfile && exit 1" INT
    # wait $run_pid
    trap "${SUBTOOLS}/linux/kill_pid_tree.sh --show true  $run_pid $tail_pid $bg_pid && echo -e '\nAll killed' && \rm $tmpfile && exit 1" INT
    wait $run_pid

    [ $? != 0 ] && kill $tail_pid && kill $bg_pid && \rm "$tmpfile" && exit 1

    wait $bg_pid
    kill $tail_pid
    \rm "$tmpfile"

    echo -e ""
}
