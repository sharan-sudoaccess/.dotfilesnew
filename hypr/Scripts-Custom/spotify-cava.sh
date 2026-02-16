#!/bin/bash
sleep 3

playerctl --player=spotify --follow status | while read status; do
    if [[ "$status" == "Playing" ]]; then
        if ! pgrep -x cava > /dev/null; then
            kitty --class cava-term -e cava &
        fi
    else
        pkill -x cava 2>/dev/null
    fi
done
