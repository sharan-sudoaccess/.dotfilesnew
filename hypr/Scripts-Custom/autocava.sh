#!/bin/bash

# Name of the terminal process to look for
CAVA_BIN="cava"

while true; do
    # Check if Spotify (or any player) is "Playing"
    STATUS=$(playerctl status 2>/dev/null)

    if [ "$STATUS" = "Playing" ]; then
        # If playing and cava IS NOT running, start it
        if ! pgrep -x "$CAVA_BIN" > /dev/null; then
            # 'kitty --class floating_cava' opens it in a specific window class
            # so we can style it in Hyprland later
            kitty --class floating_cava -e cava & 
        fi
    else
        # If paused/stopped and cava IS running, kill it
        if pgrep -x "$CAVA_BIN" > /dev/null; then
            pkill -x "$CAVA_BIN"
        fi
    fi
    # Check every 2 seconds to save CPU
    sleep 2
done
