#!/bin/bash

echo "gen mu-mimo packet start"

echo "get channel from station 0"

pscp -pw 7777 -r cloud@192.168.10.107:/home/cloud/sdr/cmu-chan0.bin /home/cloud/sdr/

echo "get channel from station 1"

pscp -pw 7777 -r cloud@192.168.10.202:/home/cloud/sdr/cmu-chan1.bin /home/cloud/sdr/

echo "run the python to generate mu-mimo signal"

python3 /home/cloud/sdr/gr-ieee80211/tools/cmu-ap.py