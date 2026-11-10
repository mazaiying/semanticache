#!/bin/bash
# usage (on server): nohup bash queues/run_queue.sh q1_main > runs/queue_q1_main.log 2>&1 &
bash $HOME/sc2/queues/$1.sh
