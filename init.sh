#!/bin/bash
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_git

export PATH="/home2/sfgm68/miniconda3/bin:$PATH"