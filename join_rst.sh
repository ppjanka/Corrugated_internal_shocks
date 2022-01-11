#!/bin/bash

if [ ! -d $1/joined_rst ]; then
    mkdir $1/joined_rst
fi
mv $1/id*/*.rst $1/joined_rst