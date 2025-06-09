#!/bin/bash

# Generate a random 32 characters alphanumeric token and output it
token=$(LC_ALL=C cat /dev/urandom | LC_ALL=C tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
echo $token
