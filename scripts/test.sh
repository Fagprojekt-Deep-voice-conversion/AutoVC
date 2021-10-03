#!/bin/sh

model_type="$1"
while getopts n: flag
do
    case "${flag}" in
        n) model_name=${OPTARG};;
    esac
done

echo "Type: $model_type"
echo "name: $model_name"


# while getopts u:a:f: flag
# do
#     case "${flag}" in
#         u) username=${OPTARG};;
#         a) age=${OPTARG};;
#         f) fullname=${OPTARG};;
#     esac
# done
# echo "Username: $username";
# echo "Age: $age";
# echo "Full Name: $fullname";