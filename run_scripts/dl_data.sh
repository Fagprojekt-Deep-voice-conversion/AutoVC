#!/bin/sh

# for i in {0..137};
# do
#     mv data/HaegueYang_10sek/chooped${i}.wav data/HaegueYang_10sek/HaegueYang_${i}.wav
# done


HaegueYang_10sek="1FrmbSKv3TlEOuX_wGvMjonrGtUoUWVWA"


gdown $HaegueYang_10sek --id -O data/
unzip data/HaegueYang_10sek.zip
rm data/HaegueYang_10sek.zip
