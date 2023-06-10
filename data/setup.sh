#!/bin/bash
# this program is used to unzip data in the project directory

# This is unecessary yet
# echo "Extracting and ordering project data (animals)..."
# rm -rf animals/
# rm -f animal_names.txt
# unzip -q zips/animals.zip
# echo "Extracting and ordering project data (fruits)..."
# rm -rf fruits/
# rm -f fruit_names.txt
# unzip -q zips/fruits.zip


echo "Extracting and ordering project data (custom)..."
rm -rf custom/
rm -f custom_names.txt
unzip -q zips/custom.zip
echo "End of data extraction program"
