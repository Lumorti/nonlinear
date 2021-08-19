#!/bin/bash

# Settings for param 1
minParam1=0.1
maxParam1=0.9
stepsParam1=20

# Settings for param 2
minParam2=0.1
maxParam2=0.9
stepsParam2=20

# Calc things for param 1
deltaParam1=$(echo "scale=3;($maxParam1-$minParam1)/$stepsParam1" | bc)
param1=$minParam1
stepsParam1=$(echo "$stepsParam1+1" | bc)

# Calc things for param 2
deltaParam2=$(echo "scale=3;($maxParam2-$minParam2)/$stepsParam2" | bc)
param2=$minParam2
stepsParam2=$(echo "$stepsParam2+1" | bc)

# Clear the output file
> bench.log

# For each param1
param1=$minParam1
for i3 in $(seq 1 $stepsParam1)
do

	# For each param2
	param2=$minParam2
	for i4 in $(seq 1 $stepsParam2)
	do

		# Run the sum
		val=$(./nl -B -b $param1 -g $param2 -t 1000)

		# Output the results
		echo "$param1 $param2 $val" >> bench.log
		echo "$param1 $param2 $val"

		# Update param 1
		param2=$(echo "scale=3;$param2+$deltaParam2" | bc)

	done

	# Update param 2
	param1=$(echo "scale=3;$param1+$deltaParam1" | bc)

done

